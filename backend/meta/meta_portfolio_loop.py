"""
AutoML_Quant_Trade - 메타 포트폴리오 백테스팅 루프

BacktestEventLoop를 확장하여, 매 이벤트 처리 시
HMM 국면 추론 → 글로벌 리스크 체크 → 동적 자본 재배치를
자동으로 수행하는 통합 루프.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.engine.event_loop import BacktestEventLoop
from backend.engine.ledger import MasterLedger
from backend.engine.transaction_model import TransactionModel
from backend.meta.capital_allocator import CapitalAllocator
from backend.meta.global_risk_manager import GlobalRiskManager
from backend.meta.rebalancing_scheduler import RebalancingScheduler
from backend.models.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class MetaPortfolioLoop(BacktestEventLoop):
    """
    메타 포트폴리오 백테스팅 루프.

    기존 BacktestEventLoop의 run()을 확장하여:
    1. 매 이벤트마다 누적 가격 이력으로 피처 추출
    2. HMM 국면 확률 벡터 γ(t) 추론
    3. GlobalRiskManager로 포트폴리오 레벨 리스크 체크
    4. RebalancingScheduler로 리밸런싱 시점 판단 및 실행
    """

    def __init__(
        self,
        ledger: MasterLedger,
        transaction_model: TransactionModel = None,
        regime_model=None,
        allocator: CapitalAllocator = None,
        global_risk_manager: GlobalRiskManager = None,
        scheduler: RebalancingScheduler = None,
        precomputed_regimes: Dict[int, np.ndarray] = None,
    ):
        super().__init__(ledger, transaction_model)
        self.regime_model = regime_model
        self.allocator = allocator or CapitalAllocator()
        self.grm = global_risk_manager or GlobalRiskManager()
        self.scheduler = scheduler or RebalancingScheduler(allocator=self.allocator)
        
        # O(1) 조회를 위한 사전 계산된 국면 사전
        self.precomputed_regimes = precomputed_regimes or {}
        
        self._last_regime_probs: Optional[np.ndarray] = None

    def run(self, market_data: Dict[str, pd.DataFrame],
            timeframe: str = "daily") -> pd.DataFrame:
        """
        메타 포트폴리오 통합 백테스팅 실행.

        기존 BacktestEventLoop.run()의 이벤트 처리 루프에
        메타 레이어 훅을 삽입한 오버라이드 구현.
        """
        from queue import PriorityQueue

        pq = self._build_event_queue(market_data, timeframe)
        total_events = pq.qsize()
        logger.info(
            f"MetaPortfolioLoop started: {total_events} events, "
            f"{len(market_data)} tickers"
        )

        event_count = 0
        current_timestamp = None

        while not pq.empty():
            ts, _, event = pq.get()
            event_count += 1

            if current_timestamp is None:
                current_timestamp = ts

            # ─── 날짜(Timestamp)가 바뀌었을 때 (일일 마감 처리) ───
            if ts != current_timestamp:
                self._end_of_bar_processing(current_timestamp)
                current_timestamp = ts

            # ─── 단일 종목 이벤트 처리 (장중/장마감 틱) ───
            # 서킷 브레이커가 발동되지 않은 평상시에만 개별 종목 매매 진행
            if not self.grm.is_circuit_broken:
                self._process_pending_signals(event)

            # 현재 가격 갱신 (항상 최신화)
            self.current_prices[event.ticker] = event.close

            if not self.grm.is_circuit_broken:
                for engine_name, strategy_list in self.strategies.items():
                    for strategy in strategy_list:
                        signal = strategy.on_market_data(event)
                        if signal is not None:
                            self._pending_signals.append((engine_name, signal))

            if event_count % 500000 == 0:
                logger.info(f"  Progress: {event_count}/{total_events}")

        # 마지막 데이터 마감 처리
        if current_timestamp is not None:
            self._end_of_bar_processing(current_timestamp)

        # 결과
        metrics = self.ledger.get_performance_metrics()
        logger.info(f"MetaPortfolioLoop complete: {event_count} events processed")
        if metrics:
            logger.info(
                f"  Return: {metrics.get('total_return', 0):.2%}, "
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"  MDD: {metrics.get('max_drawdown', 0):.2%}"
            )

        rebal_count = self.scheduler.rebalance_count
        logger.info(f"  Rebalancing events: {rebal_count}")

        return self.ledger.get_equity_curve()

    def _end_of_bar_processing(self, timestamp: int):
        """
        특정 날짜의 모든 틱 처리가 끝난 뒤(종가 기준) 포트폴리오 메타 레이어 가동.
        하루에 단 1번만 실행되어 과도한 스냅샷 및 연산 병목 방지.
        """
        # 1. 글로벌 리스크 체크
        risk_status = self.grm.check_portfolio_risk(
            self.ledger, self.current_prices
        )
        
        # 2. 서킷 브레이커가 아닐 때만 국면 추론 및 리밸런싱 판단
        if risk_status["status"] != "circuit_broken":
            self._meta_step(timestamp)

        # 3. 일 마감 에퀴티 스냅샷 (단 1회)
        self.ledger.record_equity(timestamp, self.current_prices)

    def _meta_step(self, timestamp: int):
        """
        메타 레이어 처리: 사전 계산된 HMM 국면 조회 → 리밸런싱 판단 → 실행.
        """
        # HMM 국면 모델 O(1) 메모리 참조
        n = self.allocator.n_regimes
        default_probs = np.ones(n) / n
        regime_probs = self.precomputed_regimes.get(timestamp, default_probs)

        self._last_regime_probs = regime_probs

        # 국면 기반 긴급 대피 체크
        self.grm.check_regime_risk(regime_probs)

        # 리밸런싱 시점 판단
        if self.scheduler.should_rebalance(timestamp, regime_probs):
            self.scheduler.execute_rebalance(
                ledger=self.ledger,
                regime_probs=regime_probs,
                current_date=timestamp,
                market_prices=self.current_prices,
            )
        else:
            self.scheduler.update_regime(regime_probs)

    @property
    def last_regime_probs(self) -> Optional[np.ndarray]:
        return self._last_regime_probs
