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
    ):
        super().__init__(ledger, transaction_model)
        self.regime_model = regime_model
        self.allocator = allocator or CapitalAllocator()
        self.grm = global_risk_manager or GlobalRiskManager()
        self.scheduler = scheduler or RebalancingScheduler(allocator=self.allocator)
        self.fe = FeatureEngineer()

        # 가격 이력 누적 버퍼 (피처 추출용)
        self._price_history: Dict[str, list] = {}
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

        while not pq.empty():
            _, _, event = pq.get()
            event_count += 1

            # 현재 가격 갱신
            self.current_prices[event.ticker] = event.close

            # ─── 가격 이력 누적 ───
            if event.ticker not in self._price_history:
                self._price_history[event.ticker] = []
            self._price_history[event.ticker].append({
                "date": event.timestamp,
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
            })

            # ─── 글로벌 리스크 체크 ───
            risk_status = self.grm.check_portfolio_risk(
                self.ledger, self.current_prices
            )
            if risk_status["status"] == "circuit_broken":
                # 서킷 브레이커 발동 → 시그널 전달 차단
                self.ledger.record_equity(event.timestamp, self.current_prices)
                continue

            # ─── 대기 시그널 체결 (이전 봉 시그널 → 현재 봉 시가) ───
            self._process_pending_signals(event)

            # ─── 전략에 MarketEvent 전달 ───
            # 서킷 브레이커가 걸려있지 않은 상태에서만 시그널 수집
            for engine_name, strategy_list in self.strategies.items():
                for strategy in strategy_list:
                    signal = strategy.on_market_data(event)
                    if signal is not None:
                        self._pending_signals.append((engine_name, signal))

            # ─── 메타 레이어: HMM 국면 추론 + 리밸런싱 ───
            self._meta_step(event)

            # 에퀴티 스냅샷
            self.ledger.record_equity(event.timestamp, self.current_prices)

            if event_count % 1000 == 0:
                logger.info(f"  Progress: {event_count}/{total_events}")

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

    def _meta_step(self, event):
        """
        메타 레이어 처리: HMM 국면 추론 → 리밸런싱 판단 → 실행.
        HMM 모델이 주입되지 않은 경우 균등 확률로 폴백한다.
        """
        # HMM 국면 추론
        regime_probs = self._infer_regime(event.ticker)

        if regime_probs is None:
            return  # 데이터 부족

        self._last_regime_probs = regime_probs

        # 국면 기반 긴급 대피 체크
        regime_risk = self.grm.check_regime_risk(regime_probs)

        # 리밸런싱 시점 판단
        if self.scheduler.should_rebalance(event.timestamp, regime_probs):
            self.scheduler.execute_rebalance(
                ledger=self.ledger,
                regime_probs=regime_probs,
                current_date=event.timestamp,
                market_prices=self.current_prices,
            )
        else:
            self.scheduler.update_regime(regime_probs)

    def _infer_regime(self, ticker: str) -> Optional[np.ndarray]:
        """
        현재까지 누적된 가격 이력으로 HMM 국면 확률 벡터를 추론.
        RegimeHMM 모델이 없으면 균등 확률([0.33, 0.33, 0.34])을 반환.
        """
        history = self._price_history.get(ticker)
        if history is None or len(history) < 65:
            # 최소 65일 이상 데이터가 있어야 피처 추출 가능
            return None

        if self.regime_model is None:
            # 모델이 없으면 균등 확률 폴백
            n = self.allocator.n_regimes
            return np.ones(n) / n

        # 피처 추출
        df = pd.DataFrame(history)
        features = self.fe.extract(df)
        if features.empty:
            return None

        # 마지막 시점의 확률만 사용
        try:
            proba = self.regime_model.predict_proba(features)
            return proba[-1]  # shape=(n_regimes,)
        except Exception as e:
            logger.warning(f"Regime inference error: {e}")
            n = self.allocator.n_regimes
            return np.ones(n) / n

    @property
    def last_regime_probs(self) -> Optional[np.ndarray]:
        return self._last_regime_probs
