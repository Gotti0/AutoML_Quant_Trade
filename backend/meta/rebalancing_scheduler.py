"""
AutoML_Quant_Trade - 메타 리밸런싱 스케줄러

HMM 국면 전이 감지 시 또는 주기적(월간/주간)으로
엔진별 자본을 동적으로 재배치.
"""
import logging
from typing import Dict, Optional

import numpy as np

from backend.meta.capital_allocator import CapitalAllocator
from backend.engine.ledger import MasterLedger

logger = logging.getLogger(__name__)


class RebalancingScheduler:
    """메타 리밸런싱 스케줄러"""

    def __init__(self,
                 allocator: CapitalAllocator = None,
                 rebalance_freq: str = "M",
                 regime_change_trigger: bool = True,
                 min_interval_days: int = 5):
        """
        Parameters:
            allocator: HMM 기반 자본 배분기
            rebalance_freq: 정기 리밸런싱 주기 ("M"=월간, "W"=주간)
            regime_change_trigger: 국면 전이 시 즉시 리밸런싱 여부
            min_interval_days: 리밸런싱 간 최소 간격 (일)
        """
        self.allocator = allocator or CapitalAllocator()
        self.rebalance_freq = rebalance_freq
        self.regime_change_trigger = regime_change_trigger
        self.min_interval_days = min_interval_days

        self._last_regime: Optional[int] = None
        self._last_rebalance_date: Optional[int] = None
        self._rebalance_history: list = []

    def should_rebalance(self, current_date: int,
                         regime_probs: np.ndarray) -> bool:
        """
        리밸런싱 시점인지 판단.

        트리거 조건:
          1. 국면 전이 감지 (regime_change_trigger=True일 때)
          2. 정기 주기 도래 (월초/주초)
          3. 최소 간격 이상 경과

        Parameters:
            current_date: 현재 날짜 (YYYYMMDD)
            regime_probs: γ(t) 벡터
        Returns:
            리밸런싱 필요 여부
        """
        # 최소 간격 체크
        if self._last_rebalance_date is not None:
            days_since = current_date - self._last_rebalance_date
            if days_since < self.min_interval_days:
                return False

        # 1. 국면 전이 트리거
        current_regime = int(np.argmax(regime_probs))
        if self.regime_change_trigger and self._last_regime is not None:
            if current_regime != self._last_regime:
                logger.info(
                    f"Regime change detected: {self._last_regime} → {current_regime} "
                    f"at {current_date}"
                )
                return True

        # 2. 정기 리밸런싱 (월초)
        if self.rebalance_freq == "M":
            day_of_month = current_date % 100
            if day_of_month <= 3:  # 매월 1~3일
                if self._last_rebalance_date is None:
                    return True
                last_month = (self._last_rebalance_date // 100) % 100
                current_month = (current_date // 100) % 100
                if current_month != last_month:
                    return True

        # 3. 정기 리밸런싱 (주초)
        elif self.rebalance_freq == "W":
            if self._last_rebalance_date is None:
                return True
            if current_date - self._last_rebalance_date >= 7:
                return True

        return False

    def execute_rebalance(self, ledger: MasterLedger,
                          regime_probs: np.ndarray,
                          current_date: int,
                          market_prices: Dict[str, float]):
        """
        리밸런싱 실행: 목표 배분 계산 → 서브 계정 간 자본 이동.
        
        주의: execute_rebalance 에 들어왔다는 것은 이미 should_rebalance() 가 True 였음을 뜻합니다.

        Parameters:
            ledger: 마스터 원장
            regime_probs: γ(t) 벡터
            current_date: 현재 날짜
            market_prices: 현재 시장 가격
        """
        # 0. 동적 비중 최적화 (PyTorch HRP) - 리밸런싱 당시에만 갱신
        if hasattr(self.allocator, 'update_dynamic_weights'):
            engines_order = list(self.allocator.w_star.keys())
            returns_tensor = ledger.get_subaccount_returns_tensor(window=60, engines_order=engines_order)
            if returns_tensor is not None and len(returns_tensor) >= 2:
                from backend.meta.pytorch_hrp import TorchHRPOptimizer
                hrp = TorchHRPOptimizer()
                weights = hrp.calculate_hrp_weights(returns_tensor)
                
                new_w_star = {
                    engine: weights[i].item()
                    for i, engine in enumerate(engines_order)
                }
                self.allocator.update_dynamic_weights(new_w_star)

        # 1. 목표 배분 계산
        target = self.allocator.calculate_target(regime_probs)

        # 2. 현재 에퀴티 조회
        total_equity = ledger.get_total_equity(market_prices)
        current_equity = {}
        for name, acc in ledger.sub_accounts.items():
            current_equity[name] = acc.get_equity(market_prices)

        # 3. 리밸런싱 주문 계산
        orders = self.allocator.calculate_rebalance_orders(
            current_equity, target, total_equity
        )

        # 4. 서브 계정 간 현금 이동 (간단한 현금 전송)
        for engine, amount in orders.items():
            if amount == 0:
                continue
            if engine in ledger.sub_accounts:
                ledger.sub_accounts[engine].cash += amount

        # 5. 상태 갱신
        current_regime = int(np.argmax(regime_probs))
        self._last_regime = current_regime
        self._last_rebalance_date = current_date

        # 6. 이력 기록
        self._rebalance_history.append({
            "date": current_date,
            "regime": self.allocator.get_dominant_regime(regime_probs),
            "regime_probs": regime_probs.tolist(),
            "target": target,
            "orders": orders,
            "total_equity": total_equity,
        })

        logger.info(
            f"Rebalanced at {current_date}: "
            f"regime={self.allocator.get_dominant_regime(regime_probs)}, "
            f"equity={total_equity:,.0f}"
        )

    def update_regime(self, regime_probs: np.ndarray):
        """국면 추적 업데이트 (리밸런싱 없이)."""
        self._last_regime = int(np.argmax(regime_probs))

    @property
    def rebalance_count(self) -> int:
        return len(self._rebalance_history)

    @property
    def history(self) -> list:
        return self._rebalance_history
