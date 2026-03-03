"""
AutoML_Quant_Trade - HMM 기반 동적 자본 배분기

HMM 국면 확률 벡터 γ(t) × 국면별 타깃 가중치 W* → 엔진별 목표 자본 비중 산출.

설계 문서 참조:
  w_engine(t) = Σ_regime  γ(t, regime) × W*(engine, regime)
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """HMM 확률 벡터 기반 동적 엔진 자본 배분"""

    # 국면별 타깃 가중치 W* ─ 각 엔진이 각 국면에서 받는 자본 비중
    # 행: 엔진, 열: [Bull, Bear, Crash]
    W_STAR: Dict[str, List[float]] = {
        "MidFreq":   [0.15, 0.25, 0.10],  # 중빈도: 변동성 높으면 활발
        "Swing":     [0.35, 0.15, 0.00],  # 스윙: 강세장에서 활발, 폭락시 비활성
        "MidShort":  [0.35, 0.15, 0.10],  # 중단기: 트렌드 활용
        "Long_Safe": [0.15, 0.45, 0.80],  # 장기 안전: 약세/폭락 시 방어
    }

    def __init__(self, w_star: Dict[str, List[float]] = None,
                 n_regimes: int = 3):
        """
        Parameters:
            w_star: 커스텀 타깃 가중치 (기본: W_STAR)
            n_regimes: 국면 수
        """
        self.w_star = w_star or self.W_STAR
        self.n_regimes = n_regimes

        # 가중치 정합성 검증
        self._validate_weights()

    def _validate_weights(self):
        """각 국면에서 전체 엔진 비중의 합 = 1.0 검증"""
        for regime_idx in range(self.n_regimes):
            total = sum(
                weights[regime_idx]
                for weights in self.w_star.values()
            )
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    f"Regime {regime_idx}: W* column sum = {total:.3f} "
                    f"(expected 1.0)"
                )

    def calculate_target(self, regime_probs: np.ndarray) -> Dict[str, float]:
        """
        HMM 확률 벡터로부터 엔진별 목표 자본 비중 산출.

        w_engine(t) = Σ_regime  γ(t, regime) × W*(engine, regime)

        Parameters:
            regime_probs: γ(t) 벡터, shape=(n_regimes,), 합=1.0
                          예: [0.7, 0.2, 0.1] → 70% Bull, 20% Bear, 10% Crash
        Returns:
            엔진별 비중: {"MidFreq": 0.18, "Swing": 0.28, ...}
        """
        if len(regime_probs) != self.n_regimes:
            raise ValueError(
                f"Expected {self.n_regimes} regime probabilities, "
                f"got {len(regime_probs)}"
            )

        target = {}
        for engine, weights in self.w_star.items():
            # γ(t) · W*(engine,:)  = 가중 합
            target[engine] = float(np.dot(regime_probs, weights))

        # 비중 합 정규화 (부동소수점 오차 보정)
        total = sum(target.values())
        if total > 0:
            target = {k: v / total for k, v in target.items()}

        logger.debug(f"Target allocation: {target}")
        return target

    def calculate_rebalance_orders(self,
                                    current_equity: Dict[str, float],
                                    target_allocation: Dict[str, float],
                                    total_equity: float,
                                    threshold: float = 0.03
                                    ) -> Dict[str, float]:
        """
        현재 에퀴티와 목표 배분의 차이를 계산하여 리밸런싱 금액 산출.

        Parameters:
            current_equity: 엔진별 현재 에퀴티
            target_allocation: 엔진별 목표 비중 (합=1.0)
            total_equity: 전체 NAV
            threshold: 리밸런싱 임계값 (비중 차이 < threshold이면 스킵)
        Returns:
            엔진별 리밸런싱 금액 (양수=추가 배정, 음수=회수)
        """
        orders = {}

        for engine, target_weight in target_allocation.items():
            current = current_equity.get(engine, 0)
            target_value = total_equity * target_weight
            diff = target_value - current
            diff_ratio = abs(diff) / total_equity if total_equity > 0 else 0

            # 임계값 미만이면 스킵 (무의미한 소규모 리밸런싱 방지)
            if diff_ratio < threshold:
                orders[engine] = 0.0
            else:
                orders[engine] = diff

        logger.info(
            f"Rebalance orders: "
            + ", ".join(f"{k}={v:+,.0f}" for k, v in orders.items() if v != 0)
        )

        return orders

    def get_dominant_regime(self, regime_probs: np.ndarray) -> str:
        """확률이 가장 높은 국면명 반환."""
        regime_names = ["Bull", "Bear", "Crash"]
        idx = int(np.argmax(regime_probs))
        if idx < len(regime_names):
            return regime_names[idx]
        return f"Regime_{idx}"
