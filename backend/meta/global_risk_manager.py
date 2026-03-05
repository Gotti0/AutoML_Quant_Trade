"""
AutoML_Quant_Trade - 글로벌 리스크 관리자

전체 포트폴리오 레벨의 리스크 관리:
  1. 전체 NAV 기준 서킷 브레이커
  2. 국면별 최대 레버리지 제한
  3. 긴급 대피(Emergency Evacuation) 로직
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from backend.config.settings import Settings
from backend.engine.events import SignalEvent
from backend.engine.ledger import MasterLedger

logger = logging.getLogger(__name__)


class GlobalRiskManager:
    """전체 포트폴리오 레벨 리스크 관리"""

    def __init__(self,
                 max_portfolio_drawdown: float = 0.15,
                 crash_regime_threshold: float = 0.6,
                 emergency_cash_ratio: float = 0.5):
        """
        Parameters:
            max_portfolio_drawdown: 전체 포트폴리오 MDD 한도
            crash_regime_threshold: Crash 국면 확률이 이 이상이면 긴급 모드
            emergency_cash_ratio: 긴급 대피 시 현금화 비율
        """
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.crash_threshold = crash_regime_threshold
        self.emergency_cash_ratio = emergency_cash_ratio

        self._peak_nav: float = 0.0
        self._is_emergency: bool = False
        self._circuit_broken: bool = False
        # BUG-5 FIX: 쿨다운 기간 (서킷브레이커 발동 후 5영업일 거래 금지)
        self.cooldown_period: int = 5
        self._days_since_broken: int = 0

    def check_portfolio_risk(self, ledger: MasterLedger,
                              market_prices: Dict[str, float]) -> Dict:
        """
        전체 포트폴리오 리스크 체크.

        Returns:
            {
                "status": "normal" | "warning" | "circuit_broken",
                "current_drawdown": float,
                "nav": float,
            }
        """
        nav = ledger.get_total_equity(market_prices)

        # 고점 갱신
        self._peak_nav = max(self._peak_nav, nav)

        # 현재 낙폭
        drawdown = 0.0
        if self._peak_nav > 0:
            drawdown = (nav - self._peak_nav) / self._peak_nav

        # 서킷 브레이커
        if drawdown < -self.max_portfolio_drawdown:
            liquidation_signals: List[SignalEvent] = []
            # 연속적인 로깅 방지: 이미 서킷브레이크 상태라면 로깅 안 함
            if not self._circuit_broken:
                self._circuit_broken = True
                self._days_since_broken = 0
                logger.warning(
                    f"🚨 PORTFOLIO CIRCUIT BREAKER: "
                    f"drawdown {drawdown:.2%} < limit -{self.max_portfolio_drawdown:.2%}"
                )
                # BUG-5 FIX: 전 포지션 강제 청산 시그널 생성
                liquidation_signals = self._generate_liquidation_signals(ledger)
            return {
                "status": "circuit_broken",
                "current_drawdown": drawdown,
                "nav": nav,
                "liquidation_signals": liquidation_signals,
            }

        # 경고 (한도의 70% 이상)
        if drawdown < -self.max_portfolio_drawdown * 0.7:
            return {
                "status": "warning",
                "current_drawdown": drawdown,
                "nav": nav,
            }

        return {
            "status": "normal",
            "current_drawdown": drawdown,
            "nav": nav,
        }

    def check_regime_risk(self, regime_probs: np.ndarray) -> Dict:
        """
        국면 기반 리스크 체크.

        Crash 국면 확률이 임계값 이상이면 긴급 대피 모드 진입.

        Parameters:
            regime_probs: γ(t) = [p_bull, p_bear, p_crash]
        Returns:
            {"emergency": bool, "crash_prob": float}
        """
        # Crash = index 2 (3국면 기준)
        crash_prob = float(regime_probs[-1]) if len(regime_probs) >= 3 else 0.0

        if crash_prob >= self.crash_threshold:
            if not self._is_emergency:
                logger.warning(
                    f"⚠️ EMERGENCY MODE: Crash probability {crash_prob:.2%} "
                    f">= threshold {self.crash_threshold:.2%}"
                )
            self._is_emergency = True
        else:
            if self._is_emergency:
                logger.info(f"Emergency mode deactivated: crash_prob={crash_prob:.2%}")
            self._is_emergency = False

        return {
            "emergency": self._is_emergency,
            "crash_prob": crash_prob,
        }

    def get_position_limit_multiplier(self, regime_probs: np.ndarray) -> float:
        """
        국면에 따른 포지션 한도 배수.

        - Bull: 1.0 (풀 포지션)
        - Bear: 0.7 (30% 감축)
        - Crash: 0.3 (70% 감축)
        """
        if self._circuit_broken:
            return 0.0  # 서킷 브레이커 → 전면 거래 중단

        if self._is_emergency:
            return 1.0 - self.emergency_cash_ratio  # 긴급 대피

        # 가중 평균 배수
        multipliers = [1.0, 0.7, 0.3]  # Bull, Bear, Crash
        n = min(len(regime_probs), len(multipliers))
        return float(np.dot(regime_probs[:n], multipliers[:n]))

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    @property
    def is_emergency(self) -> bool:
        return self._is_emergency

    def update_cooldown(self):
        """일 마감 시 호출: 쿨다운 카운터 증가 및 자동 해제."""
        if self._circuit_broken:
            self._days_since_broken += 1
            if self._days_since_broken > self.cooldown_period:
                self._circuit_broken = False
                self._days_since_broken = 0
                logger.info("🔓 Circuit breaker released after cooldown period.")

    def _generate_liquidation_signals(self, ledger: MasterLedger) -> List[SignalEvent]:
        """모든 서브 계정의 보유 포지션에 대해 강제 청산 시그널 생성."""
        signals: List[SignalEvent] = []
        for engine_name, account in ledger.sub_accounts.items():
            for ticker, qty in account.positions.items():
                if qty > 0:
                    signals.append((
                        engine_name,
                        SignalEvent(
                            timestamp=0,  # 루프에서 현재 timestamp로 덮어씌워짐
                            ticker=ticker,
                            direction="SELL",
                            strength=1.0,
                            strategy_name="CircuitBreaker",
                        )
                    ))
        logger.warning(f"Generated {len(signals)} forced liquidation signals.")
        return signals

    def reset(self):
        """상태 초기화."""
        self._peak_nav = 0.0
        self._is_emergency = False
        self._circuit_broken = False
        self._days_since_broken = 0
