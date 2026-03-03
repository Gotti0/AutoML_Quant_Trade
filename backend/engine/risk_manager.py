"""
AutoML_Quant_Trade - 리스크 관리자

서브 엔진별 MDD 한도 초과 시 서킷 브레이커 발동.
"""
import logging
from typing import Dict

from backend.config.settings import Settings
from backend.engine.ledger import SubEngineAccount

logger = logging.getLogger(__name__)


class RiskManager:
    """서브 엔진별 리스크 관리 — MDD 기반 서킷 브레이커"""

    def __init__(self, loss_limits: Dict[str, float] = None):
        """
        Parameters:
            loss_limits: {engine_name: MDD 한도} (예: {"MidFreq": 0.05})
        """
        self.loss_limits = loss_limits or Settings.ENGINE_LOSS_LIMITS
        self.circuit_broken: Dict[str, bool] = {
            name: False for name in self.loss_limits
        }
        self.peak_equity: Dict[str, float] = {}

    def check(self, engine_name: str, account: SubEngineAccount,
              market_prices: Dict[str, float]) -> bool:
        """
        리스크 체크: 서킷 브레이커 발동 여부 확인.

        Parameters:
            engine_name: 서브 엔진명
            account: 해당 서브 계정
            market_prices: 현재 시장 가격
        Returns:
            True = 거래 가능, False = 서킷 브레이커 발동
        """
        if engine_name not in self.loss_limits:
            return True

        if self.circuit_broken.get(engine_name, False):
            return False

        equity = account.get_equity(market_prices)

        # 고점 갱신
        if engine_name not in self.peak_equity:
            self.peak_equity[engine_name] = equity
        else:
            self.peak_equity[engine_name] = max(
                self.peak_equity[engine_name], equity
            )

        peak = self.peak_equity[engine_name]
        if peak <= 0:
            return True

        # 현재 낙폭
        drawdown = (equity - peak) / peak  # 음수

        # MDD 한도 초과 시 서킷 브레이커
        limit = self.loss_limits[engine_name]
        if drawdown < -limit:
            self.circuit_broken[engine_name] = True
            logger.warning(
                f"⚠️ CIRCUIT BREAKER: {engine_name} "
                f"drawdown {drawdown:.2%} exceeded limit -{limit:.2%}"
            )
            return False

        return True

    def reset(self, engine_name: str = None):
        """서킷 브레이커 해제."""
        if engine_name:
            self.circuit_broken[engine_name] = False
            self.peak_equity.pop(engine_name, None)
        else:
            for name in self.circuit_broken:
                self.circuit_broken[name] = False
            self.peak_equity.clear()
