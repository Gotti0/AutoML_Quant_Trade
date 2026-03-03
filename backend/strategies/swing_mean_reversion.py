"""
AutoML_Quant_Trade - 스윙 평균 회귀 전략

볼린저 밴드 Z-스코어 이탈 복귀 기반.
"""
from typing import Optional

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy


class SwingMeanReversion(BaseStrategy):
    """볼린저 밴드 Z-스코어 이탈 복귀 전략"""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 entry_z: float = -2.0, exit_z: float = 0.0):
        super().__init__(name="SwingMeanReversion")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.entry_z = entry_z
        self.exit_z = exit_z

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        self._record(event)

        closes = self.get_close_series()

        # 워밍업
        if len(closes) < self.bb_period:
            return None

        # 볼린저 밴드 계산 (과거 방향 rolling만)
        window = closes[-self.bb_period:]
        sma = sum(window) / self.bb_period
        variance = sum((x - sma) ** 2 for x in window) / self.bb_period
        std = variance ** 0.5

        if std == 0:
            return None

        # Z-스코어
        z_score = (event.close - sma) / std

        # 매수: Z-스코어가 entry_z 아래 (과매도)
        if z_score < self.entry_z:
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="BUY",
                strength=min(1.0, abs(z_score) / 3.0),
                strategy_name=self.name,
            )

        # 매도: Z-스코어가 exit_z 위 (평균 회귀 완료)
        if z_score > self.exit_z:
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="SELL",
                strength=min(1.0, z_score / 2.0),
                strategy_name=self.name,
            )

        return None

    def get_timeframe(self) -> str:
        return "daily"
