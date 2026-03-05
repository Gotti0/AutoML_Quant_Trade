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
                 entry_z: float = -2.0, exit_z: float = 1.0):
        super().__init__(name="SwingMeanReversion")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.entry_z = entry_z
        # BUG-4 FIX: exit_z 기본값 0.0 → 1.0 (과잉 매도 시그널 방지)
        self.exit_z = exit_z

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        self._record(event)
        ticker = event.ticker

        # BUG-1 FIX: 종목별 독립 종가 시계열 사용
        closes = self.get_close_series(ticker)

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
        # BUG-4 FIX: exit_z=1.0으로 상향하여 노이즈 필터링
        # 보유 포지션 없으면 event_loop의 _signal_to_order에서 held=0으로 차단됨
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
