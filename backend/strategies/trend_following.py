"""
AutoML_Quant_Trade - 중단기 트렌드 팔로잉 전략

이중 이동평균선 교차(골든/데드 크로스) + 모멘텀 확인.
"""
from typing import Dict, Optional

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy


class TrendFollowing(BaseStrategy):
    """이중 이동평균선 교차 + 모멘텀 전략"""

    def __init__(self, fast_period: int = 20, slow_period: int = 50,
                 momentum_period: int = 10):
        super().__init__(name="TrendFollowing")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.momentum_period = momentum_period
        # BUG-1 FIX: 종목별로 이전 크로스 상태 분리 추적
        self._prev_fast_above: Dict[str, Optional[bool]] = {}

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        self._record(event)
        ticker = event.ticker

        closes = self.get_close_series(ticker)

        # 워밍업
        if len(closes) < self.slow_period:
            return None

        # 이동평균 계산 (과거 방향만)
        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes[-self.slow_period:]) / self.slow_period

        fast_above = fast_ma > slow_ma

        # 모멘텀 확인 (최근 N일 수익률)
        if len(closes) >= self.momentum_period + 1:
            momentum = (closes[-1] / closes[-self.momentum_period - 1]) - 1
        else:
            momentum = 0.0

        signal = None

        prev = self._prev_fast_above.get(ticker)

        # 골든 크로스: 단기 이평이 장기 이평 상향 돌파 + 양의 모멘텀
        if prev is False and fast_above and momentum > 0:
            signal = SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="BUY",
                strength=min(1.0, abs(momentum) * 10),
                strategy_name=self.name,
            )

        # 데드 크로스: 단기 이평이 장기 이평 하향 돌파
        elif prev is True and not fast_above:
            signal = SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="SELL",
                strength=1.0,
                strategy_name=self.name,
            )

        self._prev_fast_above[ticker] = fast_above

        return signal

    def get_timeframe(self) -> str:
        return "daily"
