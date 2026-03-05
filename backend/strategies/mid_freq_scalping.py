"""
AutoML_Quant_Trade - 중빈도 스캘핑 전략

분봉 RSI 과매도 반등 + 거래량 스파이크 기반.
"""
from typing import Dict, List, Optional

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy


class MidFreqScalping(BaseStrategy):
    """분봉 RSI 과매도 반등 + 볼륨 스파이크 전략"""

    def __init__(self, rsi_period: int = 14,
                 oversold: float = 30, overbought: float = 70,
                 vol_spike_ratio: float = 2.0,
                 vol_ma_period: int = 20):
        super().__init__(name="MidFreqScalping")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.vol_spike_ratio = vol_spike_ratio
        self.vol_ma_period = vol_ma_period
        # BUG-1 FIX: 종목별로 분리 추적
        self._gains: Dict[str, List[float]] = {}
        self._losses: Dict[str, List[float]] = {}
        self._prev_close: Dict[str, float] = {}

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        self._record(event)
        ticker = event.ticker

        prev = self._prev_close.get(ticker)
        if prev is not None:
            delta = event.close - prev
            self._gains.setdefault(ticker, []).append(max(delta, 0))
            self._losses.setdefault(ticker, []).append(max(-delta, 0))
        self._prev_close[ticker] = event.close

        # 워밍업
        if len(self._gains.get(ticker, [])) < self.rsi_period:
            return None

        # RSI 계산
        rsi = self._compute_rsi(ticker)

        # 거래량 스파이크 감지
        volumes = self.get_volume_series(ticker)
        if len(volumes) >= self.vol_ma_period:
            vol_ma = sum(volumes[-self.vol_ma_period:]) / self.vol_ma_period
            is_vol_spike = event.volume > vol_ma * self.vol_spike_ratio
        else:
            is_vol_spike = False

        # 매수: RSI 과매도 + 볼륨 스파이크
        if rsi < self.oversold and is_vol_spike:
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="BUY",
                strength=min(1.0, (self.oversold - rsi) / self.oversold),
                strategy_name=self.name,
            )

        # 매도: RSI 과매수
        if rsi > self.overbought:
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=event.ticker,
                direction="SELL",
                strength=min(1.0, (rsi - self.overbought) / (100 - self.overbought)),
                strategy_name=self.name,
            )

        return None

    def _compute_rsi(self, ticker: str) -> float:
        period = self.rsi_period
        recent_gains = self._gains.get(ticker, [])[-period:]
        recent_losses = self._losses.get(ticker, [])[-period:]

        avg_gain = sum(recent_gains) / period
        avg_loss = sum(recent_losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_timeframe(self) -> str:
        return "minute"
