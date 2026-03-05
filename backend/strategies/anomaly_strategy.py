"""
AutoML_Quant_Trade - 이상 탐지 전략

AnomalyDetector의 이상 스코어를 기반으로
시장 급변 시 리스크 축소, 정상 복귀 시 재진입.

■ 이상 스코어 > threshold_high → SELL (리스크 축소)
■ 이상 스코어 < threshold_low  → BUY  (재진입)
■ 사이값 → HOLD (유지)
"""
import logging
from typing import Optional

import pandas as pd

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AnomalyStrategy(BaseStrategy):
    """Anomaly Score 기반 리스크 관리 전략"""

    def __init__(self, anomaly_detector,
                 threshold_high: float = 0.7,
                 threshold_low: float = 0.3,
                 lookback: int = 63):
        """
        Parameters:
            anomaly_detector: AnomalyDetector 인스턴스
            threshold_high: 이상 스코어 상한 (초과 시 리스크 축소)
            threshold_low: 이상 스코어 하한 (미만 시 재진입)
            lookback: 피처 계산에 필요한 최소 히스토리 길이
        """
        super().__init__()
        self.anomaly_detector = anomaly_detector
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.lookback = lookback

        self._in_risk_off: dict = {}  # {ticker: bool}

    def get_timeframe(self) -> str:
        return "daily"

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        """시장 데이터 처리 → 이상 스코어 기반 시그널."""
        self._record(event)
        ticker = event.ticker

        if ticker not in self._in_risk_off:
            self._in_risk_off[ticker] = False

        history = self._history.get(ticker, [])
        if len(history) < self.lookback:
            return None

        try:
            from backend.models.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()

            price_df = pd.DataFrame([{
                "date": e.timestamp,
                "open": e.open, "high": e.high, "low": e.low,
                "close": e.close, "volume": e.volume,
            } for e in history])
            features = fe.extract(price_df)

            if features.empty:
                return None

            # 최신 시점의 이상 스코어
            score = self.anomaly_detector.score(features).iloc[-1]

            if score > self.threshold_high and not self._in_risk_off[ticker]:
                # 이상 탐지 → 리스크 축소
                self._in_risk_off[ticker] = True
                logger.info(
                    f"⚠ {ticker} Anomaly detected (score={score:.3f}) → SELL"
                )
                return SignalEvent(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    direction="SELL",
                    strength=1.0,
                )

            elif score < self.threshold_low and self._in_risk_off[ticker]:
                # 정상 복귀 → 재진입
                self._in_risk_off[ticker] = False
                logger.info(
                    f"✅ {ticker} Normal restored (score={score:.3f}) → BUY"
                )
                return SignalEvent(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    direction="BUY",
                    strength=0.5,  # 0.5 청신호(점진적 재진입)
                )

        except Exception as e:
            logger.debug(f"AnomalyStrategy signal failed for {ticker}: {e}")

        return None
