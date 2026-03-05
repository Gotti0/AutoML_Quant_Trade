"""
AutoML_Quant_Trade - 국면 적응형 전략

HMM/GMM 국면 확률 벡터를 기반으로
포지션 방향과 크기를 동적으로 조절.

■ Bull → 공격적 롱 (target=0.8~1.0)
■ Bear → 방어적/현금화 (target=0.1~0.3)
■ Crash → 풀 현금 or 최소 포지션 (target=0.0~0.1)
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy(BaseStrategy):
    """국면 확률 벡터 기반 동적 포지션 관리"""

    def __init__(self, regime_model, screener=None,
                 bull_exposure: float = 0.9,
                 bear_exposure: float = 0.2,
                 crash_exposure: float = 0.0,
                 rebalance_days: int = 21):
        """
        Parameters:
            regime_model: RegimeHMM 또는 RegimeGMM
            screener: UnsupervisedScreener (종목 풀 동적 교체)
            bull_exposure: Bull 국면 목표 노출도
            bear_exposure: Bear 국면 목표 노출도
            crash_exposure: Crash 국면 목표 노출도
            rebalance_days: 리밸런싱 주기 (거래일)
        """
        super().__init__()
        self.regime_model = regime_model
        self.screener = screener
        self.bull_exposure = bull_exposure
        self.bear_exposure = bear_exposure
        self.crash_exposure = crash_exposure
        self.rebalance_days = rebalance_days

        self._day_counter: dict = {}   # {ticker: 마지막 리밸런싱 이후 일수}
        self._current_regime: str = "Neutral"
        self._target_weight: float = 0.5

    def get_timeframe(self) -> str:
        return "daily"

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        """시장 데이터 처리 → 국면 기반 포지션 시그널."""
        self._record(event)
        ticker = event.ticker

        # 카운터 초기화
        if ticker not in self._day_counter:
            self._day_counter[ticker] = 0
        self._day_counter[ticker] += 1

        # 워밍업: 충분한 히스토리 필요
        history = self._histories.get(ticker, [])
        if len(history) < 63:
            return None

        # 리밸런싱 주기 확인
        if self._day_counter[ticker] % self.rebalance_days != 0:
            return None

        # 피처 추출 후 국면 확률 계산
        try:
            from backend.models.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()

            price_df = pd.DataFrame(history)
            features = fe.extract(price_df)

            if features.empty:
                return None

            probs = self.regime_model.predict_proba(features)
            if len(probs) == 0:
                return None

            latest_probs = probs[-1]
            regime_id = int(np.argmax(latest_probs))

            # 국면 해석
            exposure_map = {
                0: ("Bull", self.bull_exposure),
                1: ("Bear", self.bear_exposure),
                2: ("Crash", self.crash_exposure),
            }
            regime_name, target_weight = exposure_map.get(
                regime_id, ("Neutral", 0.5)
            )
            self._current_regime = regime_name
            self._target_weight = target_weight

            logger.debug(
                f"{ticker} Regime={regime_name} "
                f"(probs={latest_probs}) → target={target_weight:.1%}"
            )

            # TARGET 시그널 생성
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=ticker,
                direction="TARGET",
                strength=target_weight,
            )

        except Exception as e:
            logger.debug(f"RegimeAdaptive signal failed for {ticker}: {e}")
            return None

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def target_weight(self) -> float:
        return self._target_weight
