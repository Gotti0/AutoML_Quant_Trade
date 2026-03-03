"""
AutoML_Quant_Trade - 특성 벡터 추출 파이프라인

시장 데이터(가격, 거래량)와 거시지표로부터
국면 감지용 특성 벡터를 추출.

■ 미래참조 편향 방지:
  - 모든 rolling() 함수는 과거 방향만 사용 (center=False)
  - min_periods를 윈도우 크기와 동일하게 설정
  - 수익률은 pct_change()로 과거 대비만 계산
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """시장 국면 감지용 특성 벡터 추출"""

    def extract(self, price_df: pd.DataFrame,
                macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        가격 데이터와 거시지표로부터 특성 벡터를 추출.

        Parameters:
            price_df: DataFrame with columns [date, open, high, low, close, volume]
                      date 기준 오름차순 정렬 필수
            macro_df: (선택) DataFrame with columns [date, 다우존스, 나스닥, ...]
                      load_macro_all()의 와이드 포맷 결과

        Returns:
            DataFrame: index=date, columns=[각종 특성], NaN 행 제거됨

        ■ 미래참조 편향 방지:
          - 모든 지표는 해당 날짜 이전 데이터만으로 계산
          - rolling(center=True) 절대 사용 금지
        """
        df = price_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # ── 수익률 (과거 대비) ──
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_21d"] = df["close"].pct_change(21)     # ~1개월
        df["return_63d"] = df["close"].pct_change(63)     # ~3개월

        # ── 롤링 실현 변동성 (연율화) ──
        df["vol_21d"] = df["return_1d"].rolling(
            window=21, min_periods=21
        ).std() * np.sqrt(252)

        df["vol_63d"] = df["return_1d"].rolling(
            window=63, min_periods=63
        ).std() * np.sqrt(252)

        # ── RSI (14일) ──
        df["rsi_14"] = self._compute_rsi(df["close"], period=14)

        # ── MACD ──
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ── 볼린저 밴드 %B (20일) ──
        sma_20 = df["close"].rolling(window=20, min_periods=20).mean()
        std_20 = df["close"].rolling(window=20, min_periods=20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        df["bb_pctb"] = (df["close"] - lower) / (upper - lower)

        # ── 롤링 최대 낙폭 (MDD, 63일) ──
        rolling_max = df["close"].rolling(window=63, min_periods=63).max()
        df["mdd_63d"] = (df["close"] - rolling_max) / rolling_max

        # ── 거래량 변화율 ──
        vol_ma_20 = df["volume"].rolling(window=20, min_periods=20).mean()
        df["volume_ratio"] = df["volume"] / vol_ma_20

        # ── 가격 모멘텀 스코어 ──
        sma_50 = df["close"].rolling(window=50, min_periods=50).mean()
        sma_200 = df["close"].rolling(window=200, min_periods=200).mean()
        df["momentum_score"] = (df["close"] / sma_50 - 1) * 100

        # 200일 이평선 존재 시 골든/데드 크로스 시그널
        if sma_200.notna().any():
            df["trend_signal"] = (sma_50 / sma_200 - 1) * 100
        else:
            df["trend_signal"] = 0.0

        # ── 거시지표 병합 (가용 시) ──
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)

        # 결과 컬럼 선택
        feature_cols = [
            "date",
            "return_1d", "return_5d", "return_21d", "return_63d",
            "vol_21d", "vol_63d",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_pctb",
            "mdd_63d",
            "volume_ratio",
            "momentum_score", "trend_signal",
        ]

        # 거시지표 컬럼이 추가되었으면 포함
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        feature_cols.extend(macro_cols)

        result = df[feature_cols].copy()

        # NaN 행 제거 (학습 초반 워밍업 구간)
        before = len(result)
        result = result.dropna()
        after = len(result)
        logger.info(
            f"Feature extraction: {before} rows → {after} rows "
            f"(dropped {before - after} warmup rows)"
        )

        return result

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(Relative Strength Index) 계산.
        Wilder's smoothing method 사용.
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # 첫 번째 평균은 SMA
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Wilder's smoothing (EMA 근사)
        for i in range(period, len(series)):
            if pd.notna(avg_gain.iloc[i - 1]):
                avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _merge_macro(self, df: pd.DataFrame,
                     macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        거시지표를 가격 데이터에 병합.
        거시지표의 수익률 변화를 특성으로 추가.
        """
        macro = macro_df.copy()

        # 거시지표별 수익률 계산
        for col in macro.columns:
            if col == "date":
                continue
            macro[f"macro_{col}_ret"] = macro[col].pct_change(1)

        # date 기준으로 병합 (left join → 거시 데이터 없는 날은 NaN)
        macro_ret_cols = [c for c in macro.columns if c.startswith("macro_")]
        merge_cols = ["date"] + macro_ret_cols

        df = df.merge(macro[merge_cols], on="date", how="left")

        # 거시지표 NaN은 0으로 채움 (비거래일 등)
        for col in macro_ret_cols:
            df[col] = df[col].fillna(0.0)

        return df
