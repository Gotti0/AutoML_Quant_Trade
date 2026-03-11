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
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PyTorch GPU 가속 (선택적)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed. GPU feature acceleration disabled.")


def _get_device():
    """CUDA 가용 시 GPU, 아니면 CPU 반환."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu') if TORCH_AVAILABLE else None


def batch_rolling_features_gpu(
    price_dict: Dict[str, pd.DataFrame],
    windows: list = None,
) -> Dict[str, pd.DataFrame]:
    """
    PyTorch as_strided를 사용한 GPU 배치 롤링 피처 계산.
    
    모든 종목의 close 가격을 하나의 2D 텐서(n_tickers × n_times)로 쌓은 뒤,
    GPU에서 동시에 rolling mean, std를 계산하여 종목별 DataFrame으로 반환.
    
    Parameters:
        price_dict: {ticker: DataFrame[date, close, ...]}
        windows: 롤링 윈도우 크기 리스트 (기본: [21, 63])
    
    Returns:
        {ticker: DataFrame[rolling_mean_21, rolling_std_21, ...]}
    """
    if not TORCH_AVAILABLE:
        return {}
    
    windows = windows or [21, 63]
    device = _get_device()
    
    # 종목을 최대 시계열 길이로 정렬
    tickers = list(price_dict.keys())
    if not tickers:
        return {}
    
    # 최소 길이 종목 필터 (가장 긴 윈도우보다 길어야 유효)
    max_window = max(windows)
    valid_tickers = [t for t in tickers if len(price_dict[t]) > max_window]
    if not valid_tickers:
        return {}
    
    # 공통 길이로 자르기 (가장 짧은 종목 기준)
    min_len = min(len(price_dict[t]) for t in valid_tickers)
    
    # 2D 텐서 구성: (n_tickers, min_len)
    close_matrix = np.stack([
        price_dict[t]["close"].values[-min_len:].astype(np.float64)
        for t in valid_tickers
    ])
    
    tensor = torch.from_numpy(close_matrix).to(device)
    
    results = {t: {} for t in valid_tickers}
    
    for window in windows:
        n_tickers, seq_len = tensor.shape
        n_windows = seq_len - window + 1
        
        if n_windows <= 0:
            continue
        
        # as_strided: Zero-Copy 롤링 윈도우 뷰
        strides = tensor.stride()
        rolling_view = tensor.as_strided(
            size=(n_tickers, n_windows, window),
            stride=(strides[0], strides[1], strides[1])
        )
        
        # GPU 병렬 집계 연산
        rolling_mean = rolling_view.mean(dim=-1).cpu().numpy()
        rolling_std = rolling_view.std(dim=-1).cpu().numpy()
        
        # 앞쪽 NaN 패딩 (원본 시계열 길이와 맞추기)
        pad_len = window - 1
        for i, ticker in enumerate(valid_tickers):
            results[ticker][f"gpu_rolling_mean_{window}"] = np.concatenate([
                np.full(pad_len, np.nan), rolling_mean[i]
            ])
            results[ticker][f"gpu_rolling_std_{window}"] = np.concatenate([
                np.full(pad_len, np.nan), rolling_std[i]
            ])
    
    # DataFrame으로 변환
    output = {}
    for ticker in valid_tickers:
        if results[ticker]:
            df = pd.DataFrame(results[ticker])
            df["date"] = price_dict[ticker]["date"].values[-min_len:]
            output[ticker] = df
    
    logger.info(
        f"GPU batch rolling features: {len(output)} tickers, "
        f"windows={windows}, device={device}"
    )
    return output

# ── 피처 캠싱 (동일 종목 중복 계산 방지) ──
_feature_cache: dict = {}

class FeatureEngineer:
    """시장 국면 감지용 특성 벡터 추출"""

    def extract(self, price_df: pd.DataFrame,
                minute_df: pd.DataFrame = None,
                macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        가격 데이터와 거시지표로부터 특성 벡터를 추출.
        동일 데이터에 대한 중복 계산을 캐싱으로 방지.
        """
        # 캐시 키: (마지막 날짜, 행 수)
        cache_key = (int(price_df["date"].iloc[-1]), len(price_df))
        if cache_key in _feature_cache:
            return _feature_cache[cache_key].copy()
        
        result = self._extract_impl(price_df, minute_df, macro_df)
        
        # 캐시 크기 제한 (메모리 보호)
        if len(_feature_cache) > 500:
            _feature_cache.clear()
        _feature_cache[cache_key] = result
        return result

    def _extract_impl(self, price_df: pd.DataFrame,
                minute_df: Optional[pd.DataFrame] = None,
                macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """실제 피처 추출 구현."""
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

        # ── 분포 특성 피처 (비지도학습 최적화) ──

        # 수익률 왜도 (비대칭성: 좌 꼬리 = 급락 위험)
        df["skewness_21d"] = df["return_1d"].rolling(
            window=21, min_periods=21
        ).skew()

        # 수익률 첨도 (팻테일 정도: 높을수록 극단적 변동)
        df["kurtosis_21d"] = df["return_1d"].rolling(
            window=21, min_periods=21
        ).kurt()

        # 허스트 지수 (Numpy 벡터화 — rolling.apply 제거)
        df["hurst_exponent"] = self._compute_hurst_vectorized(
            df["close"].values, window=63
        )

        # 수익률 정보 엔트로피 (불확실성: 높을수록 예측 곤란)
        df["entropy_21d"] = df["return_1d"].rolling(
            window=21, min_periods=21
        ).apply(self._compute_entropy, raw=True)

        # 자기상관 계수 (Numpy 벡터화 — rolling.apply + lambda 제거)
        returns_arr = df["return_1d"].values
        for lag in range(1, 6):
            df[f"autocorr_lag{lag}"] = self._rolling_autocorr_numpy(
                returns_arr, lag=lag, window=63
            )

        # ── 거시지표 병합 (가용 시) ──
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)
            
        # ── 단기 트레이딩 핵심 Feature: 09:00 ~ 09:30 수익률 병합 ──
        if minute_df is not None and not minute_df.empty:
            df = self._merge_intraday_returns(df, minute_df)

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
            # 분포 특성 피처
            "skewness_21d", "kurtosis_21d",
            "hurst_exponent", "entropy_21d",
            "autocorr_lag1", "autocorr_lag2", "autocorr_lag3",
            "autocorr_lag4", "autocorr_lag5",
        ]

        # 거시지표 컬럼이 추가되었으면 포함
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        feature_cols.extend(macro_cols)

        # 단기 수익률 컬럼이 추가되었으면 포함 (다중 청산 타겟 + 갭 피처)
        target_cols = [f"ret_{hm}" for hm in [905, 910, 915, 920, 925, 930]] + ["gap_pct"]
        for t_col in target_cols:
            if t_col in df.columns:
                feature_cols.append(t_col)

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

    @staticmethod
    def _compute_hurst_vectorized(prices: np.ndarray, window: int = 63) -> np.ndarray:
        """
        Numpy 벡터화된 Hurst 지수 계산.
        
        rolling.apply() 대신 sliding_window_view로 전체 윈도우를
        한 번에 생성하여 벡터화 R/S 분석 실행.
        """
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n < window:
            return result
        
        log_prices = np.log(prices + 1e-10)
        log_returns = np.diff(log_prices)
        
        # sliding window 생성 (zero-copy)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(log_returns, window - 1)
        # windows.shape = (n - window + 1, window - 1)
        
        n_windows = windows.shape[0]
        hurst_values = np.full(n_windows, 0.5)
        
        # 고정된 k 값들에 대해 R/S 계산
        max_k = min((window - 1) // 2, 32)
        if max_k < 4:
            result[window:] = 0.5
            return result
        
        k_values = np.arange(4, max_k + 1)
        log_ks = np.log(k_values)
        
        for w_idx in range(n_windows):
            chunk_returns = windows[w_idx]
            rs_list = []
            
            for k in k_values:
                n_chunks = len(chunk_returns) // k
                if n_chunks < 1:
                    continue
                
                rs_vals = []
                for i in range(n_chunks):
                    seg = chunk_returns[i * k:(i + 1) * k]
                    mean_seg = seg.mean()
                    cumdev = np.cumsum(seg - mean_seg)
                    R = cumdev.max() - cumdev.min()
                    S = seg.std(ddof=1)
                    if S > 1e-10:
                        rs_vals.append(R / S)
                
                if rs_vals:
                    rs_list.append(np.log(np.mean(rs_vals)))
                else:
                    rs_list.append(np.nan)
            
            valid_mask = ~np.isnan(rs_list)
            if np.sum(valid_mask) >= 2:
                valid_rs = np.array(rs_list)[valid_mask]
                valid_logk = log_ks[:len(rs_list)][valid_mask]
                coeffs = np.polyfit(valid_logk, valid_rs, 1)
                hurst_values[w_idx] = float(np.clip(coeffs[0], 0.0, 1.0))
        
        result[window:] = hurst_values[:n - window]
        return result

    @staticmethod
    def _rolling_autocorr_numpy(returns: np.ndarray, lag: int, window: int) -> np.ndarray:
        """
        Numpy 벡터화된 rolling autocorrelation.
        
        pandas Series.autocorr() + rolling.apply(lambda) 대신
        직접 상관계수를 계산하여 객체 생성 오버헤드 제거.
        """
        n = len(returns)
        result = np.full(n, np.nan)
        
        if n < window + lag:
            return result
        
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(returns, window)
        
        for i in range(len(windows)):
            w = windows[i]
            if len(w) > lag:
                x = w[:-lag]
                y = w[lag:]
                if len(x) > 1:
                    mx, my = x.mean(), y.mean()
                    sx, sy = x.std(), y.std()
                    if sx > 1e-10 and sy > 1e-10:
                        corr = np.mean((x - mx) * (y - my)) / (sx * sy)
                        result[i + window - 1] = corr
        
        return result

    @staticmethod
    def _compute_entropy(returns: np.ndarray) -> float:
        """
        수익률 분포의 정보 엔트로피 (Shannon Entropy).
        """
        if len(returns) < 5:
            return 0.0

        # 히스토그램 기반 이산 확률 분포
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]  # 0인 빈 제거
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

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

    def _merge_intraday_returns(self, daily_df: pd.DataFrame,
                                minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        분봉 데이터(09:00~09:30)에서 단기 수익률(Target Feature)과 
        시초가 갭 피처(gap_pct)를 일봉 데이터 프레임에 병합합니다.
        
        ■ 미래참조 편향 차단 적용 (shift(-1)):
          - T일 마감 기준 Feature 행에, T+1일 아침 갭하락(gap_pct) 정보와
          - 5분 단위(T+1일 0905 ~ 0930) 타겟 수익률 6개를 맵핑.
        """
        try:
            # 09:00 시초가 추출 (gap_pct 및 매수 단가 기준)
            open_prices = minute_df[minute_df["time"].isin([900, 90000])][["date", "open"]].rename(columns={"open": "open_0900"})
            
            intra_df = open_prices.copy()
            
            # 다중 청산 타겟 (9:05, 9:10, 9:15, 9:20, 9:25, 9:30) 추출 및 수익률 계산
            target_times = [905, 910, 915, 920, 925, 930]
            
            for t in target_times:
                # 3자리(905), 4자리(0905), 5/6자리(90500) 호환성 고려
                t_formats = [t, t * 100] 
                close_df = minute_df[minute_df["time"].isin(t_formats)][["date", "close"]].rename(columns={"close": f"close_{t}"})
                # 고유 기준(date) 하나만 남김 (중복 제거)
                close_df = close_df.drop_duplicates(subset=['date'])
                
                intra_df = pd.merge(intra_df, close_df, on="date", how="left")
                # (t분 종가 - 09:00 시초가) / 09:00 시초가 = t분 시점의 수익률
                intra_df[f"ret_{t}"] = (intra_df[f"close_{t}"] - intra_df["open_0900"]) / intra_df["open_0900"]

            # 일봉 df에 Left 병합
            merged_df = daily_df.merge(intra_df, on="date", how="left")
            
            # 갭 피처 생성: (오늘 아침 09:00 시초가 - 어제 종가) / 어제 종가
            # 먼저 같은 Row(당일) 기준으로 오늘치 갭을 계산한 뒤 당깁니다.
            # 주의: 분봉 데이터 기준일의 0900분 시초가와 전거래일의 종가를 비교해야함
            # 편의상 merged_df의 T일 종가("close")와 T+1일의 시가("open_0900".shift(-1))로 바로 계산
            
            merged_df["gap_pct"] = (merged_df["open_0900"].shift(-1) - merged_df["close"]) / merged_df["close"]
            
            # 타겟 수익률 6개 시프트 (-1) 적용
            ret_cols = [f"ret_{t}" for t in target_times]
            for col in ret_cols:
                merged_df[col] = merged_df[col].shift(-1)
                
            # 불필요한 중간 임시 가격 컬럼 제외하고 반환
            drop_cols = ["open_0900"] + [f"close_{t}" for t in target_times]
            merged_df = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns])
            
            return merged_df
        except Exception as e:
            logger.error(f"Failed to merge intraday targets & gap: {e}")
            return daily_df
