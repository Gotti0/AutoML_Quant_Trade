"""
AutoML_Quant_Trade - 벡터화 전략 모듈

이벤트 기반 전략 4종을 Numpy 배열 연산 기반으로 변환.
VectorizedBacktestEngine의 signal_generator로 사용.

■ 전략 구성:
  1. Regime: 사전 계산된 국면 확률 → 전 종목 노출도 조절
  2. Anomaly: 변동성 기반 이상 스코어 → 리스크 온/오프
  3. Cluster: 모멘텀 순위 → 상위 N% 선택
  4. Long_Safe: 타겟 비중 리밸런싱
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorizedSignalGenerator:
    """
    4개 전략을 통합한 벡터화 시그널 생성기.
    
    VectorizedBacktestEngine.run()의 signal_generator 콜백으로 사용.
    """

    def __init__(
        self,
        precomputed_regimes: Optional[Dict] = None,
        rebalance_freq: int = 21,
        momentum_window: int = 20,
        vol_window: int = 21,
        anomaly_vol_threshold: float = 2.0,
        top_pct: float = 0.10,
        min_volume: int = 10_000,
    ):
        """
        Parameters:
            precomputed_regimes: {date_idx: 'Bull'|'Bear'|'Crash'} 국면 매핑
            rebalance_freq: 리밸런싱 주기 (거래일)
            momentum_window: 모멘텀 계산 윈도우
            vol_window: 변동성 계산 윈도우
            anomaly_vol_threshold: 변동성 이상 탐지 임계값 (σ 배수)
            top_pct: 모멘텀 상위 비율
            min_volume: 유동성 필터 최소 거래량
        """
        self.regimes = precomputed_regimes or {}
        self.rebalance_freq = rebalance_freq
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.anomaly_threshold = anomaly_vol_threshold
        self.top_pct = top_pct
        self.min_volume = min_volume

        # 전략별 자본 배분 비율 (이벤트 엔진과 동일)
        self.allocation = {
            "regime": 0.30,
            "anomaly": 0.25,
            "cluster": 0.25,
            "long_safe": 0.20,
        }

        # 상태
        self._risk_off_mask = None      # 이상 탐지로 매도된 종목
        self._selected_tickers = None   # 클러스터 모멘텀 선정 종목
        self._last_rebalance = -999

    def __call__(
        self, prices_t: np.ndarray, t: int, context: dict
    ) -> np.ndarray:
        """
        전 종목 시그널 벡터 생성.
        
        Parameters:
            prices_t: shape (n_tickers,) 현재 시점 close 가격
            t: 시간 인덱스 (0부터)
            context: {"positions", "cash", "price_history", "dates"}
        
        Returns:
            signals: shape (n_tickers,), >0 매수, <0 매도
        """
        n = len(prices_t)
        history = context["price_history"]  # (t+1, n_tickers)
        positions = context["positions"]
        
        # 리스크 오프 마스크 초기화
        if self._risk_off_mask is None:
            self._risk_off_mask = np.zeros(n, dtype=bool)
        
        # 유효 가격 마스크
        valid = np.isfinite(prices_t) & (prices_t > 0)
        
        # 워밍업
        warmup = max(self.momentum_window, self.vol_window, 63)
        if t < warmup:
            return np.zeros(n)
        
        # ─── 1. 유동성 필터 ───
        # (volume_matrix가 없으므로 가격 변동 기반 근사)
        liquidity_ok = valid.copy()
        
        # ─── 2. Regime 전략: 국면 기반 전체 노출도 ───
        regime_signals = self._regime_strategy(prices_t, t, context, valid)
        
        # ─── 3. Anomaly 전략: 변동성 이상 탐지 ───
        anomaly_signals = self._anomaly_strategy(prices_t, t, context, valid)
        
        # ─── 4. Cluster 모멘텀 전략: 상위 N% 선택 ───
        cluster_signals = self._cluster_strategy(prices_t, t, context, valid)
        
        # ─── 5. Long_Safe 전략: 리밸런싱 ───
        long_safe_signals = self._long_safe_strategy(prices_t, t, context, valid)
        
        # ─── 가중 합산 ───
        signals = (
            regime_signals * self.allocation["regime"]
            + anomaly_signals * self.allocation["anomaly"]
            + cluster_signals * self.allocation["cluster"]
            + long_safe_signals * self.allocation["long_safe"]
        )
        
        # 유동성 필터 적용 (비유동 종목 시그널 제거)
        signals *= liquidity_ok
        
        return signals

    def _regime_strategy(
        self, prices_t, t, context, valid
    ) -> np.ndarray:
        """
        국면 기반 전 종목 노출도 조절.
        
        Bull → 공격적 매수 (0.9배)
        Bear → 방어적 축소 (0.2배)
        Crash → 전량 매도
        """
        n = len(prices_t)
        signals = np.zeros(n)
        
        # 국면 결정 (사전 계산 or 모멘텀 기반 근사)
        regime = self.regimes.get(t, None)
        if regime is None:
            # 모멘텀 기반 국면 근사
            history = context["price_history"]
            if t >= 63:
                # 대표 종목 평균 63일 수익률로 국면 추정
                past_63 = history[t - 63, :]
                returns_63 = np.where(
                    np.isfinite(past_63) & (past_63 > 0),
                    prices_t / past_63 - 1, 0
                )
                avg_return = np.nanmean(returns_63[valid])
                
                if avg_return > 0.05:
                    regime = "Bull"
                elif avg_return < -0.10:
                    regime = "Crash"
                elif avg_return < -0.02:
                    regime = "Bear"
                else:
                    regime = "Neutral"
            else:
                regime = "Neutral"
        
        # 노출도 결정
        exposure_map = {
            "Bull": 0.9, "Neutral": 0.5,
            "Bear": 0.2, "Crash": 0.0,
        }
        exposure = exposure_map.get(regime, 0.5)
        
        # 리밸런싱 주기 확인
        if (t - self._last_rebalance) < self.rebalance_freq and t > 0:
            return signals
        
        positions = context["positions"]
        cash = context["cash"]
        total_equity = cash + np.nansum(positions * prices_t)
        
        if total_equity <= 0:
            return signals
        
        # 현재 포지션 가치 비중
        current_weight = np.where(
            valid, positions * prices_t / total_equity, 0
        )
        total_invested = np.sum(current_weight)
        
        # 목표 노출도와 현재 투자 비중 차이
        diff = exposure - total_invested
        
        if diff > 0.05:
            # 추가 매수 필요 → 균등 분배
            n_valid = np.sum(valid)
            if n_valid > 0:
                per_ticker = diff / n_valid
                signals[valid] = min(per_ticker, 0.02)
        elif diff < -0.05:
            # 매도 필요 → 보유 종목 비례 매도
            held = positions > 0
            if np.any(held):
                signals[held] = diff / np.sum(held)  # 음수
        
        return signals

    def _anomaly_strategy(
        self, prices_t, t, context, valid
    ) -> np.ndarray:
        """
        변동성 기반 이상 탐지.
        
        최근 vol_window일 변동성이 장기 평균의 anomaly_threshold배 초과 시
        해당 종목 리스크 오프 (매도). 정상화 시 재진입.
        """
        n = len(prices_t)
        signals = np.zeros(n)
        history = context["price_history"]
        positions = context["positions"]
        
        # 일간 수익률 계산
        if t < 2:
            return signals
            
        prev_prices = history[t - 1, :]
        daily_returns = np.where(
            valid & np.isfinite(prev_prices) & (prev_prices > 0),
            prices_t / prev_prices - 1, 0
        )
        
        # 최근 vol_window일 변동성
        if t >= self.vol_window:
            recent_returns = np.zeros((self.vol_window, n))
            for i in range(self.vol_window):
                idx = t - self.vol_window + i
                if idx > 0:
                    p_cur = history[idx, :]
                    p_prev = history[idx - 1, :]
                    safe = np.isfinite(p_cur) & np.isfinite(p_prev) & (p_prev > 0)
                    recent_returns[i] = np.where(safe, p_cur / p_prev - 1, 0)
            
            short_vol = np.std(recent_returns, axis=0)
        else:
            return signals
        
        # 장기 변동성 (63일)
        if t >= 63:
            long_returns = np.zeros((63, n))
            for i in range(63):
                idx = t - 63 + i
                if idx > 0:
                    p_cur = history[idx, :]
                    p_prev = history[idx - 1, :]
                    safe = np.isfinite(p_cur) & np.isfinite(p_prev) & (p_prev > 0)
                    long_returns[i] = np.where(safe, p_cur / p_prev - 1, 0)
            
            long_vol = np.std(long_returns, axis=0)
        else:
            long_vol = short_vol
        
        # 이상 탐지: 단기 변동성 > 장기 변동성 × threshold
        anomaly_mask = (short_vol > long_vol * self.anomaly_threshold) & (long_vol > 0)
        
        # 이상 종목 매도
        new_anomalies = anomaly_mask & ~self._risk_off_mask & (positions > 0)
        if np.any(new_anomalies):
            signals[new_anomalies] = -1.0  # 전량 매도
        
        # 정상 복귀 종목 재진입
        recovered = ~anomaly_mask & self._risk_off_mask & valid
        if np.any(recovered):
            signals[recovered] = 0.03  # 점진적 재진입
        
        # 상태 갱신
        self._risk_off_mask = anomaly_mask.copy()
        
        return signals

    def _cluster_strategy(
        self, prices_t, t, context, valid
    ) -> np.ndarray:
        """
        모멘텀 순위 기반 종목 선택.
        
        momentum_window일 수익률 상위 top_pct% 매수,
        리밸런싱 주기마다 교체.
        """
        n = len(prices_t)
        signals = np.zeros(n)
        history = context["price_history"]
        positions = context["positions"]
        
        # 리밸런싱 주기 확인
        if t % self.rebalance_freq != 0:
            return signals
        
        self._last_rebalance = t
        
        # 모멘텀 계산
        past_prices = history[t - self.momentum_window, :]
        momentum = np.where(
            valid & np.isfinite(past_prices) & (past_prices > 0),
            prices_t / past_prices - 1, np.nan
        )
        
        # 유효 모멘텀만 추출
        valid_momentum = momentum[~np.isnan(momentum)]
        if len(valid_momentum) < 20:
            return signals
        
        # 상위 top_pct 임계값
        buy_threshold = np.nanpercentile(momentum, 100 * (1 - self.top_pct))
        
        # 매수 대상
        buy_mask = valid & (momentum >= buy_threshold) & ~np.isnan(momentum)
        n_buy = np.sum(buy_mask)
        
        if n_buy > 0:
            # 균등 비중
            per_ticker = 1.0 / n_buy
            signals[buy_mask] = min(per_ticker, 0.03)
        
        # 기존 보유 종목 중 선정에서 탈락한 종목 매도
        held_not_selected = (positions > 0) & ~buy_mask & valid
        if np.any(held_not_selected):
            signals[held_not_selected] = -0.5  # 50% 매도 (점진적)
        
        return signals

    def _long_safe_strategy(
        self, prices_t, t, context, valid
    ) -> np.ndarray:
        """
        장기 안전 자산 리밸런싱.
        
        변동성이 낮은 상위 20% 종목에 균등 비중 투자.
        """
        n = len(prices_t)
        signals = np.zeros(n)
        history = context["price_history"]
        
        # 리밸런싱 주기 확인 (Long_Safe는 더 긴 주기)
        if t % (self.rebalance_freq * 2) != 0:
            return signals
        
        if t < 63:
            return signals
        
        # 63일 변동성 계산
        returns_63 = np.zeros((63, n))
        for i in range(63):
            idx = t - 63 + i
            if idx > 0:
                p_cur = history[idx, :]
                p_prev = history[idx - 1, :]
                safe = np.isfinite(p_cur) & np.isfinite(p_prev) & (p_prev > 0)
                returns_63[i] = np.where(safe, p_cur / p_prev - 1, 0)
        
        vol_63 = np.std(returns_63, axis=0)
        
        # 유효 종목 중 변동성 하위 20% (안전 자산)
        valid_vols = vol_63[valid]
        if len(valid_vols) < 20:
            return signals
        
        low_vol_threshold = np.percentile(valid_vols[valid_vols > 0], 20)
        safe_mask = valid & (vol_63 <= low_vol_threshold) & (vol_63 > 0)
        n_safe = np.sum(safe_mask)
        
        if n_safe > 0:
            per_ticker = 1.0 / n_safe
            signals[safe_mask] = min(per_ticker, 0.02)
        
        return signals
