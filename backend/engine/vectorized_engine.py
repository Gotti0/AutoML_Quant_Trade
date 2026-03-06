"""
AutoML_Quant_Trade - 하이브리드 벡터화 백테스트 엔진

기존 이벤트 주도형 엔진(BacktestEventLoop)의 대안으로,
"시간 축 루프 + 종목 축 벡터화" 하이브리드 구조를 제공.

■ 핵심 설계:
  - 700만 개의 MarketEvent 객체를 생성하지 않음
  - 2D Numpy 행렬(시간 × 종목)을 시간 축으로 순회
  - 종목 독립 연산은 벡터화, 크로스 에셋 연산은 일 단위 처리
  - 미래참조 편향 방지 원칙을 완벽히 준수

■ 성능 기대:
  - 이벤트 기반 대비 50~100x 속도 향상 (일 단위 루프 ~2,520회)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VectorizedBacktestEngine:
    """
    하이브리드 벡터화 백테스트 엔진.
    
    기존 BacktestEventLoop를 대체하는 것이 아니라,
    고속 백테스트가 필요한 파라미터 탐색/최적화 용도로 사용.
    """

    def __init__(self, initial_capital: float = 100_000_000,
                 commission_rate: float = 0.00015,
                 tax_rate: float = 0.0018):
        """
        Parameters:
            initial_capital: 초기 자본금
            commission_rate: 매매 수수료율
            tax_rate: 매도 거래세율
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate

    def build_price_matrix(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        {ticker: DataFrame} → 2D 가격 행렬로 변환.
        
        Returns:
            price_matrix: shape (n_days, n_tickers) — close 가격
            tickers: 종목 코드 리스트 (열 순서)
            dates: 거래일 배열 (행 순서)
        """
        # 모든 종목의 날짜 합집합
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df["date"].values)
        dates = np.sort(list(all_dates))

        tickers = sorted(market_data.keys())
        n_days = len(dates)
        n_tickers = len(tickers)

        # 가격 행렬 초기화 (NaN = 미상장/거래정지)
        price_matrix = np.full((n_days, n_tickers), np.nan, dtype=np.float64)
        open_matrix = np.full((n_days, n_tickers), np.nan, dtype=np.float64)
        volume_matrix = np.full((n_days, n_tickers), 0, dtype=np.int64)

        date_to_idx = {d: i for i, d in enumerate(dates)}
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        for ticker, df in market_data.items():
            t_idx = ticker_to_idx[ticker]
            for row in df.itertuples(index=False):
                d_idx = date_to_idx.get(int(row.date))
                if d_idx is not None:
                    price_matrix[d_idx, t_idx] = float(row.close)
                    open_matrix[d_idx, t_idx] = float(row.open)
                    volume_matrix[d_idx, t_idx] = int(row.volume)

        logger.info(
            f"Price matrix built: {n_days} days × {n_tickers} tickers, "
            f"date range [{dates[0]}..{dates[-1]}]"
        )

        self._price_matrix = price_matrix
        self._open_matrix = open_matrix
        self._volume_matrix = volume_matrix
        self._tickers = tickers
        self._dates = dates
        self._ticker_to_idx = ticker_to_idx

        return price_matrix, tickers, dates

    def run(
        self,
        market_data: Dict[str, pd.DataFrame],
        signal_generator=None,
        meta_layer=None,
    ) -> pd.DataFrame:
        """
        하이브리드 벡터화 백테스트 실행.
        
        Parameters:
            market_data: {ticker: DataFrame}
            signal_generator: (prices_t, t, context) → signals_array 콜백
            meta_layer: (t, prices_t, equity, positions) → adjustments 콜백
            
        Returns:
            에퀴티 커브 DataFrame
        """
        price_matrix, tickers, dates = self.build_price_matrix(market_data)
        n_days, n_tickers = price_matrix.shape

        # 상태 벡터 (Numpy 배열)
        positions = np.zeros(n_tickers, dtype=np.float64)  # 종목별 보유 수량
        cash = np.float64(self.initial_capital)

        # 에퀴티 기록
        equity_history = []
        
        # 이전 봉에서 발생한 시그널 (t+1 시가로 체결)
        pending_signals = np.zeros(n_tickers, dtype=np.float64)

        logger.info(
            f"VectorizedBacktest started: {n_days} days, "
            f"{n_tickers} tickers, capital={self.initial_capital:,.0f}"
        )

        for t in range(n_days):
            # 현재 시점의 횡단면 가격 (Zero-Copy 뷰)
            prices_t = price_matrix[t, :]
            opens_t = self._open_matrix[t, :]

            # ── 1. 대기 시그널 체결 (t-1 시그널 → t 시가) ──
            if np.any(pending_signals != 0):
                positions, cash = self._execute_signals(
                    pending_signals, opens_t, positions, cash
                )
                pending_signals[:] = 0

            # ── 2. 시그널 생성 (종목 독립 벡터화) ──
            if signal_generator is not None:
                context = {
                    "positions": positions,
                    "cash": cash,
                    "price_history": price_matrix[:t+1, :],
                    "dates": dates[:t+1],
                }
                pending_signals = signal_generator(prices_t, t, context)

            # ── 3. 메타 레이어 (크로스 에셋, 일 단위) ──
            if meta_layer is not None:
                total_equity = cash + np.nansum(positions * prices_t)
                adjustments = meta_layer(t, prices_t, total_equity, positions)
                if adjustments is not None:
                    positions, cash = adjustments

            # ── 4. 에퀴티 스냅샷 ──
            portfolio_value = np.nansum(positions * prices_t)
            total_equity = cash + portfolio_value
            equity_history.append({
                "date": dates[t],
                "equity": total_equity,
                "cash": cash,
                "portfolio_value": portfolio_value,
            })

            if t % 500 == 0:
                logger.info(
                    f"  Day {t}/{n_days}: equity={total_equity:,.0f}, "
                    f"cash={cash:,.0f}, positions={int(np.count_nonzero(positions))}"
                )

        # 결과 생성
        equity_df = pd.DataFrame(equity_history)
        
        # 성과 지표 계산
        if len(equity_df) > 1:
            returns = equity_df["equity"].pct_change().dropna()
            total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
            sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
            
            peak = equity_df["equity"].cummax()
            drawdown = (equity_df["equity"] - peak) / peak
            mdd = drawdown.min()
            
            logger.info(
                f"VectorizedBacktest complete: {n_days} days, "
                f"Return={total_return:.2%}, Sharpe={sharpe:.2f}, MDD={mdd:.2%}"
            )

        return equity_df

    def _execute_signals(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        positions: np.ndarray,
        cash: float,
    ) -> Tuple[np.ndarray, float]:
        """
        시그널 배열을 기반으로 체결 시뮬레이션.
        
        signals > 0: 매수 비중, signals < 0: 매도 비중
        체결가: 해당 봉의 시가 (opens_t)
        
        ■ 미래참조 편향 방지:
          - 시그널은 t-1 종가 기준으로 생성됨
          - 체결은 t 시가 기준
        """
        # 유효한 가격이 있는 종목만 처리
        valid_mask = np.isfinite(prices) & (prices > 0)
        
        # 매도 먼저 처리 (현금 확보)
        sell_mask = valid_mask & (signals < 0) & (positions > 0)
        if np.any(sell_mask):
            sell_tickers = np.where(sell_mask)[0]
            for idx in sell_tickers:
                sell_qty = int(positions[idx] * abs(signals[idx]))
                sell_qty = min(sell_qty, int(positions[idx]))
                if sell_qty > 0:
                    revenue = sell_qty * prices[idx]
                    fee = revenue * (self.commission_rate + self.tax_rate)
                    positions[idx] -= sell_qty
                    cash += revenue - fee

        # 매수 처리
        buy_mask = valid_mask & (signals > 0) & (cash > 0)
        if np.any(buy_mask):
            buy_tickers = np.where(buy_mask)[0]
            for idx in buy_tickers:
                allocatable = cash * signals[idx]
                buy_price = prices[idx] * (1 + self.commission_rate)
                qty = int(allocatable / buy_price)
                if qty > 0:
                    cost = qty * buy_price
                    if cost <= cash:
                        positions[idx] += qty
                        cash -= cost

        return positions, cash
