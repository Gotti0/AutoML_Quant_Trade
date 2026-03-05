"""
AutoML_Quant_Trade - MarketEye API 재무 필드 기반 멀티팩터 스코어링

Cybos Plus CpSysDib.MarketEye 인덱스 코드를 활용하여
기본적분석(Fundamental) 멀티팩터 스코어를 산출.

■ 팩터 구성:
  - Value:   PER (#67, 역순), PBR (현재가/#89 BPS, 역순)
  - Quality: ROE (#77, 정순)
  - Yield:   배당수익률 (#74, 정순)
  - Safety:  부채비율 (#75, 역순)
  - Earnings: EPS (#70, 정순)

■ 스코어링 방식:
  각 팩터를 전체 유니버스 내 백분위 정규화 → 가중 합산 → [0, 100]
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalScorer:
    """MarketEye 재무 필드 기반 멀티팩터 스코어링"""

    # MarketEye 인덱스 → 팩터 매핑
    MARKETEYE_FIELDS = [0, 4, 67, 70, 74, 75, 77, 89]
    # 0: 종목코드, 4: 현재가, 67: PER, 70: EPS, 74: 배당수익률,
    # 75: 부채비율, 77: ROE, 89: BPS

    FACTOR_WEIGHTS: Dict[str, float] = {
        "value_per": 0.20,      # PER 역순 (낮을수록 고점수)
        "quality_roe": 0.20,    # ROE 정순 (높을수록 고점수)
        "yield_div": 0.15,      # 배당수익률 정순
        "safety_debt": 0.15,    # 부채비율 역순 (낮을수록 안전)
        "value_pbr": 0.15,      # PBR 역순 (낮을수록 저평가)
        "earnings_eps": 0.15,   # EPS 정순
    }

    def __init__(self, bridge_client=None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Parameters:
            bridge_client: BridgeClient 인스턴스 (MarketEye API 호출용)
            weights: 팩터 가중치 오버라이드
        """
        self.bridge_client = bridge_client
        if weights:
            self.FACTOR_WEIGHTS = weights

    def fetch_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """
        BridgeClient를 통해 MarketEye 재무 데이터 일괄 조회.

        Parameters:
            tickers: 종목코드 리스트 (예: ['A005930', 'A000660'])
        Returns:
            DataFrame[ticker, 현재가, PER, EPS, 배당수익률, 부채비율, ROE, BPS, PBR]
        """
        if self.bridge_client is None:
            logger.warning("BridgeClient not provided. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            df = self.bridge_client.fetch_fundamentals_batch(tickers)

            if df.empty:
                return df

            # PBR 계산: 현재가 / BPS
            if "현재가" in df.columns and "BPS" in df.columns:
                df["PBR"] = df.apply(
                    lambda r: r["현재가"] / r["BPS"] if r["BPS"] > 0 else np.nan,
                    axis=1,
                )
            else:
                df["PBR"] = np.nan

            return df

        except Exception as e:
            logger.error(f"Failed to fetch fundamentals: {e}")
            return pd.DataFrame()

    def score_from_dataframe(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        이미 로드된 재무 DataFrame에서 스코어를 산출.
        BridgeClient 없이 테스트할 때 유용.

        Parameters:
            fundamentals_df: ticker, PER, EPS, ROE, 배당수익률, 부채비율, BPS, PBR 칼럼 포함
        Returns:
            입력에 fund_score, tier 칼럼이 추가된 DataFrame
        """
        return self.score(fundamentals_df)

    def score(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        각 팩터를 백분위 정규화 → 가중 합산 → 종합 스코어 [0, 100].

        Parameters:
            fundamentals_df: fetch_fundamentals() 또는 외부 DataFrame
        Returns:
            원본에 fund_score, tier 컬럼이 추가된 DataFrame
        """
        df = fundamentals_df.copy()

        if df.empty:
            return df

        # 결측치 / 이상값 필터링
        df = self._clean_fundamentals(df)

        if len(df) < 5:
            logger.warning(f"Too few stocks for scoring: {len(df)}")
            df["fund_score"] = 50.0
            df["tier"] = "C"
            return df

        # 각 팩터 백분위 스코어 계산
        factor_scores = pd.DataFrame(index=df.index)

        # Value: PER (역순 — 낮을수록 고점수)
        if "PER" in df.columns:
            factor_scores["value_per"] = self._percentile_rank(df["PER"], ascending=True)
        else:
            factor_scores["value_per"] = 50.0

        # Quality: ROE (정순)
        if "ROE" in df.columns:
            factor_scores["quality_roe"] = self._percentile_rank(df["ROE"], ascending=False)
        else:
            factor_scores["quality_roe"] = 50.0

        # Yield: 배당수익률 (정순)
        div_col = "배당수익률" if "배당수익률" in df.columns else "dividend_yield"
        if div_col in df.columns:
            factor_scores["yield_div"] = self._percentile_rank(df[div_col], ascending=False)
        else:
            factor_scores["yield_div"] = 50.0

        # Safety: 부채비율 (역순 — 낮을수록 안전)
        debt_col = "부채비율" if "부채비율" in df.columns else "debt_ratio"
        if debt_col in df.columns:
            factor_scores["safety_debt"] = self._percentile_rank(df[debt_col], ascending=True)
        else:
            factor_scores["safety_debt"] = 50.0

        # Value: PBR (역순)
        if "PBR" in df.columns:
            factor_scores["value_pbr"] = self._percentile_rank(df["PBR"], ascending=True)
        else:
            factor_scores["value_pbr"] = 50.0

        # Earnings: EPS (정순)
        if "EPS" in df.columns:
            factor_scores["earnings_eps"] = self._percentile_rank(df["EPS"], ascending=False)
        else:
            factor_scores["earnings_eps"] = 50.0

        # 가중 합산
        total = np.zeros(len(df))
        for factor_name, weight in self.FACTOR_WEIGHTS.items():
            if factor_name in factor_scores.columns:
                total += factor_scores[factor_name].fillna(50.0).values * weight

        df["fund_score"] = np.clip(total, 0.0, 100.0)

        # 티어 분류
        df["tier"] = self.classify_tier(df["fund_score"])

        return df

    def classify_tier(self, scores: pd.Series) -> pd.Series:
        """
        스코어를 5단계 티어로 분류.

        상위 20% → A, 20~40% → B, 40~60% → C, 60~80% → D, 하위 20% → F
        """
        tiers = pd.Series("C", index=scores.index)

        if len(scores) < 5:
            return tiers

        q80 = scores.quantile(0.80)
        q60 = scores.quantile(0.60)
        q40 = scores.quantile(0.40)
        q20 = scores.quantile(0.20)

        tiers[scores >= q80] = "A"
        tiers[(scores >= q60) & (scores < q80)] = "B"
        tiers[(scores >= q40) & (scores < q60)] = "C"
        tiers[(scores >= q20) & (scores < q40)] = "D"
        tiers[scores < q20] = "F"

        return tiers

    @staticmethod
    def _percentile_rank(series: pd.Series, ascending: bool = False) -> pd.Series:
        """
        시리즈를 0~100 백분위 순위로 변환.

        Parameters:
            ascending: True면 작은 값이 높은 점수 (역순 팩터)
        """
        valid = series.dropna()
        if len(valid) < 2:
            return pd.Series(50.0, index=series.index)

        if ascending:
            # 낮을수록 고점수: rank ascending → 역순
            ranks = valid.rank(ascending=True, method="average")
        else:
            # 높을수록 고점수: rank descending → 역순
            ranks = valid.rank(ascending=False, method="average")

        # 0~100 정규화 (순위가 높을수록(작은 숫자일수록) 높은 점수)
        percentiles = 100.0 - ((ranks - 1) / (len(valid) - 1) * 100)
        result = pd.Series(np.nan, index=series.index)
        result[valid.index] = percentiles
        return result.fillna(50.0)

    @staticmethod
    def _clean_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
        """결측치, 이상값 필터링."""
        result = df.copy()

        # PER: 음수(적자) 또는 0은 제외 → NaN 처리
        if "PER" in result.columns:
            result.loc[result["PER"] <= 0, "PER"] = np.nan
            # 극단값 필터 (PER > 200)
            result.loc[result["PER"] > 200, "PER"] = np.nan

        # ROE: 극단값 필터
        if "ROE" in result.columns:
            result.loc[result["ROE"].abs() > 200, "ROE"] = np.nan

        # 부채비율: 음수 제거
        debt_col = "부채비율" if "부채비율" in result.columns else "debt_ratio"
        if debt_col in result.columns:
            result.loc[result[debt_col] < 0, debt_col] = np.nan

        # PBR: 음수 또는 극단값
        if "PBR" in result.columns:
            result.loc[result["PBR"] <= 0, "PBR"] = np.nan
            result.loc[result["PBR"] > 50, "PBR"] = np.nan

        return result
