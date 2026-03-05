"""
AutoML_Quant_Trade - 스크리너 결과 데이터 구조

UnsupervisedScreener의 출력 형식을 정의.
CLI 리포트, JSON 직렬화, DataFrame 변환을 지원.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ScreenerResult:
    """스크리너 실행 결과"""

    timestamp: int                                      # 실행 시점 (YYYYMMDD)
    regime: str                                         # "Bull" / "Bear" / "Crash"
    regime_probs: np.ndarray                            # (n_regimes,) 확률
    selected_tickers: List[str]                         # 추천 종목 목록
    cluster_assignments: Dict[str, int] = field(default_factory=dict)
    anomaly_flags: Dict[str, bool] = field(default_factory=dict)
    rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    # rankings 컬럼: ticker, cluster, rank, tech_score, fund_score, total_score, tier
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    # fundamentals 컬럼: ticker, PER, EPS, ROE, 배당수익률, 부채비율, BPS, PBR

    def to_json(self) -> dict:
        """프론트엔드 대시보드용 JSON 직렬화."""
        rankings_data = []
        if not self.rankings.empty:
            for _, row in self.rankings.iterrows():
                stock = {
                    "ticker": str(row.get("ticker", "")),
                    "name": str(row.get("name", "")),
                    "clusterId": int(row.get("cluster", -1)),
                    "techScore": round(float(row.get("tech_score", 0)), 2),
                    "fundScore": round(float(row.get("fund_score", 0)), 2),
                    "totalScore": round(float(row.get("total_score", 0)), 2),
                    "tier": str(row.get("tier", "C")),
                    "isAnomaly": bool(self.anomaly_flags.get(str(row.get("ticker", "")), False)),
                }

                # 기본적분석 지표 병합
                ticker = str(row.get("ticker", ""))
                if not self.fundamentals.empty and ticker in self.fundamentals["ticker"].values:
                    fund_row = self.fundamentals[self.fundamentals["ticker"] == ticker].iloc[0]
                    stock["fundamentals"] = {
                        "per": _safe_float(fund_row.get("PER", 0)),
                        "roe": _safe_float(fund_row.get("ROE", 0)),
                        "dividendYield": _safe_float(fund_row.get("배당수익률", 0)),
                        "debtRatio": _safe_float(fund_row.get("부채비율", 0)),
                        "pbr": _safe_float(fund_row.get("PBR", 0)),
                        "eps": _safe_float(fund_row.get("EPS", 0)),
                    }
                else:
                    stock["fundamentals"] = {
                        "per": 0, "roe": 0, "dividendYield": 0,
                        "debtRatio": 0, "pbr": 0, "eps": 0,
                    }

                rankings_data.append(stock)

        return {
            "timestamp": str(self.timestamp),
            "regime": self.regime,
            "regimeProbs": {
                "Bull": round(float(self.regime_probs[0]) if len(self.regime_probs) > 0 else 0, 4),
                "Bear": round(float(self.regime_probs[1]) if len(self.regime_probs) > 1 else 0, 4),
                "Crash": round(float(self.regime_probs[2]) if len(self.regime_probs) > 2 else 0, 4),
            },
            "stocks": rankings_data,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """전체 결과를 단일 DataFrame으로 반환."""
        if self.rankings.empty:
            return pd.DataFrame()

        df = self.rankings.copy()
        df["regime"] = self.regime
        df["is_anomaly"] = df["ticker"].map(self.anomaly_flags).fillna(False)
        return df

    def to_cli_report(self) -> str:
        """CLI 터미널용 포맷 리포트."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"  📊 스크리너 결과 — {self.timestamp}")
        lines.append(f"  🎯 현재 국면: {self.regime}")

        if len(self.regime_probs) >= 3:
            lines.append(
                f"     Bull={self.regime_probs[0]:.1%}  "
                f"Bear={self.regime_probs[1]:.1%}  "
                f"Crash={self.regime_probs[2]:.1%}"
            )
        lines.append("=" * 80)

        if self.rankings.empty:
            lines.append("  (결과 없음)")
            return "\n".join(lines)

        # 상위 30개 종목 테이블
        top_n = self.rankings.head(30)
        lines.append("")
        lines.append(f"  {'순위':>4}  {'종목코드':<10}  {'군집':>4}  "
                      f"{'기술':>6}  {'펀더':>6}  {'종합':>6}  {'티어':>4}  {'이상':>4}")
        lines.append("  " + "-" * 60)

        for _, row in top_n.iterrows():
            ticker = str(row.get("ticker", ""))
            is_anom = "⚠" if self.anomaly_flags.get(ticker, False) else ""
            lines.append(
                f"  {int(row.get('rank', 0)):>4}  {ticker:<10}  "
                f"{int(row.get('cluster', -1)):>4}  "
                f"{row.get('tech_score', 0):>6.1f}  "
                f"{row.get('fund_score', 0):>6.1f}  "
                f"{row.get('total_score', 0):>6.1f}  "
                f"{str(row.get('tier', 'C')):>4}  {is_anom:>4}"
            )

        lines.append("")
        lines.append(f"  총 추천 종목: {len(self.selected_tickers)}개")
        lines.append(f"  이상 종목: {sum(self.anomaly_flags.values())}개")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_json(self, path: str):
        """JSON 파일로 저장."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)


def _safe_float(val) -> float:
    """NaN/Inf-safe float 변환."""
    try:
        v = float(val)
        if np.isfinite(v):
            return round(v, 4)
        return 0.0
    except (ValueError, TypeError):
        return 0.0
