"""
AutoML_Quant_Trade - 자산군 통합 매퍼

국내 주식 + 해외 주식/지수 + 원자재를 통합 관리.
6단계 리스크 프로필별 자산 배분 비중 매핑.
"""
import logging
from typing import Dict, Optional, List

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class AssetUniverseMapper:
    """자산군 통합 매퍼 — 프로필별 배분 비중 + 코드 매핑"""

    # 6단계 리스크 프로필: 국내 주식, 해외 주식, 금, 채권
    PROFILES: Dict[str, Optional[Dict[str, float]]] = {
        "Aggressive":  {"Equity_Domestic": 0.30, "Equity_US": 0.60, "Gold": 0.05, "Bond": 0.05},
        "Growth":      {"Equity_Domestic": 0.30, "Equity_US": 0.40, "Gold": 0.10, "Bond": 0.20},
        "Balanced":    {"Equity_Domestic": 0.20, "Equity_US": 0.30, "Gold": 0.15, "Bond": 0.35},
        "Moderate":    {"Equity_Domestic": 0.10, "Equity_US": 0.20, "Gold": 0.15, "Bond": 0.55},
        "Stable":      {"Equity_Domestic": 0.05, "Equity_US": 0.10, "Gold": 0.15, "Bond": 0.70},
        "Custom":      None,  # HMM 국면 기반 동적 배분 (런타임에 결정)
    }

    # 자산군 → 구체적 코드 매핑
    ASSET_TO_CODES: Dict[str, Dict] = {
        "Equity_Domestic": {
            "source": "domestic",
            "codes": ["A069500"],  # KOSPI 200 ETF
        },
        "Equity_US": {
            "source": "overseas",
            "codes": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        },
        "Gold": {
            "source": "overseas",
            "codes": ["CM@NGLD"],
        },
        "Bond": {
            "source": "domestic",
            "codes": ["A304660"],  # KODEX 미국채울트라30년선물(H)
        },
    }

    def get_target_portfolio(self, profile: str,
                              regime_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        지정된 프로필의 목표 자산 배분을 반환.

        Parameters:
            profile: 프로필명 ("Aggressive" ~ "Stable" 또는 "Custom")
            regime_weights: Custom 프로필 시, HMM 기반 동적 가중치
        Returns:
            자산군별 비중 딕셔너리 (예: {"Equity_Domestic": 0.3, "Equity_US": 0.6, ...})
        """
        if profile == "Custom":
            if regime_weights is None:
                raise ValueError("Custom profile requires regime_weights")
            # 비중 합이 1이 되도록 정규화
            total = sum(regime_weights.values())
            if total > 0:
                return {k: v / total for k, v in regime_weights.items()}
            return regime_weights

        allocation = self.PROFILES.get(profile)
        if allocation is None:
            raise ValueError(f"Unknown profile: {profile}")

        return allocation.copy()

    def resolve_to_codes(self, allocation: Dict[str, float]) -> Dict[str, Dict]:
        """
        자산군 비중을 구체적 코드와 데이터 소스로 변환.

        Parameters:
            allocation: {"Equity_Domestic": 0.3, "Equity_US": 0.6, ...}
        Returns:
            {
                "A069500": {"source": "domestic", "weight": 0.3},
                "AAPL": {"source": "overseas", "weight": 0.12},
                ...
            }
        """
        resolved = {}

        for asset_class, weight in allocation.items():
            if weight <= 0:
                continue

            mapping = self.ASSET_TO_CODES.get(asset_class)
            if mapping is None:
                logger.warning(f"Unknown asset class: {asset_class}")
                continue

            codes = mapping["codes"]
            source = mapping["source"]
            # 같은 자산군 내 코드들에 동일 비중 배분
            per_code_weight = weight / len(codes)

            for code in codes:
                resolved[code] = {
                    "source": source,
                    "weight": per_code_weight,
                    "asset_class": asset_class,
                }

        return resolved

    def get_all_codes(self) -> List[str]:
        """모든 자산군의 구체적 코드 목록 반환."""
        all_codes = []
        for mapping in self.ASSET_TO_CODES.values():
            all_codes.extend(mapping["codes"])
        return all_codes

    def get_codes_by_source(self, source: str) -> List[str]:
        """데이터 소스별 코드 목록 반환."""
        codes = []
        for mapping in self.ASSET_TO_CODES.values():
            if mapping["source"] == source:
                codes.extend(mapping["codes"])
        return codes
