import logging
import json
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from backend.data.database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Global dependency or state initialization can go here
db = DatabaseManager()

@router.get("/dashboard", response_model=Dict[str, Any])
def get_dashboard_summary():
    """
    Returns the comprehensive dashboard view:
    1. Current Market Regime (Latest inferred probability)
    2. Sub-engine performances (Metrics + Equity Curves)
    """
    try:
        # 최근 백테스트에서 직렬화된 JSON 파일 읽어오기
        cache_path = Path("cache_daishin") / "latest_backtest.json"
        
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                dashboard_data = json.load(f)
            return dashboard_data
        else:
            # 파일이 없을 경우 기존의 Dummy Data 리턴
            return {
                "currentRegime": {
                    "timestamp": "1970-01-01",
                    "probabilities": { "Bull": 0.33, "Bear": 0.33, "Crash": 0.34 },
                    "dominantRegime": "Crash"
                },
                "algorithms": []
            }
        
    except Exception as e:
        logger.error(f"Failed to fetch dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/history")
def get_regime_history():
    """
    Returns the historical time series of market regime probabilities.
    """
    return []


@router.get("/screener/latest", response_model=Dict[str, Any])
def get_screener_latest():
    """최신 스크리너 결과 반환 (cache_daishin/latest_screener.json)."""
    try:
        cache_path = Path("cache_daishin") / "latest_screener.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "timestamp": "",
                "regime": "Unknown",
                "regimeProbs": {"Bull": 0.33, "Bear": 0.33, "Crash": 0.34},
                "stocks": [],
            }
    except Exception as e:
        logger.error(f"Failed to fetch screener data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

