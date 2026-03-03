from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging
import sys
import os

# 32비트 환경에서 로컬 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daishin_agent import DaishinAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Daishin 32-bit Bridge Server")

# Initialize agent globally for single instance use
agent = DaishinAgent()


@app.on_event("startup")
async def startup_event():
    """Called when the FastAPI application starts up."""
    logger.info("Starting up Daishin API Bridge Server...")
    success = agent.wait_for_login(timeout=600)
    if not success:
        logger.error("Failed to connect to Daishin HTS on startup.")


# ══════════════════════════════════════════
# 국내 주식 엔드포인트
# ══════════════════════════════════════════

@app.get("/api/dostk/daily")
async def get_daily_data(stk_cd: str, count: int = 500):
    """국내 주식 일봉 OHLCV 조회."""
    _check_connection()
    formatted_code = _format_code(stk_cd)
    logger.info(f"Daily request: {formatted_code}, count={count}")

    try:
        data = agent.get_daily_ohlcv(formatted_code, count)
        if data is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve daily data.")
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dostk/chart")
async def get_chart_data(stk_cd: str, count: int = 150000,
                         since_date: int = None, since_time: int = None):
    """국내 주식 분봉 데이터 조회."""
    _check_connection()
    formatted_code = _format_code(stk_cd)
    logger.info(f"Minute request: {formatted_code}, count={count}, since={since_date} {since_time}")

    try:
        data = agent.get_minute_chart(formatted_code, count, since_date, since_time)
        if data is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve chart data.")
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dostk/info")
async def get_stock_info(stk_cd: str):
    """종목 메타 정보 조회."""
    _check_connection()
    formatted_code = _format_code(stk_cd)

    try:
        data = agent.get_stock_info(formatted_code)
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchInfoRequest(BaseModel):
    tickers: List[str]


@app.post("/api/dostk/info_batch")
async def get_stock_info_batch(req: BatchInfoRequest):
    """최대 200개 종목 일괄 메타 정보 조회."""
    _check_connection()
    formatted_codes = [_format_code(t) for t in req.tickers]
    logger.info(f"Batch info request: {len(formatted_codes)} stocks")

    try:
        data = agent.fetch_multi_stock_info(formatted_codes)
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dostk/universe")
async def get_universe():
    """KOSPI + KOSDAQ 전체 종목 코드 목록."""
    _check_connection()

    try:
        data = agent.get_equity_universe()
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════
# 해외 주식/지수 엔드포인트
# ══════════════════════════════════════════

@app.get("/api/overseas/chart")
async def get_overseas_chart(code: str, count: int = 500):
    """해외 주식/지수/환율/원자재 과거 일봉 조회."""
    _check_connection()
    logger.info(f"Overseas chart request: {code}, count={count}")

    try:
        data = agent.get_overseas_chart(code, count)
        if data is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve overseas chart.")
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/overseas/quote")
async def get_overseas_quote(code: str):
    """해외 주식/지수 현재 시세 조회."""
    _check_connection()
    logger.info(f"Overseas quote request: {code}")

    try:
        data = agent.get_overseas_current(code)
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════

def _check_connection():
    """Cybos Plus 연결 상태 확인"""
    if agent.cybos is None or agent.cybos.IsConnect != 1:
        success = agent.wait_for_login(timeout=5)
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Daishin HTS is not connected. Please log in manually."
            )


def _format_code(stk_cd: str) -> str:
    """종목코드에 'A' 접두사 보장"""
    return stk_cd if stk_cd.startswith("A") else f"A{stk_cd}"


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on port 8000...")
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
