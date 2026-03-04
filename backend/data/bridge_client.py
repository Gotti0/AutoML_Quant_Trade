"""
AutoML_Quant_Trade - 32비트 브릿지 서버 HTTP 클라이언트

64비트 메인 엔진에서 32비트 브릿지 서버를 호출하여
Cybos Plus COM 데이터를 수신하는 HTTP 클라이언트.
"""
import logging
from typing import List, Optional

import httpx
import pandas as pd

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class BridgeClient:
    """32비트 브릿지 서버 HTTP 클라이언트"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or Settings.BRIDGE_URL
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """공통 GET 요청 핸들러"""
        try:
            response = self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise ValueError(f"Bridge server error: {data}")
            return data
        except httpx.ConnectError:
            logger.error(
                "브릿지 서버에 연결할 수 없습니다. "
                "32비트 Python 환경에서 bridge_server.py를 실행하고 "
                "Cybos Plus HTS에 로그인했는지 확인하세요."
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Bridge HTTP error: {e.response.status_code} - {e.response.text}")
            raise

    def _post(self, endpoint: str, json_data: dict = None) -> dict:
        """공통 POST 요청 핸들러"""
        try:
            response = self.client.post(endpoint, json=json_data)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise ValueError(f"Bridge server error: {data}")
            return data
        except httpx.ConnectError:
            logger.error("브릿지 서버에 연결할 수 없습니다.")
            raise

    # ══════════════════════════════════════════
    # 국내 주식 데이터
    # ══════════════════════════════════════════

    def fetch_daily_ohlcv(self, ticker: str, count: int = 500) -> pd.DataFrame:
        """
        국내 주식 일봉 OHLCV 조회.

        Parameters:
            ticker: 종목코드 (예: 'A005930')
            count: 수집할 일봉 개수
        Returns:
            DataFrame[date, open, high, low, close, volume]
        """
        data = self._get("/api/dostk/daily", params={"stk_cd": ticker, "count": count})
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows)

    def fetch_minute_chart(self, ticker: str, count: int = 5000,
                           since_date: int = None,
                           since_time: int = None) -> pd.DataFrame:
        """
        국내 주식 분봉 데이터 조회.

        Parameters:
            ticker: 종목코드
            count: 수집할 분봉 개수
            since_date: 이 날짜 이후 데이터만 (YYYYMMDD)
            since_time: since_date와 함께 사용 (HHMM)
        Returns:
            DataFrame[date, time, open, high, low, close, volume]
        """
        params = {"stk_cd": ticker, "count": count}
        if since_date is not None:
            params["since_date"] = since_date
        if since_time is not None:
            params["since_time"] = since_time

        data = self._get("/api/dostk/chart", params=params)
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows)

    def fetch_stock_info(self, ticker: str) -> dict:
        """종목 메타 정보 (시가총액, 업종, 시장 등) 조회."""
        data = self._get("/api/dostk/info", params={"stk_cd": ticker})
        return data.get("data", {})

    def fetch_stock_info_batch(self, tickers: List[str]) -> List[dict]:
        """최대 200개 종목 일괄 메타 정보 조회."""
        data = self._post("/api/dostk/info_batch", json_data={"tickers": tickers})
        return data.get("data", [])

    # ══════════════════════════════════════════
    # 해외 주식/지수 데이터
    # ══════════════════════════════════════════

    def fetch_overseas_chart(self, code: str, count: int = 500) -> pd.DataFrame:
        """
        해외 주식/지수/환율/원자재 과거 일봉 조회 (CpSvrNew8300).

        Parameters:
            code: 해외 코드 (예: 'AAPL', '.DJI', 'DS#USDKRW', 'CM@PWTI')
            count: 수집할 일봉 개수
        Returns:
            DataFrame[date, open, high, low, close, volume]
        """
        data = self._get("/api/overseas/chart", params={"code": code, "count": count})
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows)

    def fetch_overseas_quote(self, code: str) -> dict:
        """해외 주식/지수 현재 시세 조회."""
        data = self._get("/api/overseas/quote", params={"code": code})
        return data.get("data", {})

    # ══════════════════════════════════════════
    # 유니버스
    # ══════════════════════════════════════════

    def fetch_universe(self) -> List[str]:
        """KOSPI + KOSDAQ 전체 종목 코드 리스트 조회."""
        data = self._get("/api/dostk/universe")
        return data.get("data", [])

    def fetch_overseas_universe(self, us_type: int = 1) -> List[str]:
        """
        해외 종목 코드 목록 조회 (CpUsCode.GetUsCodeList).

        Parameters:
            us_type: 카테고리 코드
                0=금리, 1=전체, 2=국가대표지수, 3=업종지수,
                4=해외개별주식, 5=ADR, 6=원자재, 7=환율
        Returns:
            해당 카테고리의 해외 종목 코드 리스트
        """
        data = self._get("/api/overseas/universe", params={"us_type": us_type})
        return data.get("data", [])

    def close(self):
        """HTTP 클라이언트 연결 해제."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
