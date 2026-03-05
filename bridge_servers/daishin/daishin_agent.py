"""
AutoML_Quant_Trade - Cybos Plus COM 객체 래퍼

32비트 Python 환경에서 동작하는 대신증권 Cybos Plus API 통합 래퍼.
bridge_server.py가 이 모듈을 직접 import하여 사용.

주의: 이 모듈은 반드시 32비트 Python + Cybos Plus HTS 로그인 상태에서 실행해야 합니다.
"""
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# win32com은 32비트 환경에서만 사용 가능
try:
    from win32com.client import Dispatch
    HAS_COM = True
except ImportError:
    HAS_COM = False
    logger.warning("win32com not available — COM agent will not function")


class DaishinAgent:
    """Cybos Plus COM 객체 통합 래퍼"""

    # API 호출 제한: 15초당 최대 60건 → 약 0.25초 간격
    THROTTLE_INTERVAL = 0.25

    def __init__(self):
        if not HAS_COM:
            self.cybos = None
            return

        self.cybos = Dispatch("CpUtil.CpCybos")
        self.code_mgr = Dispatch("CpUtil.CpCodeMgr")
        self.chart = Dispatch("CpSysDib.StockChart")
        self.market_eye = Dispatch("CpSysDib.MarketEye")
        self.us_code = Dispatch("CpUtil.CpUsCode")
        self.overseas = Dispatch("CpSysDib.CpSvrNew8300")

    def wait_for_login(self, timeout: int = 600) -> bool:
        """
        Cybos Plus HTS 로그인 대기.

        Parameters:
            timeout: 최대 대기 시간 (초)
        Returns:
            연결 성공 여부
        """
        if not HAS_COM or self.cybos is None:
            logger.error("COM objects not available")
            return False

        start = time.time()
        while time.time() - start < timeout:
            if self.cybos.IsConnect == 1:
                logger.info("Cybos Plus connected successfully")
                return True
            logger.info("Waiting for Cybos Plus login...")
            time.sleep(5)

        logger.error(f"Cybos Plus login timeout after {timeout}s")
        return False

    def _wait_for_request(self):
        """남은 요청 횟수를 확인하고 필요 시 대기"""
        remain = self.cybos.GetLimitRemainCount(1)  # 1: 시세 조회
        if remain <= 0:
            logger.debug("API throttle limit reached, waiting...")
            time.sleep(self.THROTTLE_INTERVAL)

    # ══════════════════════════════════════════
    # 국내 주식 데이터
    # ══════════════════════════════════════════

    def get_daily_ohlcv(self, ticker: str, count: int = 500) -> Optional[List[dict]]:
        """
        국내 주식 일봉 OHLCV 조회 (StockChart).

        Parameters:
            ticker: 종목코드 (예: 'A005930')
            count: 수집할 일봉 개수
        Returns:
            List[dict] with keys: date, open, high, low, close, volume
        """
        self._wait_for_request()

        self.chart.SetInputValue(0, ticker)
        self.chart.SetInputValue(1, ord("2"))     # 1: 기간, 2: 개수
        self.chart.SetInputValue(4, count)
        self.chart.SetInputValue(5, [0, 2, 3, 4, 5, 8])  # 날짜, 시가, 고가, 저가, 종가, 거래량
        self.chart.SetInputValue(6, ord("D"))     # D: 일봉
        self.chart.SetInputValue(9, ord("1"))     # 수정주가

        result = []
        total_received = 0

        while total_received < count:
            self.chart.BlockRequest()

            status = self.chart.GetDibStatus()
            if status != 0:
                logger.error(f"StockChart error: status={status}, msg={self.chart.GetDibMsg1()}")
                break

            received = self.chart.GetHeaderValue(3)
            if received == 0:
                break

            for i in range(received):
                row = {
                    "date": self.chart.GetDataValue(0, i),
                    "open": self.chart.GetDataValue(1, i),
                    "high": self.chart.GetDataValue(2, i),
                    "low": self.chart.GetDataValue(3, i),
                    "close": self.chart.GetDataValue(4, i),
                    "volume": self.chart.GetDataValue(5, i),
                }
                result.append(row)

            total_received += received

            if not self.chart.Continue:
                break

            self._wait_for_request()

        logger.info(f"Retrieved {len(result)} daily rows for {ticker}")
        return result

    def get_minute_chart(self, ticker: str, count: int = 5000,
                         since_date: int = None,
                         since_time: int = None) -> Optional[List[dict]]:
        """
        국내 주식 분봉 데이터 조회 (StockChart).

        Parameters:
            ticker: 종목코드
            count: 수집할 분봉 개수
            since_date: 이 날짜 이후 데이터만 (YYYYMMDD)
            since_time: since_date와 함께 사용 (HHMM)
        Returns:
            List[dict] with keys: date, time, open, high, low, close, volume
        """
        self._wait_for_request()

        self.chart.SetInputValue(0, ticker)
        self.chart.SetInputValue(1, ord("2"))
        self.chart.SetInputValue(4, count)
        self.chart.SetInputValue(5, [0, 1, 2, 3, 4, 5, 8])  # 날짜, 시간, OHLCV
        self.chart.SetInputValue(6, ord("m"))     # m: 분봉
        self.chart.SetInputValue(9, ord("1"))

        result = []
        total_received = 0

        while total_received < count:
            self.chart.BlockRequest()

            status = self.chart.GetDibStatus()
            if status != 0:
                logger.error(f"StockChart minute error: status={status}")
                break

            received = self.chart.GetHeaderValue(3)
            if received == 0:
                break

            for i in range(received):
                row_date = self.chart.GetDataValue(0, i)
                row_time = self.chart.GetDataValue(1, i)

                # since_date/time 필터
                if since_date is not None:
                    if row_date < since_date:
                        continue
                    if row_date == since_date and since_time is not None:
                        if row_time <= since_time:
                            continue

                row = {
                    "date": row_date,
                    "time": row_time,
                    "open": self.chart.GetDataValue(2, i),
                    "high": self.chart.GetDataValue(3, i),
                    "low": self.chart.GetDataValue(4, i),
                    "close": self.chart.GetDataValue(5, i),
                    "volume": self.chart.GetDataValue(6, i),
                }
                result.append(row)

            total_received += received

            if not self.chart.Continue:
                break

            self._wait_for_request()

        logger.info(f"Retrieved {len(result)} minute rows for {ticker}")
        return result

    def get_stock_info(self, ticker: str) -> dict:
        """종목 메타 정보 조회 (MarketEye)."""
        self._wait_for_request()

        # MarketEye 필드: 0=코드, 4=종목명, 20=시가총액, 23=업종
        fields = [0, 4, 20, 23]
        self.market_eye.SetInputValue(0, fields)
        self.market_eye.SetInputValue(1, [ticker])
        self.market_eye.BlockRequest()

        if self.market_eye.GetDibStatus() != 0:
            return {}

        return {
            "ticker": self.market_eye.GetDataValue(0, 0),
            "name": self.market_eye.GetDataValue(1, 0),
            "market_cap": self.market_eye.GetDataValue(2, 0),
            "sector": self.market_eye.GetDataValue(3, 0),
        }

    def fetch_multi_stock_info(self, tickers: List[str]) -> List[dict]:
        """최대 200개 종목 일괄 메타 정보 조회 (MarketEye)."""
        self._wait_for_request()

        fields = [0, 4, 20, 23]
        self.market_eye.SetInputValue(0, fields)
        self.market_eye.SetInputValue(1, tickers)
        self.market_eye.BlockRequest()

        if self.market_eye.GetDibStatus() != 0:
            return []

        count = self.market_eye.GetHeaderValue(2)
        result = []
        for i in range(count):
            result.append({
                "ticker": self.market_eye.GetDataValue(0, i),
                "name": self.market_eye.GetDataValue(1, i),
                "market_cap": self.market_eye.GetDataValue(2, i),
                "sector": self.market_eye.GetDataValue(3, i),
            })
        return result

    # ══════════════════════════════════════════
    # 해외 주식/지수 데이터 (CpSvrNew8300)
    # ══════════════════════════════════════════

    def get_overseas_chart(self, code: str, count: int = 500) -> Optional[List[dict]]:
        """
        해외 주식/지수/환율/원자재 과거 일봉 조회 (CpSvrNew8300).

        Parameters:
            code: 해외 코드 (예: 'AAPL', '.DJI', 'DS#USDKRW', 'CM@PWTI')
            count: 수집할 일봉 개수
        Returns:
            List[dict] with keys: date, open, high, low, close, volume
        """
        self._wait_for_request()

        self.overseas.SetInputValue(0, code)
        self.overseas.SetInputValue(1, ord("2"))   # 개수 기반
        self.overseas.SetInputValue(4, count)
        self.overseas.SetInputValue(5, [0, 1, 2, 3, 4, 6])  # 날짜, OHLCV
        self.overseas.SetInputValue(6, ord("D"))

        result = []
        total_received = 0

        while total_received < count:
            self.overseas.BlockRequest()

            status = self.overseas.GetDibStatus()
            if status != 0:
                logger.error(f"CpSvrNew8300 chart error: status={status}, code={code}")
                break

            received = self.overseas.GetHeaderValue(3)
            if received == 0:
                break

            for i in range(received):
                row = {
                    "date": self.overseas.GetDataValue(0, i),
                    "open": self.overseas.GetDataValue(1, i),
                    "high": self.overseas.GetDataValue(2, i),
                    "low": self.overseas.GetDataValue(3, i),
                    "close": self.overseas.GetDataValue(4, i),
                    "volume": self.overseas.GetDataValue(5, i),
                }
                result.append(row)

            total_received += received

            if not self.overseas.Continue:
                break

            self._wait_for_request()

        logger.info(f"Retrieved {len(result)} overseas daily rows for {code}")
        return result

    def get_overseas_current(self, code: str) -> dict:
        """해외 주식/지수 현재 시세 조회 (CpSvrNew8300 HeaderValue)."""
        self._wait_for_request()

        self.overseas.SetInputValue(0, code)
        self.overseas.BlockRequest()

        if self.overseas.GetDibStatus() != 0:
            return {}

        return {
            "code": code,
            "name": self.overseas.GetHeaderValue(0),
            "close": self.overseas.GetHeaderValue(4),
            "change": self.overseas.GetHeaderValue(6),
            "volume": self.overseas.GetHeaderValue(9),
        }

    # ══════════════════════════════════════════
    # 유니버스
    # ══════════════════════════════════════════

    def get_equity_universe(self) -> List[str]:
        """KOSPI + KOSDAQ 전체 종목 코드 목록."""
        result = []
        # 1: KOSPI, 2: KOSDAQ
        for market in [1, 2]:
            codes = self.code_mgr.GetStockListByMarket(market)
            result.extend(list(codes))
        # 종목 코드에 'A' 접두사 확인
        return [f"A{c}" if not c.startswith("A") else c for c in result]

    def get_overseas_universe(self, us_type: int = 1) -> List[str]:
        """
        해외 종목 코드 목록 조회 (CpUsCode.GetUsCodeList).

        Parameters:
            us_type: 카테고리 코드
                0=금리 등 기타, 1=전체, 2=국가대표지수, 3=업종지수,
                4=해외개별주식, 5=예탁증서(ADR), 6=원자재/반도체, 7=환율
        Returns:
            해당 카테고리의 해외 종목 코드 리스트
        """
        codes = self.us_code.GetUsCodeList(us_type)
        return list(codes) if codes else []

    def get_overseas_code_names(self, codes: List[str]) -> List[dict]:
        """
        해외 종목 코드의 한글명 매핑 조회 (CpUsCode.GetNameByUsCode).

        Parameters:
            codes: 해외 종목 코드 리스트
        Returns:
            [{"code": "AAPL", "name": "애플"}, ...]
        """
        result = []
        for code in codes:
            name = self.us_code.GetNameByUsCode(code)
            result.append({"code": code, "name": name or ""})
        return result

    # ══════════════════════════════════════════
    # 재무 데이터 (MarketEye 기본적분석)
    # ══════════════════════════════════════════

    def get_fundamentals_batch(self, tickers: List[str]) -> List[dict]:
        """
        MarketEye로 재무 데이터 일괄 조회.

        MarketEye 필드 코드:
            0  = 종목코드
            4  = 종목명
            10 = 현재가 (PBR 계산용)
            67 = PER (주가수익비율)
            70 = EPS (주당순이익)
            74 = 배당수익률
            75 = 부채비율
            77 = ROE (자기자본이익률)
            89 = BPS (주당순자산)

        Parameters:
            tickers: 종목코드 리스트 (최대 200개씩 청크 처리)
        Returns:
            List[dict] with keys: ticker, name, price, per, eps,
                                  div_yield, debt_ratio, roe, bps
        """
        CHUNK_SIZE = 200
        FIELDS = [0, 4, 10, 67, 70, 74, 75, 77, 89]
        FIELD_NAMES = ["ticker", "name", "price", "per", "eps",
                       "div_yield", "debt_ratio", "roe", "bps"]

        result = []
        for i in range(0, len(tickers), CHUNK_SIZE):
            chunk = tickers[i:i + CHUNK_SIZE]
            # MarketEye는 오름차순 정렬된 코드 목록을 권장
            chunk_sorted = sorted(chunk)

            self._wait_for_request()
            self.market_eye.SetInputValue(0, FIELDS)
            self.market_eye.SetInputValue(1, chunk_sorted)
            self.market_eye.BlockRequest()

            if self.market_eye.GetDibStatus() != 0:
                logger.error(
                    f"MarketEye fundamentals error: {self.market_eye.GetDibMsg1()}"
                )
                continue

            count = self.market_eye.GetHeaderValue(2)
            for j in range(count):
                row = {}
                for col_idx, name in enumerate(FIELD_NAMES):
                    row[name] = self.market_eye.GetDataValue(col_idx, j)
                result.append(row)

            # 다음 청크 전 스로틀 대기
            if i + CHUNK_SIZE < len(tickers):
                self._wait_for_request()

        logger.info(f"Retrieved fundamentals for {len(result)} stocks")
        return result
