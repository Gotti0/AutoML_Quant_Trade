"""
AutoML_Quant_Trade - 국내 주식 OHLCV 수집기

BridgeClient를 통해 Cybos Plus StockChart 일봉/분봉 데이터를
수집하고 SQLite 데이터베이스에 저장.
"""
import logging
import time
from typing import List, Optional

from backend.config.settings import Settings
from backend.data.bridge_client import BridgeClient
from backend.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class StockCollector:
    """국내 주식 OHLCV 수집기"""

    def __init__(self, db: DatabaseManager = None, client: BridgeClient = None):
        self.db = db or DatabaseManager()
        self.client = client or BridgeClient()

    def collect_daily_all(self, tickers: List[str],
                          count: int = None) -> int:
        """
        복수 종목의 일봉 데이터를 일괄 수집하여 DB에 저장.

        Parameters:
            tickers: 종목코드 리스트 (예: ['A005930', 'A000660'])
            count: 종목당 수집할 일봉 개수 (기본: Settings.DEFAULT_DAILY_COUNT)
        Returns:
            수집 성공한 종목 수
        """
        count = count or Settings.DEFAULT_DAILY_COUNT
        success_count = 0

        for i, ticker in enumerate(tickers):
            try:
                df = self.client.fetch_daily_ohlcv(ticker, count)
                if not df.empty:
                    self.db.upsert_stock_daily(ticker, df)
                    success_count += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(tickers)} tickers collected")

            except Exception as e:
                logger.error(f"Failed to collect daily for {ticker}: {e}")
                continue

        logger.info(f"Daily collection complete: {success_count}/{len(tickers)} succeeded")
        return success_count

    def collect_daily_incremental(self, tickers: List[str]) -> int:
        """
        증분 수집: 마지막 수집일 이후 데이터만 추가 수집.

        Parameters:
            tickers: 종목코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0

        for ticker in tickers:
            try:
                last_date = self.db.get_last_date("stock_daily", "ticker", ticker)
                # 마지막 수집일이 있으면 그 이후 데이터만, 없으면 기본 건수
                if last_date is not None:
                    # 최근 30일만 추가 수집 (겹치는 부분은 INSERT OR IGNORE로 무시)
                    df = self.client.fetch_daily_ohlcv(ticker, count=30)
                else:
                    df = self.client.fetch_daily_ohlcv(ticker, count=Settings.DEFAULT_DAILY_COUNT)

                if not df.empty:
                    self.db.upsert_stock_daily(ticker, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed incremental daily for {ticker}: {e}")
                continue

        logger.info(f"Incremental daily collection: {success_count}/{len(tickers)} succeeded")
        return success_count

    def collect_minute(self, tickers: List[str],
                       count: int = None) -> int:
        """
        복수 종목의 분봉 데이터를 일괄 수집하여 DB에 저장.

        Parameters:
            tickers: 종목코드 리스트
            count: 종목당 수집할 분봉 개수
        Returns:
            수집 성공한 종목 수
        """
        count = count or Settings.DEFAULT_MINUTE_COUNT
        success_count = 0

        for i, ticker in enumerate(tickers):
            try:
                df = self.client.fetch_minute_chart(ticker, count)
                if not df.empty:
                    self.db.upsert_stock_minute(ticker, df)
                    success_count += 1

                if (i + 1) % 20 == 0:
                    logger.info(f"Minute progress: {i + 1}/{len(tickers)} tickers collected")

            except Exception as e:
                logger.error(f"Failed to collect minute for {ticker}: {e}")
                continue

        logger.info(f"Minute collection complete: {success_count}/{len(tickers)} succeeded")
        return success_count
