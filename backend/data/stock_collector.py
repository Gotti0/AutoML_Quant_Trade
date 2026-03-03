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
            finally:
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

        logger.info(f"Daily collection complete: {success_count}/{len(tickers)} succeeded")
        return success_count

    def collect_daily_insert(self, tickers: List[str]) -> int:
        """
        신규 수집: DB에 마지막 수집일이 없는 종목만 전체(DEFAULT) 수집.

        Parameters:
            tickers: 종목코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0
        skipped_count = 0

        for ticker in tickers:
            try:
                last_date = self.db.get_last_date("stock_daily", "ticker", ticker)
                # 이미 데이터가 있으면 이번 Insert 작업에서는 통과(Skip)
                if last_date is not None:
                    skipped_count += 1
                    continue

                df = self.client.fetch_daily_ohlcv(ticker, count=Settings.DEFAULT_DAILY_COUNT)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_stock_daily(ticker, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed daily insert for {ticker}: {e}")
                continue

        logger.info(f"Daily Insert collection: {success_count} succeeded, {skipped_count} skipped")
        return success_count

    def collect_daily_update(self, tickers: List[str]) -> int:
        """
        증분 통용 수집: DB에 마지막 수집일이 있는 종목만 30일치 최신 데이터를 추가 수집.

        Parameters:
            tickers: 종목코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0
        skipped_count = 0

        for ticker in tickers:
            try:
                last_date = self.db.get_last_date("stock_daily", "ticker", ticker)
                # 데이터가 없는 신규 종목이면 이번 Update 작업에서는 통과(Skip)
                if last_date is None:
                    skipped_count += 1
                    continue

                df = self.client.fetch_daily_ohlcv(ticker, count=30)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_stock_daily(ticker, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed daily update for {ticker}: {e}")
                continue

        logger.info(f"Daily Update collection: {success_count} succeeded, {skipped_count} skipped")
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
            finally:
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

        logger.info(f"Minute collection complete: {success_count}/{len(tickers)} succeeded")
        return success_count
