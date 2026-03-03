"""
AutoML_Quant_Trade - 해외 주식/지수/환율/원자재 수집기

CpSvrNew8300을 통해 해외 자산의 과거 일봉 시계열을 수집하고
SQLite 데이터베이스에 저장.
"""
import logging
from typing import List

from backend.config.settings import Settings
from backend.data.bridge_client import BridgeClient
from backend.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class OverseasCollector:
    """해외 주식/지수/환율/원자재 과거 일봉 수집기 (CpSvrNew8300)"""

    def __init__(self, db: DatabaseManager = None, client: BridgeClient = None):
        self.db = db or DatabaseManager()
        self.client = client or BridgeClient()

    def collect_chart(self, code: str, count: int = 500) -> int:
        """
        단일 해외 종목/지수의 과거 일봉 수집.

        Parameters:
            code: 해외 코드 (예: 'AAPL', '.DJI', 'DS#USDKRW', 'CM@PWTI')
            count: 수집할 일봉 개수
        Returns:
            수집한 행 수
        """
        try:
            df = self.client.fetch_overseas_chart(code, count)
            if not df.empty:
                self.db.upsert_overseas_daily(code, df)
                logger.info(f"Collected {len(df)} overseas daily rows for {code}")
                return len(df)
            else:
                logger.warning(f"No overseas data returned for {code}")
                return 0
        except Exception as e:
            logger.error(f"Failed to collect overseas chart for {code}: {e}")
            raise

    def collect_batch(self, codes: List[str], count: int = 500) -> int:
        """
        다수 해외 종목/지수를 일괄 수집.

        Parameters:
            codes: 해외 코드 리스트
            count: 종목당 수집할 일봉 개수
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0

        for i, code in enumerate(codes):
            try:
                rows = self.collect_chart(code, count)
                if rows > 0:
                    success_count += 1

                if (i + 1) % 20 == 0:
                    logger.info(f"Overseas progress: {i + 1}/{len(codes)} collected")

            except Exception as e:
                logger.error(f"Failed overseas batch for {code}: {e}")
                continue

        logger.info(f"Overseas batch collection: {success_count}/{len(codes)} succeeded")
        return success_count

    def collect_incremental(self, codes: List[str]) -> int:
        """
        증분 수집: 마지막 수집일 이후 데이터만 추가.

        Parameters:
            codes: 해외 코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0

        for code in codes:
            try:
                last_date = self.db.get_last_date("overseas_daily", "code", code)
                if last_date is not None:
                    df = self.client.fetch_overseas_chart(code, count=30)
                else:
                    df = self.client.fetch_overseas_chart(code, count=Settings.DEFAULT_DAILY_COUNT)

                if not df.empty:
                    self.db.upsert_overseas_daily(code, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed incremental overseas for {code}: {e}")
                continue

        logger.info(f"Overseas incremental: {success_count}/{len(codes)} succeeded")
        return success_count
