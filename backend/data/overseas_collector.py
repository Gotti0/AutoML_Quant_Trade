"""
AutoML_Quant_Trade - 해외 주식/지수/환율/원자재 수집기

CpSvrNew8300을 통해 해외 자산의 과거 일봉 시계열을 수집하고
SQLite 데이터베이스에 저장.
"""
import logging
import time
from datetime import datetime
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

    def collect_insert(self, codes: List[str]) -> int:
        """
        신규 수집: DB에 마지막 수집일이 없는 종목만 전체 수집.

        Parameters:
            codes: 해외 코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0
        skipped_count = 0

        for code in codes:
            try:
                last_date = self.db.get_last_date("overseas_daily", "code", code)
                if last_date is not None:
                    skipped_count += 1
                    continue

                df = self.client.fetch_overseas_chart(code, count=Settings.DEFAULT_DAILY_COUNT)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_overseas_daily(code, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed overseas insert for {code}: {e}")
                continue

        logger.info(f"Overseas Insert: {success_count} succeeded, {skipped_count} skipped")
        return success_count

    def collect_update(self, codes: List[str]) -> int:
        """
        증분 수집: 마지막 수집일과 오늘 사이의 공백 일수를 계산하여
        필요한 만큼만 최신 데이터를 추가 수집.

        Parameters:
            codes: 해외 코드 리스트
        Returns:
            수집 성공한 종목 수
        """
        success_count = 0
        skipped_count = 0
        today = datetime.now()

        for code in codes:
            try:
                last_date = self.db.get_last_date("overseas_daily", "code", code)
                if last_date is None:
                    skipped_count += 1
                    continue

                # 공백 일수 기반 동적 수집 건수 계산
                last_dt = datetime.strptime(str(last_date), "%Y%m%d")
                gap_days = (today - last_dt).days
                fetch_count = max(30, min(gap_days + 10, Settings.DEFAULT_DAILY_COUNT))

                df = self.client.fetch_overseas_chart(code, count=fetch_count)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_overseas_daily(code, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed overseas update for {code}: {e}")
                continue

        logger.info(f"Overseas Update: {success_count} succeeded, {skipped_count} skipped")
        return success_count
