"""
AutoML_Quant_Trade - 거시지표 수집기

OverseasCollector를 활용하여 해외 지수·환율·원자재의 과거 일봉을
수집하고, 단일 거시지표 시계열로 통합.
"""
import logging
import time
from typing import Dict

import pandas as pd

from backend.config.settings import Settings
from backend.data.bridge_client import BridgeClient
from backend.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class MacroCollector:
    """거시지표 과거 시계열 수집 및 통합"""

    # 수집 대상 거시지표
    MACRO_CODES: Dict[str, str] = Settings.MACRO_CODES

    def __init__(self, db: DatabaseManager = None, client: BridgeClient = None):
        self.db = db or DatabaseManager()
        self.client = client or BridgeClient()

    def collect_all(self, count: int = 500) -> int:
        """
        전체 거시지표의 과거 시계열을 일괄 수집.

        Parameters:
            count: 지표당 수집할 일봉 개수
        Returns:
            수집 성공한 지표 수
        """
        success_count = 0

        for indicator, code in self.MACRO_CODES.items():
            try:
                df = self.client.fetch_overseas_chart(code, count)

                if not df.empty:
                    self.db.upsert_macro_daily(indicator, code, df)
                    success_count += 1
                    logger.info(f"Collected {len(df)} macro rows for {indicator} ({code})")
                else:
                    logger.warning(f"No macro data returned for {indicator} ({code})")

            except Exception as e:
                logger.error(f"Failed to collect macro for {indicator} ({code}): {e}")
                continue

        logger.info(f"Macro collection complete: {success_count}/{len(self.MACRO_CODES)} succeeded")
        return success_count

    def collect_insert(self) -> int:
        """
        신규 수집: 마지막 수집일이 없는 지표만 전체 수집.

        Returns:
            수집 성공한 지표 수
        """
        success_count = 0
        skipped_count = 0

        for indicator, code in self.MACRO_CODES.items():
            try:
                last_date = self.db.get_last_date("macro_daily", "indicator", indicator)

                if last_date is not None:
                    skipped_count += 1
                    continue

                df = self.client.fetch_overseas_chart(code, count=Settings.DEFAULT_DAILY_COUNT)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_macro_daily(indicator, code, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed macro insert for {indicator}: {e}")
                continue

        logger.info(f"Macro Insert: {success_count} succeeded, {skipped_count} skipped")
        return success_count

    def collect_update(self) -> int:
        """
        증분 수집: 마지막 수집일 이후 데이터만 추가.

        Returns:
            수집 성공한 지표 수
        """
        success_count = 0
        skipped_count = 0

        for indicator, code in self.MACRO_CODES.items():
            try:
                last_date = self.db.get_last_date("macro_daily", "indicator", indicator)

                if last_date is None:
                    skipped_count += 1
                    continue

                df = self.client.fetch_overseas_chart(code, count=30)
                time.sleep(Settings.CYBOS_THROTTLE_WAIT)

                if not df.empty:
                    self.db.upsert_macro_daily(indicator, code, df)
                    success_count += 1

            except Exception as e:
                logger.error(f"Failed macro update for {indicator}: {e}")
                continue

        logger.info(f"Macro Update: {success_count} succeeded, {skipped_count} skipped")
        return success_count

    def load_history(self, start_date: int = None,
                     end_date: int = None) -> pd.DataFrame:
        """
        축적된 거시지표 시계열을 와이드 포맷 DataFrame으로 로드.

        Returns:
            DataFrame: index=date, columns=[다우존스, 나스닥, S&P500, USD/KRW, WTI원유, ...]
        """
        return self.db.load_macro_all(start_date=start_date, end_date=end_date)
