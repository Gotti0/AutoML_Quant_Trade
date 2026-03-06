"""
AutoML_Quant_Trade - 백테스트 전용 Parquet 캐시

SQLite DB의 시장 데이터를 Apache Parquet 포맷으로 캐싱하여
백테스트 시작 시 데이터 로딩 속도를 극적으로 향상.

■ 최초 1회: SQLite → Parquet 변환 (수십 초)
■ 이후: Parquet → PyArrow Memory Map → {ticker: DataFrame} (수 초)
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# PyArrow는 선택적 종속성
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("pyarrow not installed. Parquet cache disabled. pip install pyarrow")


class ParquetCache:
    """백테스트 전용 Parquet 캐시 관리자"""

    CACHE_DIR = Path("cache_daishin/parquet")
    DOMESTIC_FILE = "domestic_daily.parquet"
    OVERSEAS_FILE = "overseas_daily.parquet"

    @classmethod
    def is_available(cls) -> bool:
        """Parquet 캐시 사용 가능 여부."""
        return PYARROW_AVAILABLE

    @classmethod
    def cache_exists(cls) -> bool:
        """캐시 파일이 존재하는지 확인."""
        domestic = cls.CACHE_DIR / cls.DOMESTIC_FILE
        return domestic.exists()

    @classmethod
    def build_cache(cls, db, min_volume_domestic: int = 50000,
                    min_volume_overseas: int = 10000) -> None:
        """
        SQLite DB에서 시장 데이터를 읽어 Parquet 캐시를 생성.
        
        Parameters:
            db: DatabaseManager 인스턴스
            min_volume_domestic: 국내 종목 최소 거래량 필터
            min_volume_overseas: 해외 종목 최소 거래량 필터
        """
        if not PYARROW_AVAILABLE:
            logger.error("pyarrow not installed. Cannot build Parquet cache.")
            return

        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # --- 국내 주식 ---
        logger.info("Building domestic Parquet cache...")
        domestic_df = db.query_dataframe(
            f"SELECT ticker, date, open, high, low, close, volume "
            f"FROM stock_daily WHERE ticker IN "
            f"(SELECT ticker FROM stock_daily GROUP BY ticker "
            f"HAVING max(volume) >= {min_volume_domestic}) "
            f"ORDER BY date, ticker"
        )
        if not domestic_df.empty:
            table = pa.Table.from_pandas(domestic_df, preserve_index=False)
            pq.write_table(table, cls.CACHE_DIR / cls.DOMESTIC_FILE,
                           compression='snappy')
            logger.info(f"Domestic cache: {len(domestic_df)} rows, "
                        f"{domestic_df['ticker'].nunique()} tickers")

        # --- 해외 자산 ---
        logger.info("Building overseas Parquet cache...")
        overseas_df = db.query_dataframe(
            f"SELECT code AS ticker, date, open, high, low, close, volume "
            f"FROM overseas_daily WHERE code IN "
            f"(SELECT code FROM overseas_daily GROUP BY code "
            f"HAVING max(volume) >= {min_volume_overseas}) "
            f"ORDER BY date, code"
        )
        if not overseas_df.empty:
            table = pa.Table.from_pandas(overseas_df, preserve_index=False)
            pq.write_table(table, cls.CACHE_DIR / cls.OVERSEAS_FILE,
                           compression='snappy')
            logger.info(f"Overseas cache: {len(overseas_df)} rows, "
                        f"{overseas_df['ticker'].nunique()} tickers")

        logger.info("Parquet cache build complete.")

    @classmethod
    def load_market_data(cls, include_overseas: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Parquet 캐시에서 시장 데이터를 고속 로드.
        
        Returns:
            {ticker: DataFrame[date, open, high, low, close, volume]}
        """
        if not PYARROW_AVAILABLE:
            raise RuntimeError("pyarrow not installed.")

        market_data = {}

        # 국내
        domestic_path = cls.CACHE_DIR / cls.DOMESTIC_FILE
        if domestic_path.exists():
            logger.info("Loading domestic data from Parquet cache...")
            df = pq.read_table(domestic_path).to_pandas()
            for ticker, group in df.groupby("ticker"):
                market_data[ticker] = group.drop(columns=["ticker"]).reset_index(drop=True)
            logger.info(f"Domestic: {len(market_data)} tickers loaded")

        # 해외
        if include_overseas:
            overseas_path = cls.CACHE_DIR / cls.OVERSEAS_FILE
            if overseas_path.exists():
                logger.info("Loading overseas data from Parquet cache...")
                df = pq.read_table(overseas_path).to_pandas()
                count_before = len(market_data)
                for ticker, group in df.groupby("ticker"):
                    market_data[ticker] = group.drop(columns=["ticker"]).reset_index(drop=True)
                logger.info(f"Overseas: {len(market_data) - count_before} tickers loaded")

        return market_data
