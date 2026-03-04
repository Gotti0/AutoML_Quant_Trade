"""
AutoML_Quant_Trade - SQLite 데이터베이스 관리자

ACID 트랜잭션, UNIQUE 제약, 인덱스를 통해
데이터 일관성·정합성을 보장하는 중앙 데이터 저장소.
"""
import sqlite3
import logging
from contextlib import contextmanager
from typing import Optional, List

import pandas as pd

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 기반 중앙 데이터베이스 관리자"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Settings.DB_PATH
        Settings.ensure_dirs()
        self._init_schema()

    def _init_schema(self):
        """테이블이 없으면 자동 생성"""
        with self._connection() as conn:
            cursor = conn.cursor()

            # 1. 국내 주식 일봉
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_daily (
                    ticker    TEXT    NOT NULL,
                    date      INTEGER NOT NULL,
                    open      REAL    NOT NULL,
                    high      REAL    NOT NULL,
                    low       REAL    NOT NULL,
                    close     REAL    NOT NULL,
                    volume    INTEGER NOT NULL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_daily_date
                ON stock_daily(date)
            """)

            # 2. 국내 주식 분봉
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_minute (
                    ticker    TEXT    NOT NULL,
                    date      INTEGER NOT NULL,
                    time      INTEGER NOT NULL,
                    open      REAL    NOT NULL,
                    high      REAL    NOT NULL,
                    low       REAL    NOT NULL,
                    close     REAL    NOT NULL,
                    volume    INTEGER NOT NULL,
                    PRIMARY KEY (ticker, date, time)
                )
            """)

            # 3. 해외 주식/지수 일봉 (CpSvrNew8300)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS overseas_daily (
                    code      TEXT    NOT NULL,
                    date      INTEGER NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL    NOT NULL,
                    volume    REAL,
                    PRIMARY KEY (code, date)
                )
            """)

            # 4. 거시지표 일별
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_daily (
                    indicator TEXT    NOT NULL,
                    code      TEXT    NOT NULL,
                    date      INTEGER NOT NULL,
                    close     REAL    NOT NULL,
                    change    REAL,
                    PRIMARY KEY (indicator, date)
                )
            """)

            # 5. 시스템 통합 로그
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level     TEXT    NOT NULL,
                    source    TEXT    NOT NULL,
                    message   TEXT    NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_logs_source_level
                ON system_logs(source, level)
            """)

            logger.info(f"Database schema initialized: {self.db_path}")

    @contextmanager
    def _connection(self):
        """트랜잭션 스코프 관리 — 예외 시 자동 롤백"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")     # 동시 읽기 성능 향상
        conn.execute("PRAGMA synchronous=NORMAL")   # 성능과 안전성 균형
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ══════════════════════════════════════════
    # 쓰기 (INSERT OR IGNORE → 중복 무시)
    # ══════════════════════════════════════════

    def upsert_stock_daily(self, ticker: str, df: pd.DataFrame):
        """
        국내 주식 일봉 데이터 삽입.
        기존에 같은 (ticker, date) 행이 있으면 무시.

        Parameters:
            ticker: 종목코드 (예: 'A005930')
            df: columns=[date, open, high, low, close, volume]
        """
        if df.empty:
            return

        records = [
            (ticker, int(row["date"]), float(row["open"]), float(row["high"]),
             float(row["low"]), float(row["close"]), int(row["volume"]))
            for _, row in df.iterrows()
        ]

        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO stock_daily "
                "(ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                records
            )
            logger.info(f"Upserted {len(records)} daily rows for {ticker}")

    def upsert_stock_minute(self, ticker: str, df: pd.DataFrame):
        """
        국내 주식 분봉 데이터 삽입.

        Parameters:
            ticker: 종목코드
            df: columns=[date, time, open, high, low, close, volume]
        """
        if df.empty:
            return

        records = [
            (ticker, int(row["date"]), int(row["time"]),
             float(row["open"]), float(row["high"]),
             float(row["low"]), float(row["close"]), int(row["volume"]))
            for _, row in df.iterrows()
        ]

        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO stock_minute "
                "(ticker, date, time, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            logger.info(f"Upserted {len(records)} minute rows for {ticker}")

    def upsert_overseas_daily(self, code: str, df: pd.DataFrame):
        """
        해외 주식/지수 일봉 데이터 삽입.

        Parameters:
            code: 해외 코드 (예: 'AAPL', '.DJI', 'DS#USDKRW')
            df: columns=[date, open, high, low, close, volume]
        """
        if df.empty:
            return

        records = [
            (code, int(row["date"]),
             float(row["open"]) if pd.notna(row.get("open")) else None,
             float(row["high"]) if pd.notna(row.get("high")) else None,
             float(row["low"]) if pd.notna(row.get("low")) else None,
             float(row["close"]),
             float(row["volume"]) if pd.notna(row.get("volume")) else None)
            for _, row in df.iterrows()
        ]

        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO overseas_daily "
                "(code, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                records
            )
            logger.info(f"Upserted {len(records)} overseas daily rows for {code}")

    def upsert_macro_daily(self, indicator: str, code: str, df: pd.DataFrame):
        """
        거시지표 일별 데이터 삽입.

        Parameters:
            indicator: 지표명 (예: '다우존스')
            code: API 코드 (예: '.DJI')
            df: columns=[date, close, change(optional)]
        """
        if df.empty:
            return

        records = [
            (indicator, code, int(row["date"]), float(row["close"]),
             float(row["change"]) if pd.notna(row.get("change")) else None)
            for _, row in df.iterrows()
        ]

        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO macro_daily "
                "(indicator, code, date, close, change) "
                "VALUES (?, ?, ?, ?, ?)",
                records
            )
            logger.info(f"Upserted {len(records)} macro rows for {indicator}")

    def insert_log(self, level: str, source: str, message: str):
        """
        시스템 통합 로그 단건 삽입.
        (주로 Backend/Bridge SQLiteLogHandler 나 Frontend API 연동 시 사용)
        """
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO system_logs (level, source, message) VALUES (?, ?, ?)",
                (level, source, message)
            )

    def load_recent_logs(self, limit: int = 100, source: Optional[str] = None):
        """
        최신 시스템 로그 조회.
        """
        query = "SELECT id, timestamp, level, source, message FROM system_logs"
        params = []
        
        if source:
            query += " WHERE source = ?"
            params.append(source)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df


    # ══════════════════════════════════════════
    # 읽기 (기간 필터링 + DataFrame 반환)
    # ══════════════════════════════════════════

    def query_dataframe(self, query: str) -> pd.DataFrame:
        """임의의 SELECT 쿼리를 실행하여 DataFrame 반환."""
        with self._connection() as conn:
            return pd.read_sql_query(query, conn)

    def load_stock_daily(self, ticker: str,
                         start_date: int = None,
                         end_date: int = None) -> pd.DataFrame:
        """국내 주식 일봉 조회. 기간 필터링 지원."""
        query = "SELECT date, open, high, low, close, volume FROM stock_daily WHERE ticker = ?"
        params: list = [ticker]

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def load_stock_minute(self, ticker: str,
                          start_date: int = None,
                          end_date: int = None) -> pd.DataFrame:
        """국내 주식 분봉 조회."""
        query = "SELECT date, time, open, high, low, close, volume FROM stock_minute WHERE ticker = ?"
        params: list = [ticker]

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC, time ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def load_overseas_daily(self, code: str,
                            start_date: int = None,
                            end_date: int = None) -> pd.DataFrame:
        """해외 주식/지수 일봉 조회."""
        query = "SELECT date, open, high, low, close, volume FROM overseas_daily WHERE code = ?"
        params: list = [code]

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def load_macro_all(self, start_date: int = None,
                       end_date: int = None) -> pd.DataFrame:
        """
        전체 거시지표를 피벗하여 반환.
        반환 형태: index=date, columns=[다우존스, 나스닥, S&P500, USD/KRW, ...]
        """
        query = "SELECT indicator, date, close FROM macro_daily WHERE 1=1"
        params: list = []

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        # 롱 포맷 → 와이드 포맷 피벗
        pivot_df = df.pivot(index="date", columns="indicator", values="close")
        pivot_df = pivot_df.reset_index()
        return pivot_df

    # ══════════════════════════════════════════
    # 메타 조회
    # ══════════════════════════════════════════

    def get_last_date(self, table: str, key_column: str, key_value: str) -> Optional[int]:
        """
        특정 테이블의 특정 키에 대한 최신 날짜를 반환.
        증분 수집 시 마지막 수집 날짜를 확인하는 데 사용.

        Parameters:
            table: 테이블명 (예: 'stock_daily', 'overseas_daily')
            key_column: 키 컬럼명 (예: 'ticker', 'code')
            key_value: 키 값 (예: 'A005930', 'AAPL')
        """
        # SQL injection 방지: 테이블/컬럼명은 화이트리스트 검증
        allowed_tables = {"stock_daily", "stock_minute", "overseas_daily", "macro_daily"}
        allowed_columns = {"ticker", "code", "indicator"}

        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")
        if key_column not in allowed_columns:
            raise ValueError(f"Invalid key column: {key_column}")

        query = f"SELECT MAX(date) FROM {table} WHERE {key_column} = ?"

        with self._connection() as conn:
            cursor = conn.execute(query, (key_value,))
            result = cursor.fetchone()

        return result[0] if result and result[0] is not None else None

    def list_tickers(self, table: str = "stock_daily") -> List[str]:
        """저장된 종목/코드 목록 반환."""
        allowed_tables = {"stock_daily", "stock_minute", "overseas_daily", "macro_daily"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")

        key_col = {
            "stock_daily": "ticker",
            "stock_minute": "ticker",
            "overseas_daily": "code",
            "macro_daily": "indicator",
        }[table]

        query = f"SELECT DISTINCT {key_col} FROM {table} ORDER BY {key_col}"

        with self._connection() as conn:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def get_row_count(self, table: str) -> int:
        """테이블의 총 행 수 반환."""
        allowed_tables = {"stock_daily", "stock_minute", "overseas_daily", "macro_daily"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")

        with self._connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]
