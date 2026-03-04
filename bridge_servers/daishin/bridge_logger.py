"""
32비트 브릿지 서버용 SQLite 기반 로거.
백엔드 쪽 파일 시스템(경로)이 다를 수 있어 스탠드어론으로 심플하게 포팅합니다.
"""
import logging
import sqlite3
import queue
import threading
import os

class BridgeSQLiteLogHandler(logging.Handler):
    def __init__(self, db_path: str, source: str = "bridge"):
        super().__init__()
        self.db_path = db_path
        self.source = source
        self._log_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # 테이블이 없을 경우 대비 초기화 (보통 백엔드에서 미리 만들지만 안전상 생성)
        self._init_schema()

        self._writer_thread = threading.Thread(
            target=self._write_logs_loop,
            daemon=True,
            name="BridgeLogWriterThread"
        )
        self._writer_thread.start()

    def _init_schema(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS system_logs ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                    "level TEXT NOT NULL, "
                    "source TEXT NOT NULL, "
                    "message TEXT NOT NULL)"
                )
        except Exception as e:
            print(f"Failed to init bridge log schema: {e}")

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._log_queue.put((record.levelname, getattr(record, 'source', self.source), msg))
        except Exception:
            self.handleError(record)

    def _write_logs_loop(self):
        while not self._stop_event.is_set():
            try:
                level, source, message = self._log_queue.get(timeout=1.0)
                try:
                    with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute(
                            "INSERT INTO system_logs (level, source, message) VALUES (?, ?, ?)",
                            (level, source, message)
                        )
                except Exception as e:
                    print(f"[BridgeSQLiteLogHandler] Failed to write log: {e}")
                finally:
                    self._log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[BridgeSQLiteLogHandler] Error: {e}")

    def close(self):
        self._stop_event.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        super().close()


def setup_bridge_logger(db_path: str) -> logging.Logger:
    logger = logging.getLogger("bridge")
    if any(isinstance(h, BridgeSQLiteLogHandler) for h in logger.handlers):
        return logger

    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # DB 핸들러
    db_handler = BridgeSQLiteLogHandler(db_path=db_path)
    db_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(db_handler)

    return logger
