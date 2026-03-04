import logging
import sqlite3
import queue
import threading
from typing import Optional

from backend.data.database import DatabaseManager

class SQLiteLogHandler(logging.Handler):
    """
    백엔드/브릿지 서버의 로그를 SQLite DB `system_logs` 테이블에 기록하는 커스텀 핸들러.
    순차적 쓰기 작업을 위해 내부 백그라운드 스레드와 Queue를 사용합니다.
    (FastAPI나 브릿지 서버의 비동기 흐름을 블로킹하지 않기 위함)
    """

    def __init__(self, db_manager: DatabaseManager, source: str):
        super().__init__()
        self.db = db_manager
        self.source = source
        self._log_queue = queue.Queue()
        self._stop_event = threading.Event()

        # 백그라운드 쓰기 스레드 시작
        self._writer_thread = threading.Thread(
            target=self._write_logs_loop,
            daemon=True,
            name=f"LogWriterThread-{source}"
        )
        self._writer_thread.start()

    def emit(self, record: logging.LogRecord):
        """로거 캐치 시 큐에 적재"""
        try:
            msg = self.format(record)
            self._log_queue.put((record.levelname, getattr(record, 'source', self.source), msg))
        except Exception:
            self.handleError(record)

    def _write_logs_loop(self):
        """큐에서 로그를 꺼내 순차적으로 DB에 기록하는 루프"""
        while not self._stop_event.is_set():
            try:
                # 큐에서 데이터를 가져올 때 무한정 기다리지 않음(1초 타임아웃)
                level, source, message = self._log_queue.get(timeout=1.0)
                
                try:
                    self.db.insert_log(level=level, source=source, message=message)
                except sqlite3.DatabaseError as e:
                    # SQLite 쓰기 에러 발생 시 콘솔에만 출력 (재귀 방지)
                    print(f"[SQLiteLogHandler] Failed to write log: {e}")
                finally:
                    self._log_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SQLiteLogHandler] Unexpected error in writer thread: {e}")

    def close(self):
        """핸들러 종료 시 스레드 정리"""
        self._stop_event.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        super().close()


def setup_integrated_logger(db_manager: DatabaseManager, source: str) -> logging.Logger:
    """
    통합 로거(console + DB) 설정. 
    기존에 설정된 로거가 있다면 중복되지 않게 Handler를 추가합니다.
    """
    logger = logging.getLogger(source)
    
    # 이미 해당 소스의 핸들러가 구성되어 있는지 검사
    if any(isinstance(h, SQLiteLogHandler) for h in logger.handlers):
        return logger

    # 레벨 설정 (기본 INFO)
    logger.setLevel(logging.INFO)

    # 1. 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 2. 통합 DB 핸들러
    db_handler = SQLiteLogHandler(db_manager=db_manager, source=source)
    # DB 핸들러는 메시지만 순수하게 저장 (시간, 레벨, 소스는 스키마 레벨에서 관리)
    db_format = logging.Formatter('%(message)s')
    db_handler.setFormatter(db_format)
    logger.addHandler(db_handler)

    return logger
