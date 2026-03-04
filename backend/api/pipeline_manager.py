"""
PipelineManager — 파이프라인 실행 상태 관리 + 로그 버퍼 싱글턴

백그라운드 스레드에서 파이프라인을 실행하고,
logging.Handler → asyncio.Queue 브릿지를 통해
SSE 스트림으로 실시간 로그를 전달한다.
"""
import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

import queue
import sys
from typing import Dict, Any, Callable
import subprocess
import os

from backend.data.database import DatabaseManager
from backend.data.bridge_client import BridgeClient
from backend.utils.logger import setup_integrated_logger

# 이 모듈이 임포트될 때 기본 logger 생성 (main과 동일한 방식)
# 단, pipeline_manager 자체의 DB 인스턴스가 없을 수 있으니 초기화 시 주입 필요성 고려
# 여기서는 싱글톤처럼 새로 생성해서 붙임
_db_for_logger = DatabaseManager()
logger = setup_integrated_logger(_db_for_logger, source="backend")


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueLogHandler(logging.Handler):
    """logging → asyncio.Queue 브릿지 (스레드-세이프, OOM 방지)"""

    def __init__(self, log_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.log_queue = log_queue
        self.loop = loop

    def emit(self, record: logging.LogRecord):
        msg = {
            "level": record.levelname,
            "message": self.format(record),
            "timestamp": record.created,
            "logger": record.name,
        }
        try:
            self.loop.call_soon_threadsafe(self.log_queue.put_nowait, msg)
        except asyncio.QueueFull:
            # Ring buffer: 가장 오래된 항목 제거 후 재시도 (OOM 방지)
            try:
                self.log_queue.get_nowait()
                self.loop.call_soon_threadsafe(self.log_queue.put_nowait, msg)
            except Exception:
                pass  # 최악의 경우 로그 1건 유실 — OOM보다 안전


# 유효한 파이프라인 명령어 정의
VALID_COMMANDS = {
    "collect-insert": "신규 종목 전체 수집 (국내+해외+거시)",
    "collect-update": "기존 종목 증분 수집 (국내+해외+거시)",
    "collect-overseas": "해외 자산만 수집",
    "collect-macro": "거시지표만 수집",
    "train-regime": "국면 모델 학습 (Phase 2)",
    "backtest": "백테스팅 실행 (Phase 3)",
}


class PipelineManager:
    """파이프라인 실행 상태 관리 + 로그 버퍼 싱글턴"""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._lock = threading.Lock()
        self._status = PipelineStatus.IDLE
        self._current_task_id: Optional[str] = None
        self._current_command: Optional[str] = None
        self._log_queue: Optional[asyncio.Queue] = None
        self._handler: Optional[QueueLogHandler] = None
        self._error: Optional[str] = None
        self._started_at: Optional[str] = None

    @property
    def status(self) -> PipelineStatus:
        return self._status

    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id

    def get_status_info(self) -> dict:
        """현재 파이프라인 상태 정보 반환"""
        return {
            "status": self._status.value,
            "taskId": self._current_task_id,
            "command": self._current_command,
            "startedAt": self._started_at,
            "error": self._error,
        }

    def start(self, command: str) -> str:
        """
        파이프라인을 백그라운드 스레드에서 시작.

        Returns:
            task_id: 고유 작업 식별자
        Raises:
            RuntimeError: 이미 실행 중인 경우
            ValueError: 유효하지 않은 명령어
        """
        if command not in VALID_COMMANDS:
            raise ValueError(f"Unknown command: {command}. "
                             f"Valid commands: {list(VALID_COMMANDS.keys())}")

        with self._lock:
            if self._status == PipelineStatus.RUNNING:
                raise RuntimeError("Pipeline is already running. "
                                   "Wait for completion or restart the server.")

            task_id = str(uuid.uuid4())[:8]
            self._current_task_id = task_id
            self._current_command = command
            self._status = PipelineStatus.RUNNING
            self._error = None
            self._started_at = datetime.now().isoformat()

            # 로그 큐 생성 (maxsize=2000 — OOM 방지)
            self._log_queue = asyncio.Queue(maxsize=2000)

            # 로깅 핸들러 등록 — 모든 backend.* 로거의 출력을 캡처
            self._handler = QueueLogHandler(self._log_queue, self._loop)
            self._handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
            )
            logging.getLogger("backend").addHandler(self._handler)

        # 백그라운드 스레드에서 파이프라인 실행
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(command, task_id),
            daemon=True,
        )
        thread.start()

        logger.info(f"Pipeline started: command={command}, task_id={task_id}")
        return task_id

    def get_log_stream(self, task_id: str) -> asyncio.Queue:
        """주어진 task_id의 로그 스트림 큐 반환"""
        if self._current_task_id != task_id:
            raise ValueError(f"Unknown task_id: {task_id}")
        if self._log_queue is None:
            raise RuntimeError("Log queue not initialized")
        return self._log_queue

    def _run_pipeline(self, command: str, task_id: str):
        """백그라운드 스레드에서 파이프라인 실행"""
        from backend.data.database import DatabaseManager
        from backend.data.bridge_client import BridgeClient

        db = None
        client = None

        try:
            db = DatabaseManager()

            # 데이터 수집 명령어인 경우에만 Bridge Client 활성화
            is_collect_command = command.startswith("collect-")
            if is_collect_command:
                client = BridgeClient()
                # 브릿지 서버 연결 확인
                try:
                    client.health_check()
                except Exception as e:
                    raise ConnectionError(
                        f"Bridge server is not reachable: {e}. "
                        f"Please start run_bridge_32bit.bat first."
                    )

            # 명령어에 따라 파이프라인 함수 호출
            from backend.main import (
                run_collect_insert,
                run_collect_update,
                run_collect_overseas,
                run_collect_macro,
                run_train_regime,
                run_backtest,
            )

            if command == "collect-insert":
                run_collect_insert(db, client)
            elif command == "collect-update":
                run_collect_update(db, client)
            elif command == "collect-overseas":
                run_collect_overseas(db, client)
            elif command == "collect-macro":
                run_collect_macro(db, client)
            elif command == "train-regime":
                run_train_regime(db)
            elif command == "backtest":
                run_backtest(db)

            self._status = PipelineStatus.COMPLETED

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._status = PipelineStatus.FAILED
            self._error = str(e)

        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

            # 완료/실패 메시지를 로그 큐에 push
            complete_msg = {
                "type": "complete",
                "status": self._status.value,
                "error": self._error,
            }
            try:
                self._loop.call_soon_threadsafe(
                    self._log_queue.put_nowait, complete_msg
                )
            except Exception:
                pass

            # 로깅 핸들러 제거
            if self._handler:
                logging.getLogger("backend").removeHandler(self._handler)
                self._handler = None
