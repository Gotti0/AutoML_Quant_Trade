"""
Pipeline API Routes — 파이프라인 제어 전용 FastAPI 라우터

엔드포인트:
- POST /pipeline/run       — 파이프라인 실행
- GET  /pipeline/status     — 현재 상태 조회
- GET  /pipeline/logs/{id}  — SSE 실시간 로그 스트림
- GET  /pipeline/commands   — 사용 가능한 명령어 목록
"""
import asyncio
import json
import logging

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.api.pipeline_manager import PipelineManager, VALID_COMMANDS
from backend.data.database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _get_manager(request: Request) -> PipelineManager:
    """app.state에서 PipelineManager 인스턴스를 가져옴"""
    manager = getattr(request.app.state, "pipeline_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="PipelineManager not initialized")
    return manager


# 프론트엔드에서 보낼 로그 데이터 스펙
class ClientLogPayload(BaseModel):
    level: str
    message: str


@router.post("/logs")
def report_client_log(payload: ClientLogPayload):
    """프론트엔드 등 클라이언트에서 발생한 예외/상태를 시스템 DB에 기록"""
    db = DatabaseManager()
    db.insert_log(level=payload.level, source="frontend", message=payload.message)
    return {"status": "ok"}


@router.get("/logs")
def get_system_logs(limit: int = 200, source: str = None):
    """통합 시스템 로그 조회"""
    db = DatabaseManager()
    df = db.load_recent_logs(limit=limit, source=source)
    if df.empty:
        return {"logs": []}
    
    # DataFrame을 dict list로 변환 (timestamp 직렬화 위해 문자열 변환 적용)
    logs_list = df.to_dict(orient="records")
    return {"logs": logs_list}


@router.get("/commands")
def list_commands():
    """사용 가능한 파이프라인 명령어 목록"""
    return {
        "commands": [
            {"command": cmd, "description": desc}
            for cmd, desc in VALID_COMMANDS.items()
        ]
    }


@router.get("/status")
def get_status(request: Request):
    """현재 파이프라인 상태 조회"""
    manager = _get_manager(request)
    return manager.get_status_info()


@router.post("/run")
def run_pipeline(request: Request, command: str = Query(..., description="실행할 파이프라인 명령어")):
    """
    파이프라인 실행 트리거.

    즉시 202 Accepted 반환 후 백그라운드에서 실행.
    """
    manager = _get_manager(request)

    try:
        task_id = manager.start(command)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"taskId": task_id, "command": command, "status": "running"}


@router.get("/logs/{task_id}")
async def stream_logs(request: Request, task_id: str):
    """
    SSE 스트림으로 실시간 파이프라인 로그 전송.

    - 15초간 로그가 없으면 keep-alive ping 전송
    - 파이프라인 완료 시 `event: complete` 이벤트 전송 후 종료
    """
    manager = _get_manager(request)

    try:
        queue = manager.get_log_stream(task_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    async def event_generator():
        try:
            while True:
                # 클라이언트 연결 끊김 감지
                if await request.is_disconnected():
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # keep-alive ping — 연결 유지 (SSE 주석)
                    yield ": keep-alive\n\n"
                    continue

                if msg.get("type") == "complete":
                    yield f"event: complete\ndata: {json.dumps(msg)}\n\n"
                    break

                yield f"data: {json.dumps(msg)}\n\n"

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 프록시 대응
        },
    )
