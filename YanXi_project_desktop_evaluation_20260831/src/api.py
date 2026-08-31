"""
FastAPI REST API 服务
=====================
基于 Terry 项目的 api.py 设计搭建的统一 REST API。

使用方式:
    poetry run python src/api.py
    # 或
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import load_config
from src.orchestration.orchestrator import CallOrchestrator
from src.models import (
    NewCallRequest, TurnRequest, TurnResponse,
    SystemStatus,
)
from src.utils.logger import get_logger, configure_logging

logger = get_logger(__name__)

# 全局变量
orchestrator: Optional[CallOrchestrator] = None
sessions: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理"""
    global orchestrator
    config = load_config()
    configure_logging(config)
    logger.info("初始化 CallOrchestrator...")
    orchestrator = CallOrchestrator(config)
    logger.info("API 服务启动完成")
    yield
    logger.info("API 服务关闭")


app = FastAPI(
    title="言犀 AI 智能通话助手 API",
    description="基于 LangGraph 的多智能体通话处理系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 通话处理 API
# ============================================================

@app.post("/api/call/new", response_model=TurnResponse)
async def new_call(req: NewCallRequest):
    """处理新的来电"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    session_id = f"call_{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    result = orchestrator.run(req.call_text, caller_number=req.caller_number)
    timing = {"total": time.time() - t0}

    sessions[session_id] = {
        "result": result,
        "caller_number": req.caller_number,
    }

    return TurnResponse(
        session_id=session_id,
        type_id=result.get("type_id", "general"),
        call_type_name=result.get("call_type_name", "未知"),
        confidence=result.get("confidence", 0.0),
        classify_method=result.get("classify_method", ""),
        final_action=result.get("final_action", ""),
        agent_reply=result.get("agent_reply", ""),
        notification_card=result.get("notification_card"),
        timing=timing,
    )


@app.post("/api/call/turn", response_model=TurnResponse)
async def call_turn(req: TurnRequest):
    """多轮对话继续"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    t0 = time.time()
    result = orchestrator.resume_conversation(session["result"], req.call_text)
    timing = {"total": time.time() - t0}

    session["result"] = result

    return TurnResponse(
        session_id=req.session_id,
        type_id=result.get("type_id", "general"),
        call_type_name=result.get("call_type_name", "未知"),
        confidence=result.get("confidence", 0.0),
        classify_method=result.get("classify_method", ""),
        final_action=result.get("final_action", ""),
        agent_reply=result.get("agent_reply", ""),
        notification_card=result.get("notification_card"),
        timing=timing,
    )


# ============================================================
# 系统状态 API
# ============================================================

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """获取系统状态"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    return SystemStatus(
        version="1.0.0",
        llm_backend=orchestrator.llm_client.backend.value,
        available_backends=orchestrator.llm_client.available_backends,
        vector_available=orchestrator.hybrid_retriever.vector.is_available(),
        bm25_available=orchestrator.hybrid_retriever.bm25.is_available(),
        hybrid_available=orchestrator.hybrid_retriever.is_available(),
        presence_mode=orchestrator.presence_mode,
    )


@app.post("/api/presence")
async def set_presence(mode: str = Query(..., description="free/busy/dnd/driving"), reason: str = Query("")):
    """设置机主状态"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    orchestrator.set_presence(mode, reason)
    return {"status": "ok", "mode": mode}


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "1.0.0"}


def start():
    """启动 API 服务（入口函数）"""
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
