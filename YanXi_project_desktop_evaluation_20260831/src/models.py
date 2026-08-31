"""
Pydantic 数据模型
=================
所有 API 请求/响应结构定义。
统一来自 Terry 项目的 models.py 设计。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# 通话处理
# ============================================================

class NewCallRequest(BaseModel):
    """新来电请求"""
    caller_number: str = Field(default="", description="来电号码")
    call_text: str = Field(..., description="STT 识别的来电文本")


class TurnRequest(BaseModel):
    """多轮对话请求"""
    session_id: str = Field(..., description="会话 ID")
    call_text: str = Field(..., description="本轮来电文本")
    caller_number: str = Field(default="", description="来电号码")


class TurnResponse(BaseModel):
    """多轮对话响应"""
    session_id: str
    type_id: str
    call_type_name: str
    confidence: float
    classify_method: str
    final_action: str
    agent_reply: str = ""
    notification_card: Optional[Dict[str, Any]] = None
    timing: Dict[str, float] = Field(default_factory=dict)


class CallSummary(BaseModel):
    """通话摘要"""
    session_id: str
    caller_number: str
    call_type: str
    final_action: str
    timestamp: str
    duration_ms: float


# ============================================================
# 用户画像
# ============================================================

class UserProfileUpdate(BaseModel):
    """用户画像更新"""
    phone_number: str = Field(..., description="电话号码")
    contact_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    is_blacklisted: bool = False
    is_whitelisted: bool = False


class UserProfileResponse(BaseModel):
    """用户画像响应"""
    phone_number: str
    contact_name: Optional[str] = None
    tags: List[str] = []
    call_count: int = 0
    trust_score: float = 0.5
    is_blacklisted: bool = False
    is_whitelisted: bool = False
    notes: Optional[str] = None


# ============================================================
# 系统状态
# ============================================================

class SystemStatus(BaseModel):
    """系统状态"""
    version: str = "1.0.0"
    llm_backend: str
    available_backends: List[str]
    vector_available: bool
    bm25_available: bool = False
    hybrid_available: bool = False
    presence_mode: str
