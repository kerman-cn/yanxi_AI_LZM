"""
编排状态定义
============
LangGraph StateGraph 的全局状态 TypedDict。
"""

from typing import TypedDict, Optional, Dict


class OrchestratorState(TypedDict, total=False):
    """LangGraph 全局状态"""

    # 输入
    call_text: str
    caller_number: str

    # 分类结果
    type_id: str
    call_type_name: str
    confidence: float
    classify_method: str

    # 机主状态
    presence_mode: str
    presence_reason: str

    # 来电者画像
    caller_profile: Optional[dict]

    # 检索上下文
    retrieval_context: str

    # 动作
    final_action: str
    final_message: str
    agent_reply: str

    # 通知卡片
    notification_card: Optional[dict]

    # 对话记忆
    conversation_memory: Optional[dict]

    # 元信息
    timing: Dict[str, float]
