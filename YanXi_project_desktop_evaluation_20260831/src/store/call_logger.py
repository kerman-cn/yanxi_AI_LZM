"""
通话记录持久化模块
==================
记录每次来电的完整处理流程，包括：
  - 来电时间、来电者信息
  - 分类结果、处理动作
  - 对话历史
  - 生成的通知卡片

数据存储在本地 JSON 文件中，按日期归档。

使用方式:
    from src.store.call_logger import CallLogger

    logger = CallLogger()
    session_id = logger.start_session("13800138000", "我是美团外卖的")
    logger.log_classification(session_id, "food_delivery", 0.95, "keyword")
    logger.log_action(session_id, "proxy", "好的，请问放哪里？")
    logger.end_session(session_id, {"card_id": "card_abc123"})
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CallSession:
    """单次通话会话记录"""
    session_id: str = ""
    # 来电信息
    caller_number: str = ""
    caller_text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    # 分类结果
    type_id: str = ""
    call_type_name: str = ""
    confidence: float = 0.0
    classify_method: str = ""
    # 处理结果
    final_action: str = ""
    agent_reply: str = ""
    # 对话历史
    conversation_history: list[dict] = field(default_factory=list)
    # 通知卡片
    notification_card: Optional[dict] = None
    # 机主当时状态
    presence_mode: str = ""
    presence_reason: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"call_{uuid.uuid4().hex[:8]}"
        if not self.start_time:
            self.start_time = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration_sec(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0


class CallLogger:
    """通话记录管理器。"""

    def __init__(self, persist_dir: str = "./data/call_logs/"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: dict[str, CallSession] = {}

    def start_session(
        self,
        caller_number: str = "",
        caller_text: str = "",
        presence_mode: str = "",
        presence_reason: str = "",
    ) -> str:
        """开始记录一次通话，返回 session_id。"""
        session = CallSession(
            caller_number=caller_number,
            caller_text=caller_text,
            presence_mode=presence_mode,
            presence_reason=presence_reason,
        )
        self._active_sessions[session.session_id] = session
        logger.debug(f"通话会话开始: {session.session_id}")
        return session.session_id

    def log_classification(
        self,
        session_id: str,
        type_id: str,
        call_type_name: str,
        confidence: float,
        method: str,
    ):
        """记录分类结果。"""
        session = self._active_sessions.get(session_id)
        if session:
            session.type_id = type_id
            session.call_type_name = call_type_name
            session.confidence = confidence
            session.classify_method = method

    def log_action(self, session_id: str, action: str, reply: str = ""):
        """记录处理动作。"""
        session = self._active_sessions.get(session_id)
        if session:
            session.final_action = action
            if reply:
                session.agent_reply = reply

    def log_conversation(self, session_id: str, role: str, content: str):
        """记录对话轮次。"""
        session = self._active_sessions.get(session_id)
        if session:
            session.conversation_history.append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })

    def end_session(self, session_id: str, notification_card: Optional[dict] = None):
        """结束通话会话并持久化。"""
        session = self._active_sessions.pop(session_id, None)
        if not session:
            logger.warning(f"未找到会话: {session_id}")
            return

        session.end_time = time.time()
        if notification_card:
            session.notification_card = notification_card

        # 按日期归档
        date_str = time.strftime("%Y-%m-%d", time.localtime(session.start_time))
        filepath = self.persist_dir / f"{date_str}.jsonl"

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            f"通话记录已保存: {session.session_id} "
            f"类型={session.call_type_name} "
            f"动作={session.final_action} "
            f"时长={session.duration_sec:.1f}s"
        )

    def get_today_stats(self) -> dict:
        """获取今日通话统计。"""
        date_str = time.strftime("%Y-%m-%d")
        filepath = self.persist_dir / f"{date_str}.jsonl"

        stats = {
            "total": 0,
            "by_action": {},
            "by_type": {},
        }

        if not filepath.exists():
            return stats

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    stats["total"] += 1
                    action = record.get("final_action", "unknown")
                    stats["by_action"][action] = stats["by_action"].get(action, 0) + 1
                    type_name = record.get("call_type_name", "unknown")
                    stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return stats
