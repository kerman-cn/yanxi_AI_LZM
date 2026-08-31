"""
多轮对话记忆增强 (ConversationMemory)
======================================
管理通话中的对话上下文，实现：
  - 短期记忆：当前通话的对话历史
  - 工作记忆：当前提取的关键信息（谁、什么事、放哪里）
  - 长期记忆：跨通话的长期记忆（来自 CallerProfile）
  - 上下文窗口：自动裁剪历史，控制 token 消耗

核心改进：
  - 避免重复询问已确认的信息
  - 根据已提取信息决定下一步问什么
  - 对话策略自动调整

使用方式:
    from src.agents.conversation_memory import ConversationMemory

    memory = ConversationMemory()
    memory.add_user_message("我是美团外卖的")
    memory.add_assistant_message("好的，请问放哪里？")
    memory.add_user_message("放3号楼门口")

    # 提取关键信息
    memory.update_working_memory({
        "caller_identity": "美团外卖员",
        "delivery_location": "3号楼门口",
    })

    # 构建上下文
    context = memory.get_context_window(window=5)
    info = memory.get_info_summary()
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class WorkingMemory:
    """
    工作记忆：当前通话中提取的关键信息。

    这些信息会在对话过程中逐步填充，
    Agent 可以根据已有信息决定下一步策略。
    """
    # 来电者身份
    caller_identity: str = ""        # "美团外卖员" / "顺丰快递" / "妈妈"
    caller_name: str = ""            # 具体姓名（如果对方说了）
    caller_company: str = ""         # 所属公司/平台

    # 来电目的
    caller_purpose: str = ""         # "送外卖" / "送快递" / "通知开会"
    purpose_detail: str = ""         # "黄焖鸡米饭" / "一个包裹"

    # 配送信息
    delivery_location: str = ""      # "3号楼门口" / "菜鸟驿站"
    delivery_notes: str = ""         # "放门口就行" / "到付"

    # 紧急程度
    urgency_level: str = ""          # "high" / "medium" / "low"
    urgency_reason: str = ""         # "家人生病" / "会议通知"

    # 其他信息
    extra_info: dict = field(default_factory=dict)

    # 信息完整度评分 (0-1)
    completeness: float = 0.0

    def update(self, data: dict) -> None:
        """批量更新工作记忆（只更新非空字段）。"""
        for key, value in data.items():
            if value and hasattr(self, key):
                setattr(self, key, value)
            elif key == "extra_info" and isinstance(value, dict):
                self.extra_info.update(value)
        self._recalculate_completeness()

    def _recalculate_completeness(self) -> None:
        """重新计算信息完整度。"""
        fields = [
            self.caller_identity,
            self.caller_purpose,
        ]
        filled = sum(1 for f in fields if f)
        self.completeness = filled / max(len(fields), 1)

    def is_complete_for_delivery(self) -> bool:
        """配送场景信息是否完整。"""
        return bool(self.caller_identity and self.delivery_location)

    def is_complete_for_urgent(self) -> bool:
        """紧急场景信息是否完整。"""
        return bool(self.caller_identity and self.urgency_reason)

    def get_missing_fields(self) -> list[str]:
        """获取缺失的关键信息字段。"""
        missing = []
        if not self.caller_identity:
            missing.append("caller_identity")
        if not self.caller_purpose:
            missing.append("caller_purpose")
        return missing

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """生成信息摘要文本。"""
        parts = []
        if self.caller_identity:
            parts.append(f"身份: {self.caller_identity}")
        if self.caller_purpose:
            parts.append(f"目的: {self.caller_purpose}")
        if self.delivery_location:
            parts.append(f"地点: {self.delivery_location}")
        if self.urgency_level:
            parts.append(f"紧急: {self.urgency_level}")
        return " | ".join(parts) if parts else "暂无信息"


class ConversationMemory:
    """
    多轮对话记忆管理器。

    三层记忆架构:
    ┌─────────────────────────────────────┐
    │  短期记忆 (short_term)              │  ← 当前通话对话历史
    │  最近 N 轮对话                      │
    ├─────────────────────────────────────┤
    │  工作记忆 (working_memory)          │  ← 当前提取的关键信息
    │  谁、什么事、放哪里、紧急程度        │
    ├─────────────────────────────────────┤
    │  长期记忆 (long_term)               │  ← 跨通话记忆
    │  来自 CallerProfile 的历史信息      │
    └─────────────────────────────────────┘
    """

    def __init__(self, max_short_term: int = 20):
        """
        参数:
            max_short_term: 短期记忆最大轮数
        """
        self.short_term: list[dict] = []
        self.working_memory = WorkingMemory()
        self.long_term: list[dict] = []
        self.max_short_term = max_short_term

        # 对话元信息
        self._turn_count: int = 0
        self._start_time: float = time.time()

    def add_user_message(self, text: str) -> None:
        """记录来电者消息。"""
        self.short_term.append({
            "role": "user",
            "content": text,
            "turn": self._turn_count,
            "timestamp": time.time(),
        })
        self._turn_count += 1
        self._trim_short_term()

    def add_assistant_message(self, text: str) -> None:
        """记录助手消息。"""
        self.short_term.append({
            "role": "assistant",
            "content": text,
            "turn": self._turn_count,
            "timestamp": time.time(),
        })
        self._trim_short_term()

    def update_working_memory(self, data: dict) -> None:
        """更新工作记忆。"""
        self.working_memory.update(data)
        logger.debug(f"工作记忆更新: {self.working_memory.summary()}")

    def set_long_term(self, caller_profile_data: dict) -> None:
        """
        设置长期记忆（来自 CallerProfile）。

        参数:
            caller_profile_data: 来电者画像数据
        """
        self.long_term = []
        if caller_profile_data:
            # 提取关键信息作为长期记忆
            if caller_profile_data.get("contact_name"):
                self.long_term.append({
                    "role": "system",
                    "content": f"该来电者已知姓名: {caller_profile_data['contact_name']}",
                })
            if caller_profile_data.get("tags"):
                tags = ", ".join(caller_profile_data["tags"])
                self.long_term.append({
                    "role": "system",
                    "content": f"该号码历史标签: {tags}",
                })
            if caller_profile_data.get("call_count", 0) > 0:
                dominant = caller_profile_data.get("dominant_type", "")
                self.long_term.append({
                    "role": "system",
                    "content": (
                        f"该号码已来电 {caller_profile_data['call_count']} 次，"
                        f"最常见类型: {dominant}"
                    ),
                })
            if caller_profile_data.get("is_whitelisted"):
                self.long_term.append({
                    "role": "system",
                    "content": "该号码在白名单中，是机主的重要联系人。",
                })
            if caller_profile_data.get("is_blacklisted"):
                self.long_term.append({
                    "role": "system",
                    "content": "该号码在黑名单中，疑似骚扰/诈骗。",
                })

    def get_context_window(self, window: int = 5) -> list[dict]:
        """
        获取上下文窗口（用于 LLM prompt）。

        结构:
          [长期记忆] + [工作记忆摘要] + [最近 N 轮对话]

        参数:
            window: 最近对话轮数

        返回:
            list[dict]: 消息列表
        """
        context = []

        # 1. 长期记忆
        context.extend(self.long_term)

        # 2. 工作记忆摘要
        if self.working_memory.summary() != "暂无信息":
            context.append({
                "role": "system",
                "content": f"当前已提取信息: {self.working_memory.summary()}",
            })

        # 3. 短期记忆（最近 N 轮）
        recent = self.short_term[-window * 2:]  # 每轮 = user + assistant
        for msg in recent:
            context.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        return context

    def get_info_summary(self) -> str:
        """获取当前工作记忆摘要。"""
        return self.working_memory.summary()

    def get_next_question_hint(self) -> Optional[str]:
        """
        根据工作记忆的缺失字段，建议下一个问题。

        返回:
            str: 建议的问题，如果信息完整则返回 None
        """
        wm = self.working_memory

        if not wm.caller_identity:
            return "请问您是哪位？/您代表哪个平台？"
        if not wm.caller_purpose:
            return "请问有什么事？"
        if wm.caller_purpose in ("送外卖", "送快递", "配送") and not wm.delivery_location:
            return "请问放哪里？"
        if wm.urgency_level == "high" and not wm.urgency_reason:
            return "请问是什么紧急情况？"

        return None  # 信息已完整

    def get_dialogue_summary(self) -> str:
        """生成对话摘要（用于通话记录）。"""
        lines = []
        for msg in self.short_term:
            role = "来电者" if msg["role"] == "user" else "言犀"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def duration_sec(self) -> float:
        return time.time() - self._start_time

    def _trim_short_term(self) -> None:
        """裁剪短期记忆，保留最近 max_short_term 条。"""
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]

    def reset(self) -> None:
        """重置所有记忆（新通话开始时调用）。"""
        self.short_term = []
        self.working_memory = WorkingMemory()
        self.long_term = []
        self._turn_count = 0
        self._start_time = time.time()

    def to_dict(self) -> dict:
        """序列化为字典（用于通话记录）。"""
        return {
            "short_term": self.short_term,
            "working_memory": self.working_memory.to_dict(),
            "turn_count": self._turn_count,
            "duration_sec": round(self.duration_sec, 1),
        }
