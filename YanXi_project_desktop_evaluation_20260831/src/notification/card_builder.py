"""
通知与卡片生成模块
==================
当 AI 代接来电后，生成结构化的通知卡片，供手机锁屏页面展示。

卡片类型:
  - delivery_card:   外卖/快递送达卡片
  - message_card:    留言记录卡片
  - urgent_alert:    紧急来电提醒
  - scam_log:        诈骗拦截记录

使用方式:
    from src.notification.card_builder import CardBuilder

    builder = CardBuilder()
    card = builder.build_delivery_card(
        call_type="外卖",
        company="美团外卖",
        item="黄焖鸡米饭",
        location="3号楼楼下",
        notes="放门口"
    )
    print(card.to_notification())
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CardType(Enum):
    """卡片类型"""
    DELIVERY = "delivery_card"
    MESSAGE = "message_card"
    URGENT_ALERT = "urgent_alert"
    SCAM_LOG = "scam_log"


@dataclass
class NotificationCard:
    """通知卡片数据结构"""
    card_id: str = ""
    card_type: str = ""
    title: str = ""
    subtitle: str = ""
    body: str = ""
    icon: str = ""
    priority: str = "normal"  # low / normal / high / urgent
    timestamp: float = 0.0
    # 结构化数据
    data: dict = field(default_factory=dict)
    # 是否已读
    read: bool = False

    def __post_init__(self):
        if not self.card_id:
            self.card_id = f"card_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = time.time()

    def to_notification(self) -> dict:
        """转换为手机通知格式（供前端消费）。"""
        return {
            "id": self.card_id,
            "type": self.card_type,
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "icon": self.icon,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return asdict(self)


class CardBuilder:
    """卡片构建器，提供各类卡片的便捷构建方法。"""

    def build_delivery_card(
        self,
        call_type: str,
        company: str = "",
        item: str = "",
        location: str = "",
        notes: str = "",
        contact_person: str = "",
    ) -> NotificationCard:
        """
        构建外卖/快递送达卡片。

        参数:
            call_type: "外卖" / "快递" / "打车" 等
            company: 平台名称
            item: 物品描述
            location: 送达地点
            notes: 备注
            contact_person: 联系人
        """
        icon_map = {
            "外卖": "🍕",
            "快递": "📦",
            "打车": "🚗",
            "家政": "🏠",
        }
        icon = icon_map.get(call_type, "📋")

        title = f"{icon} {call_type}已送达"
        subtitle = company or call_type
        body_parts = []
        if item:
            body_parts.append(f"物品: {item}")
        if location:
            body_parts.append(f"地点: {location}")
        if notes:
            body_parts.append(f"备注: {notes}")
        if contact_person:
            body_parts.append(f"联系人: {contact_person}")
        body = "\n".join(body_parts)

        return NotificationCard(
            card_type=CardType.DELIVERY.value,
            title=title,
            subtitle=subtitle,
            body=body,
            icon=icon,
            priority="normal",
            data={
                "call_type": call_type,
                "company": company,
                "item_description": item,
                "location": location,
                "additional_notes": notes,
                "contact_person": contact_person,
            },
        )

    def build_message_card(
        self,
        caller: str = "",
        message: str = "",
        relationship: str = "",
    ) -> NotificationCard:
        """构建留言记录卡片。"""
        title = f"💬 留言来自 {caller or '未知'}"
        subtitle = relationship or "来电留言"
        body = message or "对方未留言"

        return NotificationCard(
            card_type=CardType.MESSAGE.value,
            title=title,
            subtitle=subtitle,
            body=body,
            icon="💬",
            priority="normal",
            data={
                "caller": caller,
                "message": message,
                "relationship": relationship,
            },
        )

    def build_urgent_alert(
        self,
        caller: str = "",
        reason: str = "",
        urgency_level: str = "high",
        relationship: str = "",
    ) -> NotificationCard:
        """构建紧急来电提醒卡片。"""
        title = f"🚨 紧急来电: {caller or '未知'}"
        subtitle = f"关系: {relationship or '未知'} | 紧急度: {urgency_level}"
        body = reason or "对方有紧急事项需要联系您"

        priority_map = {
            "low": "normal",
            "medium": "high",
            "high": "urgent",
            "critical": "urgent",
        }

        return NotificationCard(
            card_type=CardType.URGENT_ALERT.value,
            title=title,
            subtitle=subtitle,
            body=body,
            icon="🚨",
            priority=priority_map.get(urgency_level, "high"),
            data={
                "caller": caller,
                "reason": reason,
                "urgency_level": urgency_level,
                "relationship": relationship,
            },
        )

    def build_scam_log(
        self,
        scam_type: str = "",
        reason: str = "",
        confidence: float = 0.0,
    ) -> NotificationCard:
        """构建诈骗拦截记录卡片。"""
        title = "🛡️ 诈骗拦截"
        subtitle = scam_type or "未知诈骗类型"
        body = f"置信度: {confidence:.0%}\n原因: {reason or '未记录'}"

        return NotificationCard(
            card_type=CardType.SCAM_LOG.value,
            title=title,
            subtitle=subtitle,
            body=body,
            icon="🛡️",
            priority="low",
            data={
                "scam_type": scam_type,
                "reason": reason,
                "confidence": confidence,
            },
        )


class NotificationStore:
    """通知持久化存储，将卡片保存到本地文件。"""

    def __init__(self, persist_path: str = "./data/notifications/"):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def save(self, card: NotificationCard) -> str:
        """保存卡片，返回文件路径。"""
        date_str = time.strftime("%Y-%m-%d", time.localtime(card.timestamp))
        filename = f"{date_str}_{card.card_id}.json"
        filepath = self.persist_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)

        logger.debug(f"通知卡片已保存: {filepath}")
        return str(filepath)

    def load_recent(self, count: int = 20) -> list[NotificationCard]:
        """加载最近的通知卡片。"""
        cards = []
        files = sorted(self.persist_path.glob("*.json"), reverse=True)

        for fp in files[:count]:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cards.append(NotificationCard(**data))
            except Exception as e:
                logger.warning(f"加载通知卡片失败 {fp}: {e}")

        return cards
