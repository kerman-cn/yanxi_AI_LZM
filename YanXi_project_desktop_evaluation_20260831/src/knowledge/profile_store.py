"""
来电者画像系统 (CallerProfile)
================================
记录每个来电号码的历史信息，实现：
  - 号码识别：已知号码直接走历史策略
  - 信任评分：根据历史通话自动计算信任度
  - 标签管理：自动/手动为号码打标签
  - 黑名单：低信任号码自动拦截

数据结构:
  CallerProfile:
    - phone_number:  电话号码
    - call_count:    来电次数
    - last_call_time: 上次来电时间
    - call_type_history: 历史分类列表
    - trust_score:   信任分数 (0-1)
    - tags:          标签列表
    - notes:         备注
    - is_blacklisted: 是否黑名单

使用方式:
    from src.knowledge.profile_store import CallerProfileStore

    store = CallerProfileStore()
    profile = store.get_or_create("13800138000")
    profile.add_call("food_delivery", 0.95)
    store.update(profile)

    # 查询
    known = store.lookup("13800138000")  # 返回 None 或 CallerProfile
    blacklisted = store.get_blacklisted()
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CallerProfile:
    """来电者画像"""
    phone_number: str = ""
    # 来电统计
    call_count: int = 0
    last_call_time: float = 0.0
    first_call_time: float = 0.0
    # 历史分类
    call_type_history: list[str] = field(default_factory=list)
    # 信任评分 (0=完全不可信, 1=完全可信)
    trust_score: float = 0.5
    # 标签
    tags: list[str] = field(default_factory=list)
    # 备注
    notes: str = ""
    # 黑名单
    is_blacklisted: bool = False
    # 白名单（家人/朋友/重要联系人）
    is_whitelisted: bool = False
    # 联系人名称（机主可手动设置）
    contact_name: str = ""

    def add_call(self, call_type: str, confidence: float = 1.0) -> None:
        """
        记录一次来电。

        参数:
            call_type: 来电分类 (如 "food_delivery", "scam")
            confidence: 分类置信度
        """
        self.call_count += 1
        now = time.time()
        self.last_call_time = now
        if self.first_call_time == 0:
            self.first_call_time = now
        self.call_type_history.append(call_type)

        # 自动更新信任分数
        self._update_trust_score(call_type, confidence)

        # 自动打标签
        self._auto_tag(call_type)

    def _update_trust_score(self, call_type: str, confidence: float) -> None:
        """
        根据来电类型更新信任分数。

        规则:
        - 诈骗来电: 大幅降低信任
        - 业务来电: 轻微降低（可能是骚扰）
        - 正常来电: 轻微提升
        - 紧急来电: 提升
        """
        delta = 0.0
        if call_type == "scam":
            delta = -0.3 * confidence
        elif call_type in ("telemarketing", "promotion"):
            delta = -0.1 * confidence
        elif call_type in ("food_delivery", "express_delivery"):
            delta = -0.02  # 业务电话轻微降低
        elif call_type == "urgent":
            delta = 0.1
        elif call_type == "normal":
            delta = 0.05
        else:
            delta = -0.01  # 未知类型轻微降低

        self.trust_score = max(0.0, min(1.0, self.trust_score + delta))

        # 信任极低自动拉黑
        if self.trust_score < 0.15 and self.call_count >= 2:
            self.is_blacklisted = True

    def _auto_tag(self, call_type: str) -> None:
        """根据来电类型自动打标签。"""
        tag_map = {
            "scam": "疑似诈骗",
            "telemarketing": "推销",
            "promotion": "推广",
            "food_delivery": "外卖",
            "express_delivery": "快递",
            "urgent": "紧急联系人",
            "normal": "普通来电",
        }
        tag = tag_map.get(call_type)
        if tag and tag not in self.tags:
            self.tags.append(tag)

        # 根据来电次数打标签
        if self.call_count >= 5 and "频繁来电" not in self.tags:
            self.tags.append("频繁来电")
        if self.call_count >= 10 and "高频来电" not in self.tags:
            self.tags.append("高频来电")

    def get_dominant_type(self) -> str:
        """获取最频繁的来电类型。"""
        if not self.call_type_history:
            return "unknown"
        from collections import Counter
        counter = Counter(self.call_type_history)
        return counter.most_common(1)[0][0]

    def should_auto_reject(self) -> bool:
        """是否应该自动拒接。"""
        return self.is_blacklisted

    def should_auto_whitelist(self) -> bool:
        """是否应该自动转接（白名单）。"""
        return self.is_whitelisted or self.trust_score >= 0.8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CallerProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CallerProfileStore:
    """来电者画像存储"""

    def __init__(self, persist_path: str = "./data/caller_profiles/profiles.json"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, CallerProfile] = self._load()

    def get_or_create(self, phone_number: str) -> CallerProfile:
        """
        获取或创建来电者画像。

        参数:
            phone_number: 电话号码

        返回:
            CallerProfile: 来电者画像
        """
        if phone_number not in self._profiles:
            self._profiles[phone_number] = CallerProfile(phone_number=phone_number)
            logger.debug(f"新建画像: {phone_number}")
        return self._profiles[phone_number]

    def lookup(self, phone_number: str) -> Optional[CallerProfile]:
        """
        查询来电者画像（不存在返回 None）。

        参数:
            phone_number: 电话号码

        返回:
            CallerProfile 或 None
        """
        return self._profiles.get(phone_number)

    def update(self, profile: CallerProfile) -> None:
        """
        更新来电者画像并持久化。

        参数:
            profile: 更新后的画像
        """
        self._profiles[profile.phone_number] = profile
        self._save()

    def get_blacklisted(self) -> list[CallerProfile]:
        """获取所有黑名单号码。"""
        return [p for p in self._profiles.values() if p.is_blacklisted]

    def get_whitelisted(self) -> list[CallerProfile]:
        """获取所有白名单号码。"""
        return [p for p in self._profiles.values() if p.is_whitelisted]

    def search_by_tag(self, tag: str) -> list[CallerProfile]:
        """按标签搜索。"""
        return [p for p in self._profiles.values() if tag in p.tags]

    def set_blacklist(self, phone_number: str, blacklisted: bool = True) -> None:
        """设置/取消黑名单。"""
        profile = self.get_or_create(phone_number)
        profile.is_blacklisted = blacklisted
        if blacklisted:
            profile.trust_score = 0.0
            if "黑名单" not in profile.tags:
                profile.tags.append("黑名单")
        else:
            profile.trust_score = 0.3
            profile.tags = [t for t in profile.tags if t != "黑名单"]
        self.update(profile)

    def set_whitelist(self, phone_number: str, name: str = "", whitelisted: bool = True) -> None:
        """设置/取消白名单。"""
        profile = self.get_or_create(phone_number)
        profile.is_whitelisted = whitelisted
        if name:
            profile.contact_name = name
        if whitelisted:
            profile.trust_score = 1.0
            if "白名单" not in profile.tags:
                profile.tags.append("白名单")
        else:
            profile.trust_score = 0.5
            profile.tags = [t for t in profile.tags if t != "白名单"]
        self.update(profile)

    def get_stats(self) -> dict:
        """获取画像统计。"""
        total = len(self._profiles)
        return {
            "total_profiles": total,
            "blacklisted": sum(1 for p in self._profiles.values() if p.is_blacklisted),
            "whitelisted": sum(1 for p in self._profiles.values() if p.is_whitelisted),
            "avg_trust": sum(p.trust_score for p in self._profiles.values()) / max(total, 1),
        }

    def _load(self) -> dict[str, CallerProfile]:
        """从文件加载画像数据。"""
        if not self.persist_path.exists():
            return {}
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                num: CallerProfile.from_dict(d)
                for num, d in data.items()
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"加载画像数据失败: {e}")
            return {}

    def _save(self) -> None:
        """持久化画像数据。"""
        data = {
            num: profile.to_dict()
            for num, profile in self._profiles.items()
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
