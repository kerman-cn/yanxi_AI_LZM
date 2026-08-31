"""
来电类型定义与处理规则
======================
定义 16 种来电类型及对应的预设处理规则。
参考 YanXi-KCN 项目设计，结合我们的 free/busy/dnd/driving 状态系统。

每种来电类型有：
- 优先级：数字越大越优先
- 处理动作：reject(拒接)/forward(转接)/proxy(代接)/record(记录)/ask(询问)
- 默认回复模板：根据机主状态有不同的回复话术
"""

from dataclasses import dataclass
from enum import Enum


class CallAction(Enum):
    """处理动作"""
    REJECT = "reject"       # 拒接挂断，不通知机主
    FORWARD = "forward"     # 转接给机主
    PROXY = "proxy"         # AI 代接，与来电者对话
    RECORD = "record"       # 记录留言，通知机主
    ASK = "ask"             # 追问信息


@dataclass
class CallType:
    """来电类型定义"""
    type_id: str                    # 类型ID，如 "food_delivery"
    name: str                       # 中文名，如 "外卖配送"
    emoji: str                      # 图标
    priority: int                   # 优先级 (1-10)
    action: CallAction              # 默认处理动作
    # 不同机主状态下的回复模板
    reply_free: str = ""            # 空闲模式回复
    reply_busy: str = ""            # 忙碌模式回复
    reply_dnd: str = ""             # 免打扰回复
    reply_driving: str = ""         # 开车回复
    # 通知机主的模板
    notification: str = ""          # 给机主的通知模板


# ============================================================
# 16 种来电类型定义
# ============================================================

CALL_TYPES: dict[str, CallType] = {
    # --- 诈骗/风险类（最高优先级，直接拒接）---
    "scam": CallType(
        type_id="scam",
        name="诈骗电话",
        emoji="🚫",
        priority=10,
        action=CallAction.REJECT,
        notification="⚠️ 诈骗电话已被拦截：{call_text}",
    ),
    "scam_risk": CallType(
        type_id="scam_risk",
        name="疑似诈骗",
        emoji="⚠️",
        priority=9,
        action=CallAction.REJECT,
        notification="⚠️ 疑似诈骗来电已拦截：{call_text}",
    ),

    # --- 业务类（代接对话）---
    "food_delivery": CallType(
        type_id="food_delivery",
        name="外卖配送",
        emoji="🍜",
        priority=5,
        action=CallAction.PROXY,
        reply_free="您好，机主现在不方便亲自接，我是助手。请问是哪个平台的？送到哪里？",
        reply_busy="机主正在忙，我是助手。请问是哪个平台的？东西放门口就好。",
        reply_dnd="机主正在休息，请把东西放门口，谢谢。",
        reply_driving="机主正在开车，请把东西放门口或快递柜，谢谢。",
        notification="📦 外卖配送：{call_text}",
    ),
    "express": CallType(
        type_id="express",
        name="快递取件",
        emoji="📦",
        priority=5,
        action=CallAction.PROXY,
        reply_free="您好，我是机主的助手。请问是哪个快递公司的？放菜鸟驿站还是送上门？",
        reply_busy="机主正在忙，请放菜鸟驿站或快递柜，谢谢。",
        reply_dnd="机主正在休息，请放菜鸟驿站，谢谢。",
        reply_driving="机主正在开车，请放快递柜，谢谢。",
        notification="📦 快递：{call_text}",
    ),
    "taxi_arrived": CallType(
        type_id="taxi_arrived",
        name="打车到达",
        emoji="🚗",
        priority=5,
        action=CallAction.PROXY,
        reply_free="您好，机主马上出来，请稍等一下。",
        reply_busy="好的，机主正在忙，请稍等片刻。",
        reply_dnd="好的，机主马上出来，请稍等。",
        reply_driving="好的，请稍等。",
        notification="🚗 打车到达：{call_text}",
    ),

    # --- 推销/广告类（拒接）---
    "telemarketing": CallType(
        type_id="telemarketing",
        name="推销电话",
        emoji="📢",
        priority=8,
        action=CallAction.REJECT,
        notification="📢 推销来电已拦截：{call_text}",
    ),
    "game_promo": CallType(
        type_id="game_promo",
        name="游戏推广",
        emoji="🎮",
        priority=8,
        action=CallAction.REJECT,
        notification="🎮 游戏推广来电已拦截：{call_text}",
    ),

    # --- 重要联系人（需判断紧急程度）---
    "family": CallType(
        type_id="family",
        name="家人来电",
        emoji="👨‍👩‍👧",
        priority=3,
        action=CallAction.ASK,
        reply_free="您好，机主现在有空，请问有什么事吗？急事的话马上帮您转接。",
        reply_busy="机主正在忙，请问是急事吗？急事的话我帮您转接。",
        reply_dnd="机主正在休息，如果是急事我可以帮您转达。",
        reply_driving="机主正在开车，如果是急事请稍等，我帮您转接。",
        notification="👨‍👩‍👧 家人来电：{call_text}",
    ),
    "leader": CallType(
        type_id="leader",
        name="领导来电",
        emoji="👔",
        priority=2,
        action=CallAction.ASK,
        reply_free="您好，机主现在有空，请问有什么安排？我帮您转接。",
        reply_busy="机主正在忙，请问有急事吗？我帮您转达。",
        reply_dnd="机主正在休息，如有急事我可以转达。",
        reply_driving="机主正在开车，请问有什么安排？",
        notification="👔 领导来电：{call_text}",
    ),
    "friend": CallType(
        type_id="friend",
        name="熟人问候",
        emoji="🤝",
        priority=4,
        action=CallAction.ASK,
        reply_free="您好，机主有空。请问您是？有什么事吗？",
        reply_busy="机主正在忙，请问您是？急事的话帮您转达。",
        reply_dnd="机主正在休息，请问有什么急事吗？",
        reply_driving="机主正在开车，请问您是？",
        notification="🤝 朋友来电：{call_text}",
    ),
    "colleague": CallType(
        type_id="colleague",
        name="同事协作",
        emoji="💼",
        priority=4,
        action=CallAction.RECORD,
        reply_free="您好，机主现在有空，请问是什么事？",
        reply_busy="机主正在忙，不紧急的话请留言。",
        reply_dnd="机主正在休息，请留言，稍后回复。",
        reply_driving="机主正在开车，请留言。",
        notification="💼 同事来电：{call_text}",
    ),
    "client": CallType(
        type_id="client",
        name="客户来电",
        emoji="🤵",
        priority=3,
        action=CallAction.RECORD,
        reply_free="您好，机主现在有空，请问有什么事？我帮您转达。",
        reply_busy="机主正在忙，请问有紧急事项吗？",
        reply_dnd="机主正在休息，如有紧急事项请留言。",
        reply_driving="机主正在开车，请留言，稍后处理。",
        notification="🤵 客户来电：{call_text}",
    ),
    "bank": CallType(
        type_id="bank",
        name="银行来电",
        emoji="🏦",
        priority=4,
        action=CallAction.RECORD,
        reply_free="您好，机主有空，请问是什么业务？",
        reply_busy="机主正在忙，请问有紧急事项吗？",
        reply_dnd="机主正在休息，请留言。",
        reply_driving="机主正在开车，请留言。",
        notification="🏦 银行来电：{call_text}",
    ),
    "interview": CallType(
        type_id="interview",
        name="面试通知",
        emoji="📋",
        priority=2,
        action=CallAction.FORWARD,
        reply_free="您好！机主有空，马上帮您转接～",
        reply_busy="机主正在忙，但这是重要来电，我帮您转接。",
        reply_dnd="这是重要来电，我帮您转接机主。",
        reply_driving="机主正在开车，我帮您转接。",
        notification="📋 面试通知来电！已转接。内容：{call_text}",
    ),

    # --- 普通来电 ---
    "general": CallType(
        type_id="general",
        name="其他来电",
        emoji="📞",
        priority=6,
        action=CallAction.ASK,
        reply_free="您好，请问您是哪位？找机主有什么事吗？",
        reply_busy="机主正在忙，请问您是哪位？",
        reply_dnd="机主正在休息，请问您是哪位？有急事吗？",
        reply_driving="机主正在开车，请问您是哪位？",
        notification="📞 来电：{call_text}",
    ),
    "meaningless": CallType(
        type_id="meaningless",
        name="无意义输入",
        emoji="❓",
        priority=1,
        action=CallAction.ASK,
        reply_free="不好意思没听清，能再说一遍吗？",
        reply_busy="不好意思没听清，能再说一遍吗？",
        reply_dnd="不好意思没听清，能再说一遍吗？",
        reply_driving="不好意思没听清，能再说一遍吗？",
        notification="",
    ),
}


def get_call_type(type_id: str) -> CallType:
    """获取来电类型定义，不存在返回 general"""
    return CALL_TYPES.get(type_id, CALL_TYPES["general"])


def get_reply(call_type: CallType, presence_mode: str, default: str = "") -> str:
    """根据机主状态获取对应回复模板"""
    reply_map = {
        "free": call_type.reply_free,
        "busy": call_type.reply_busy,
        "dnd": call_type.reply_dnd,
        "driving": call_type.reply_driving,
    }
    return reply_map.get(presence_mode, default or call_type.reply_free)


def get_action(call_type: CallType, presence_mode: str) -> str:
    """
    根据来电类型和机主状态，返回最终处理动作。
    空闲模式下更多来电会转接；免打扰模式下几乎不转接。
    """
    base_action = call_type.action

    # 拒绝类始终拒绝
    if base_action == CallAction.REJECT:
        return "reject"

    # 空闲模式：更多来电转接
    if presence_mode == "free":
        if call_type.type_id in ("family", "leader", "friend", "interview"):
            return "forward"
        if base_action == CallAction.ASK:
            return "continue_conversation"  # 先问清楚
        return base_action.value

    # 忙碌模式
    if presence_mode == "busy":
        if base_action == CallAction.FORWARD:
            return "forward"
        if base_action == CallAction.ASK:
            return "continue_conversation"
        return base_action.value

    # 免打扰/开车：几乎不转接
    if presence_mode in ("dnd", "driving"):
        if call_type.type_id == "interview":
            return "forward"
        if base_action == CallAction.ASK:
            return "general_reply"
        return base_action.value

    return base_action.value
