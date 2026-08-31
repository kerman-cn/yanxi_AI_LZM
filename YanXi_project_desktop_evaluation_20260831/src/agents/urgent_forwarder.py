"""
紧急来电转接 Agent
------------------
识别亲友、同事等重要联系人的紧急来电，在确认紧急程度后，
触发呼叫机主（人工转接）的流程。

判定逻辑:
1. 判断来电者身份：亲友？同事？陌生人？
2. 判断紧急程度：是否涉及人身安全、重大事件、紧急事务？
3. 决定是否转接：重要+紧急 → 转接；重要+不紧急 → 留言；陌生人 → 礼貌拒绝

转接条件（需同时满足）：
- 来电者与机主有明确的关系（亲友/同事/家人）
- 事情具有紧急性（事故、疾病、紧急事务等）
- 🌟 .

使用方式:
    from src.agents.urgent_forwarder import UrgentForwarder

    forwarder = UrgentForwarder(config)
    result = forwarder.assess(call_text)
    if result["should_forward"]:
        print(f"🔔 紧急来电! {result['reason']}")
        print(f"   来自: {result['caller_identity']}")
"""

import json

from openai import OpenAI

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

URGENT_FORWARDER_SYSTEM_PROMPT = """你是一个AI来电筛选助手，负责判断来电是否重要紧急到需要立即转接给机主。

## ⚠️ 核心原则：减少误打扰
**只有在来电确实重要且紧急时才转接。** 机主可能正在开会、开车或休息，不必要的打扰会影响机主。
默认策略是"不转接，代处理"，只有满足严格条件时才转接。

## 转接条件（需同时满足至少两条）
1. **身份明确**: 来电者与机主有明确关系（直系亲属、配偶、老板、同事、密友），而非陌生人
2. **事项紧急**: 涉及人身安全、重大事故、紧急决定、时间敏感事务
3. **必须本人**: 事情必须机主本人处理，无法由助手代劳

## 不转接的情况
- 外卖、快递、普通推销 → 由业务助手代处理
- 朋友闲聊、普通问候 → 告知稍后回电
- 陌生人来电、无法确认身份 → 礼貌拒绝
- 疑似诈骗 → 已由前置检测拒绝

## 需要追问的情况
- 来电者只说了"急事"但没有说具体什么事 → 追问细节
- 来电者自称"朋友"但名字机主不认识 → 追问确认身份

## 输出格式
请严格按照以下 JSON 格式输出：
```json
{
    "should_forward": true或false,
    "urgency_level": "high高 / medium中 / low低",
    "caller_identity": "来电者身份描述，如'自称是机主的母亲'",
    "relationship": "来电者与机主的关系，如'母女'、'同事'、'陌生人'",
    "emergency_type": "紧急类型: medical医疗 / accident事故 / family家庭急事 / work工作紧急 / none不紧急",
    "reason": "判定理由，50字以内",
    "agent_action": "forward转接 / ask_more追问 / reject拒绝 / take_message留言",
    "agent_text": "助手要对来电者说的话",
    "notification": "如果要转接，给机主的通知文本；否则为空"
}
```"""

# 紧急关键词列表（兜底策略，在 LLM 不可用时使用）
URGENT_KEYWORDS = [
    "医院", "急救", "车祸", "出事", "不行了", "危险",
    "马上", "赶紧", "快", "紧急", "救命", "死了",
    "住院", "手术", "重症", "ICU", "警察", "火灾",
]

IDENTITY_KEYWORDS = [
    "妈妈", "爸爸", "妈", "爸", "老婆", "老公", "儿子", "女儿",
    "哥哥", "姐姐", "弟弟", "妹妹", "奶奶", "爷爷",
    "老板", "领导", "经理", "同事",
]


class UrgentForwarder:
    """
    紧急来电转接 Agent。

    对非诈骗来电进行重要性和紧急性评估，判断是否需要转接给机主。
    """

    def __init__(self, config: dict):
        """
        初始化紧急转接 Agent。

        参数:
            config: 全局配置字典
        """
        llm_cfg = config.get("llm", {})

        self.llm = OpenAI(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        )
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = llm_cfg.get("temperature", 0.3)  # 低温度，保守判断
        self.max_tokens = llm_cfg.get("max_tokens", 2048)

        logger.info(f"紧急转接 Agent 已初始化 (模型={self.model})")

    def assess(self, call_text: str, history: list[dict] | None = None) -> dict:
        """
        评估来电是否需要转接给机主。

        参数:
            call_text: 来电者说的话（语音转文字）
            history: 可选，之前的对话历史

        返回:
            dict: 评估结果
                {
                    "should_forward": bool,
                    "urgency_level": "high" | "medium" | "low",
                    "caller_identity": str,
                    "relationship": str,
                    "emergency_type": str,
                    "reason": str,
                    "agent_action": "forward" | "ask_more" | "reject" | "take_message",
                    "agent_text": str,
                    "notification": str
                }
        """
        if not call_text or not call_text.strip():
            return self._default_response("来电内容为空")

        logger.info(f"📞 紧急评估中... 来电内容: {call_text[:80]}...")

        # --- 构建 Prompt ---
        history_text = ""
        if history:
            for msg in history[-6:]:
                role = "来电者" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

        user_message = f"""## 对话历史
{history_text or "（首轮对话）"}

## 来电者最新说的话
{call_text}

请评估这个来电是否需要转接给机主。记住：默认不转接，只有确实重要紧急时才转接。"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": URGENT_FORWARDER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            llm_output = response.choices[0].message.content.strip()
            logger.debug(f"LLM 输出: {llm_output}")

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return self._rule_based_assess(call_text)

        # --- 解析 JSON ---
        result = self._parse_output(llm_output)

        if result.get("should_forward"):
            logger.warning(
                f"🔔 紧急来电需转接! "
                f"身份={result.get('caller_identity')}, "
                f"紧急度={result.get('urgency_level')}, "
                f"原因={result.get('reason')}"
            )
        else:
            logger.info(
                f"来电不满足转接条件: {result.get('reason', '无')}"
            )

        return result

    def _parse_output(self, llm_output: str) -> dict:
        """
        解析 LLM 的输出 JSON。
        """
        try:
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            return {
                "should_forward": bool(result.get("should_forward", False)),
                "urgency_level": str(result.get("urgency_level", "low")),
                "caller_identity": str(result.get("caller_identity", "")),
                "relationship": str(result.get("relationship", "陌生人")),
                "emergency_type": str(result.get("emergency_type", "none")),
                "reason": str(result.get("reason", "")),
                "agent_action": str(result.get("agent_action", "reject")),
                "agent_text": str(result.get("agent_text", "")),
                "notification": str(result.get("notification", "")),
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON 解析失败: {e}")
            return self._default_response("LLM 输出解析失败")

    def _rule_based_assess(self, call_text: str) -> dict:
        """
        基于规则的降级评估（LLM 不可用时使用）。

        检测紧急关键词和身份关键词，做简单的规则匹配。

        参数:
            call_text: 来电文本

        返回:
            dict: 评估结果
        """
        text = call_text.lower()

        # 检测紧急关键词
        urgent_matches = [kw for kw in URGENT_KEYWORDS if kw in text]
        has_urgency = len(urgent_matches) > 0

        # 检测身份关键词
        identity_matches = [kw for kw in IDENTITY_KEYWORDS if kw in text]
        has_identity = len(identity_matches) > 0

        # 决策逻辑
        if has_identity and has_urgency:
            return {
                "should_forward": True,
                "urgency_level": "high",
                "caller_identity": f"包含身份词: {', '.join(identity_matches)}",
                "relationship": "亲友（规则判定）",
                "emergency_type": "unknown",
                "reason": f"包含身份关键词({', '.join(identity_matches)})和紧急关键词({', '.join(urgent_matches)})",
                "agent_action": "forward",
                "agent_text": "好的，我马上帮您转接机主。",
                "notification": f"🔔 紧急来电！\n来电内容: {call_text[:200]}\n紧急词: {', '.join(urgent_matches)}",
            }
        elif has_urgency:
            return {
                "should_forward": False,
                "urgency_level": "medium",
                "caller_identity": "未知（规则判定）",
                "relationship": "未知",
                "emergency_type": "unknown",
                "reason": "有紧急关键词但无法确认身份",
                "agent_action": "ask_more",
                "agent_text": "请问您是哪位？和机主是什么关系？",
                "notification": "",
            }
        else:
            return {
                "should_forward": False,
                "urgency_level": "low",
                "caller_identity": "",
                "relationship": "陌生人",
                "emergency_type": "none",
                "reason": "未检测到紧急关键词",
                "agent_action": "reject",
                "agent_text": "好的，我已经记录了。机主稍后会查看。",
                "notification": "",
            }

    def _default_response(self, reason: str) -> dict:
        """
        生成默认响应（无法判断时使用）。
        """
        return {
            "should_forward": False,
            "urgency_level": "low",
            "caller_identity": "",
            "relationship": "未知",
            "emergency_type": "none",
            "reason": reason,
            "agent_action": "take_message",
            "agent_text": "不好意思，信号不太好，请您再说一遍。",
            "notification": "",
        }


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试紧急转接 Agent:
        python -m src.agents.urgent_forwarder

    注意: 需要配置有效的 DeepSeek API Key。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.utils.logger import load_config

    config = load_config("config.yaml")
    forwarder = UrgentForwarder(config)

    test_cases = [
        ("紧急", "喂？我是你妈，你爸刚才摔倒了，现在在去医院的路上，你赶紧过来！"),
        ("紧急", "喂我是你老板，客户那边出问题了，你现在立刻上线开会！"),
        ("朋友闲聊", "喂老张，晚上有空吗，一起吃个饭？"),
        ("陌生人", "你好，我是XX保险公司的，想跟您介绍一下我们的新产品..."),
    ]

    for category, text in test_cases:
        print("\n" + "=" * 60)
        print(f"[{category}] {text[:60]}...")
        print("=" * 60)

        result = forwarder.assess(text)

        print(f"  是否转接: {'🔔 是! (转接机主)' if result['should_forward'] else '否'}")
        print(f"  紧急程度: {result['urgency_level']}")
        print(f"  来电身份: {result['caller_identity']}")
        print(f"  关系:     {result['relationship']}")
        print(f"  紧急类型: {result['emergency_type']}")
        print(f"  原因:     {result['reason']}")
        print(f"  Agent动作: {result['agent_action']}")
        print(f"  Agent回复: {result['agent_text']}")
        if result.get("notification"):
            print(f"  通知机主: {result['notification']}")
