"""
业务来电处理 Agent
------------------
处理外卖、快递、家政等服务类来电。通过多轮对话与来电者交流，
从对话中逐步提取关键信息，最终生成结构化的"信息摘要卡片"呈现给机主。

对话策略:
- 第一轮: 确认来电者身份和来意（你是谁？有什么事？）
- 后续: 针对性提问，逐步收集所需信息
- 最后一轮: 确认信息无误，告知机主会看到

提取的信息字段 (Schema):
- call_type:       来电类型 (外卖/快递/家政/其他)
- company:         公司/平台名称 (如"美团外卖"、"顺丰快递")
- item_description: 物品描述 (如"一份黄焖鸡米饭")
- location:        送达/取件/服务地点
- contact_person:  联系人称呼
- additional_notes: 其他备注 (如"放门口"、"放门卫"、"3号楼")

使用方式:
    from src.agents.business_handler import BusinessHandler

    handler = BusinessHandler(config)
    # 单轮处理
    result = handler.process_turn(call_text, history)
    # 检查是否信息收集完成
    if result["is_complete"]:
        print(result["summary_card"])
"""

import json

from openai import OpenAI

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

BUSINESS_SYSTEM_PROMPT = """你是AI助手，替机主接电话。对方是外卖员/快递员/司机/家政等服务人员。

## 你的任务
收集关键信息并生成摘要卡片给机主，按重要性排序：
1. **放哪里** — 对方最关心的，优先处理。放门口/外卖柜/菜鸟驿站/传达室
2. **哪个平台** — 美团/饿了么/顺丰/圆通/滴滴？对方没主动说就多问一句
3. **什么东西** — 外卖/包裹/文件/生鲜？有助于机主判断要不要马上去取
4. **特殊要求** — 冷藏食品快点拿/到付需要现金/放快递柜短信通知

## 对话策略
- 对方只说了"外卖到了" → 问一句"哪个平台的？放门口行吗？"
- 对方说了平台+物品 → 直接说放哪里，别啰嗦
- 收集到 2 项以上信息（平台+放哪里）就可以结束
- 口语化，友好，不超过 2 轮

## 示例
来电: "外卖到了"
回复: "好的，哪个平台的呢？放门口就行"  ← 追问平台
来电: "美团，餐到楼下了"
回复: "放楼下外卖柜吧，谢谢！"  ← 信息够了，is_complete=true
来电: "顺丰快递，有个包裹"
回复: "放菜鸟驿站吧，是到付的吗？"  ← 追问支付
来电: "不是到付，那我放驿站了"
回复: "好的，谢谢！"  ← is_complete=true

输出 JSON（不要 markdown 代码块）：
{"agent_text":"回复","is_complete":true或false,"collected_info":{"call_type":"外卖/快递/打车/家政","company":"平台","item_description":"物品","location":"地点","additional_notes":"备注"},"summary_text":"摘要"}
"""


class BusinessHandler:
    """
    业务来电处理 Agent。

    负责与外卖员、快递员、家政服务人员等来电者进行多轮对话，
    逐步提取关键信息，最终生成结构化的摘要卡片。

    典型对话示例:
        AI: 您好，机主现在不方便接电话，我是助手，请问您是哪里？
        来电者: 我是美团外卖的
        AI: 好的，请问送到哪里呢？
        来电者: 送到3号楼楼下
        AI: 明白了，已记录。美团外卖，送到3号楼楼下。机主稍后会看到，谢谢！
    """

    def __init__(self, config: dict):
        """
        初始化业务处理 Agent。

        参数:
            config: 全局配置字典
        """
        llm_cfg = config.get("llm", {})

        self.llm = OpenAI(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        )
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = llm_cfg.get("temperature", 0.5)  # 稍高温度使对话更自然
        self.max_tokens = llm_cfg.get("max_tokens", 2048)

        orchestrator_cfg = config.get("orchestrator", {})
        self.max_rounds = orchestrator_cfg.get("max_conversation_rounds", 10)

        logger.info(f"业务处理 Agent 已初始化 (模型={self.model})")

    def start_conversation(self) -> dict:
        """
        生成开场白，用于第一轮通话。
        来电者接通后，Agent 先说第一句话。
        开场白会根据机主当前状态动态生成。

        返回:
            dict: 包含 agent_text（开场白）和初始化状态
        """
        from src.utils.presence import get_presence
        presence = get_presence()
        mode = presence.get_mode()

        if mode == "free":
            opening = "好的，您说，我帮机主听着～"
        elif mode == "busy":
            opening = "机主在忙，您说，我帮您转达。"
        elif mode == "dnd":
            opening = "机主在休息，东西放门口就行，谢谢。"
        elif mode == "driving":
            opening = "机主在开车，您说，我转达。"
        else:
            opening = "好的，您说，我帮机主记一下。"
        return {
            "agent_text": opening,
            "is_complete": False,
            "collected_info": {
                "call_type": "",
                "company": "",
                "item_description": "",
                "location": "",
                "contact_person": "",
                "additional_notes": "",
            },
            "summary_text": "",
            "history": [],
        }

    def process_turn(
        self,
        caller_text: str,
        collected_info: dict,
        history: list[dict],
    ) -> dict:
        """
        处理一轮对话。

        接收来电者说的话，分析并决定下一步：
        - 如果信息还不够 → 生成追问，继续对话
        - 如果信息已足够 → 生成摘要卡片，结束对话

        参数:
            caller_text: 来电者本轮说的话（经 STT 转为文字）
            collected_info: 截至目前已收集到的信息字典
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]

        返回:
            dict: 包含 agent_text（回复）、is_complete（是否完成）、
                  collected_info（更新后的信息）、summary_text（摘要）
        """
        logger.info(f"💬 业务对话轮次 {len(history)//2 + 1}: 来电者说: {caller_text[:80]}...")

        # --- 构建包含对话历史和已收集信息的 Prompt ---
        # 将对话历史格式化为文本
        history_text = ""
        for msg in history[-10:]:  # 只取最近 10 条，避免 prompt 过长
            role = "来电者" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content']}\n"

        collected_json = json.dumps(collected_info, ensure_ascii=False, indent=2)

        user_message = f"""## 已收集的信息
{collected_json}

## 对话历史
{history_text}

## 来电者最新说的话
{caller_text}

请根据以上信息，决定下一步操作：
- 如果关键信息还不完整，生成一个简短的追问
- 如果信息已足够（至少要有 call_type, company, item_description 或 location），标记 is_complete=true 并生成摘要"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BUSINESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            llm_output = response.choices[0].message.content.strip()
            logger.debug(f"LLM 输出: {llm_output}")

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return self._fallback_response(e)

        # --- 解析 JSON 输出 ---
        result = self._parse_output(llm_output)

        # --- 合并已收集的信息 ---
        # 新版 collected_info 可能新增了字段，做一次合并
        new_info = result.get("collected_info", {})
        merged_info = {**collected_info, **{k: v for k, v in new_info.items() if v}}

        result["collected_info"] = merged_info
        result["history"] = history + [
            {"role": "user", "content": caller_text},
            {"role": "assistant", "content": result.get("agent_text", "")},
        ]

        # 检查是否达到最大对话轮次
        current_round = len(result["history"]) // 2
        if current_round >= self.max_rounds:
            logger.info(f"达到最大对话轮次 ({self.max_rounds})，强制结束")
            result["is_complete"] = True
            result["summary_text"] = self._generate_summary(merged_info)

        if result.get("is_complete"):
            logger.info(f"✅ 业务信息收集完成:")
            logger.info(f"  类型: {merged_info.get('call_type', '未识别')}")
            logger.info(f"  平台: {merged_info.get('company', '未识别')}")
            logger.info(f"  内容: {merged_info.get('item_description', '未识别')}")
            logger.info(f"  地点: {merged_info.get('location', '未识别')}")

        return result

    def _parse_output(self, llm_output: str) -> dict:
        """
        解析 LLM 的 JSON 输出，做容错处理。

        参数:
            llm_output: LLM 原始输出

        返回:
            dict: 解析后的结构化结果
        """
        try:
            # 清理 markdown 代码块
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            return json.loads(cleaned)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON 解析失败: {e}，使用降级输出")
            return {
                "agent_text": "好的，我已经记录了。机主稍后会看到您的信息，谢谢！",
                "is_complete": True,
                "collected_info": {},
                "summary_text": f"（原始回复: {llm_output[:200]}）",
            }

    def _generate_summary(self, info: dict) -> str:
        """
        根据收集到的信息生成摘要卡片文本。

        参数:
            info: 收集到的信息字典

        返回:
            str: 格式化的摘要文本
        """
        parts = ["📋 **来电摘要**", ""]

        if info.get("call_type"):
            parts.append(f"📞 类型: {info['call_type']}")
        if info.get("company"):
            parts.append(f"🏢 来源: {info['company']}")
        if info.get("item_description"):
            parts.append(f"📦 内容: {info['item_description']}")
        if info.get("location"):
            parts.append(f"📍 地点: {info['location']}")
        if info.get("contact_person"):
            parts.append(f"👤 联系人: {info['contact_person']}")
        if info.get("additional_notes"):
            parts.append(f"📝 备注: {info['additional_notes']}")

        return "\n".join(parts)

    def _fallback_response(self, error: Exception) -> dict:
        """
        LLM 调用失败时的降级响应。
        礼貌地告知来电者稍后再联系。

        参数:
            error: 异常信息

        返回:
            dict: 降级响应
        """
        return {
            "agent_text": "不好意思，信号不太好，我这边先记录一下，机主稍后给您回电。",
            "is_complete": True,
            "collected_info": {},
            "summary_text": f"（系统异常，通话中断: {str(error)[:100]}）",
            "history": [],
        }


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试业务处理 Agent:
        python -m src.agents.business_handler

    模拟外卖员来电的多轮对话。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.utils.logger import load_config

    config = load_config("config.yaml")
    handler = BusinessHandler(config)

    # 模拟外卖员来电的对话
    print("=" * 60)
    print("模拟: 外卖员来电")
    print("=" * 60)

    # 开场
    state = handler.start_conversation()
    print(f"\n🤖 助手: {state['agent_text']}")

    # 第1轮 - 来电者回答
    turn1 = handler.process_turn(
        "我是美团外卖的，你点的餐到了",
        state["collected_info"],
        state["history"],
    )
    print(f"\n📞 来电者: 我是美团外卖的，你点的餐到了")
    print(f"🤖 助手: {turn1['agent_text']}")
    print(f"   完成? {turn1['is_complete']}")

    # 第2轮 - 来电者回答
    turn2 = handler.process_turn(
        "送到3号楼下吧，快到的时候给我打电话",
        turn1["collected_info"],
        turn1["history"],
    )
    print(f"\n📞 来电者: 送到3号楼下吧，快到的时候给我打电话")
    print(f"🤖 助手: {turn2['agent_text']}")
    print(f"   完成? {turn2['is_complete']}")

    if turn2.get("is_complete"):
        print(f"\n{'='*40}")
        print("📋 摘要卡片:")
        print(turn2.get("summary_text", "无"))
        print(f"{'='*40}")
