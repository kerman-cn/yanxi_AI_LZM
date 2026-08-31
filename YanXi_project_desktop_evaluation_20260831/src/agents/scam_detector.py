"""
诈骗电话检测 Agent
-------------------
对来电内容进行诈骗风险评估。结合 RAG 检索（从知识库查找相似诈骗案例）
和 LLM 推理，判断当前来电是否为诈骗电话。

工作流程:
1. 接收来电的语音转文字文本
2. 通过 RAG 检索知识库中语义相似的历史案例
3. 将「来电文本 + RAG 案例」组合为 Prompt 发给 LLM
4. LLM 输出结构化的检测结果 (JSON)

LLM 输出格式:
{
    "is_scam": true/false,
    "scam_type": "冒充公检法" / "投资理财诈骗" / "" (正常来电为空),
    "confidence": 0.0~1.0,
    "reason": "判定理由的简要说明",
    "action": "reject" / "continue" / "uncertain"
}

使用方式:
    from src.agents.scam_detector import ScamDetector

    detector = ScamDetector(config)
    result = detector.detect("您好，我是公安局的...")
    if result["is_scam"]:
        print(f"⚠️ 诈骗电话: {result['scam_type']}")
"""

import json
import re

from openai import OpenAI

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

SCAM_DETECTION_SYSTEM_PROMPT = """你是一名电信诈骗检测专家。你的任务是根据来电内容和参考案例，判断来电是否为诈骗电话。

## 判断标准
请从以下维度综合判断：

1. **身份冒充**：是否自称公检法、银行、客服、快递公司等机构人员？
2. **紧急威胁**：是否制造紧迫感，如"限时处理"、"不配合就逮捕"、"账户即将冻结"？
3. **索要敏感信息**：是否索要银行卡号、密码、验证码、身份证号？
4. **诱导转账**：是否要求转账到"安全账户"、缴纳保证金、支付税费？
5. **异常利益承诺**：是否承诺中奖、高回报投资、低价商品等不合理的利益？
6. **号码伪装**：是否声称来自官方但行为异常？

## 常见的诈骗类型
- **冒充公检法**：自称公安局/检察院/法院，声称涉嫌犯罪，要求配合调查
- **冒充客服**：自称电商/银行客服，声称订单异常、退款、注销会员
- **投资理财诈骗**：承诺高收益、稳赚不赔，诱导下载投资APP
- **贷款诈骗**：声称无抵押低息贷款，要求先交保证金/手续费
- **兼职刷单诈骗**：声称高佣金兼职，要求先垫付资金
- **网购退款诈骗**：声称商品有问题要退款，索要银行卡信息
- **杀猪盘/交友诈骗**：通过感情建立信任后诱导投资或借钱
- **中奖诈骗**：声称中奖，要求先缴纳税费
- **冒充熟人**：冒充亲友/领导，以急事为由要求转账

## 重要提示
- 如果来电内容很短或信息不足，请保守判断，标记为 uncertain
- 外卖、快递、正常商务沟通不应被判定为诈骗
- 考虑中国大陆常见的诈骗话术和套路
- 结合参考案例进行对比分析

## 输出格式
请严格按照以下 JSON 格式输出，不要输出其他内容：
```json
{
    "is_scam": true或false,
    "scam_type": "诈骗类型，正常来电为空字符串",
    "confidence": 0.0到1.0之间的置信度,
    "reason": "简短的判定理由，50字以内",
    "action": "reject拒接 / continue继续处理 / uncertain不确定需进一步确认"
}
```"""


class ScamDetector:
    """
    诈骗电话检测 Agent。

    结合 RAG 检索和 LLM 推理，对来电内容进行诈骗风险评估。
    如果判定为诈骗，直接给出 reject 指令；如果无法确定，标记为 uncertain。
    """

    def __init__(self, config: dict):
        """
        初始化诈骗检测 Agent。

        参数:
            config: 全局配置字典
        """
        llm_cfg = config.get("llm", {})
        scam_cfg = config.get("orchestrator", {})

        # --- 初始化 LLM 客户端 (DeepSeek, OpenAI 兼容接口) ---
        self.llm = OpenAI(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        )
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens = llm_cfg.get("max_tokens", 2048)

        # --- 初始化 RAG 检索器 ---
        self.retriever = ScamKnowledgeRetriever(config)

        # 诈骗判定阈值
        self.confidence_threshold = scam_cfg.get("scam_confidence_threshold", 0.7)

        logger.info(f"诈骗检测 Agent 已初始化 (模型={self.model}, RAG_K={self.retriever.top_k})")

    def detect(self, call_text: str) -> dict:
        """
        检测来电是否为诈骗电话。

        参数:
            call_text: 来电的语音转文字文本（第一段/开场白）

        返回:
            dict: 检测结果
                {
                    "is_scam": bool,
                    "scam_type": str,
                    "confidence": float,
                    "reason": str,
                    "action": "reject" | "continue" | "uncertain"
                }
        """
        if not call_text or not call_text.strip():
            logger.warning("来电文本为空，无法检测")
            return {
                "is_scam": False,
                "scam_type": "",
                "confidence": 0.0,
                "reason": "来电内容为空",
                "action": "uncertain",
            }

        logger.info(f"🔍 诈骗检测中... 来电内容: {call_text[:80]}...")

        # --- 第一步: RAG 检索相似案例 ---
        # 尝试检索，如果知识库不可用则跳过
        rag_context = ""
        try:
            rag_context = self.retriever.retrieve_as_context(call_text)
            logger.debug(f"RAG 检索完成，上下文长度: {len(rag_context)} 字符")
        except Exception as e:
            logger.warning(f"RAG 检索失败（将仅使用 LLM 判断）: {e}")
            rag_context = "（知识库暂不可用，请仅根据来电内容进行判断）"

        # --- 第二步: 构建 Prompt 并调用 LLM ---
        user_message = f"""## 来电内容
{call_text}

## 参考案例（来自诈骗知识库）
{rag_context}

请根据以上信息，判断该来电是否为诈骗电话。"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SCAM_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            llm_output = response.choices[0].message.content.strip()
            logger.debug(f"LLM 原始输出: {llm_output}")

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            # 降级策略：LLM 不可用时，仅根据 RAG 相似度判断
            return self._fallback_detection(call_text)

        # --- 第三步: 解析 LLM 输出的 JSON ---
        result = self._parse_llm_output(llm_output)

        # --- 第四步: 根据置信度阈值调整 action ---
        if result["is_scam"]:
            if result["confidence"] >= self.confidence_threshold:
                result["action"] = "reject"
                logger.warning(
                    f"⚠️ 检测到诈骗电话! "
                    f"类型={result['scam_type']}, 置信度={result['confidence']:.0%}"
                )
            else:
                result["action"] = "uncertain"
                logger.info(
                    f"疑似诈骗但置信度不足 ({result['confidence']:.0%} < {self.confidence_threshold:.0%})，"
                    f"标记为 uncertain"
                )
        else:
            result["action"] = "continue"
            logger.info(f"✅ 正常来电 (置信度={result['confidence']:.0%})")

        return result

    def _parse_llm_output(self, llm_output: str) -> dict:
        """
        解析 LLM 输出的 JSON，做容错处理。
        如果 LLM 输出格式异常，尝试用正则提取或返回安全的默认值。

        参数:
            llm_output: LLM 的原始输出文本

        返回:
            dict: 解析后的检测结果
        """
        # 尝试直接从输出中提取 JSON
        try:
            # 移除可能的 markdown 代码块标记
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            # 验证必要字段
            return {
                "is_scam": bool(result.get("is_scam", False)),
                "scam_type": str(result.get("scam_type", "")),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": str(result.get("reason", "")),
                "action": str(result.get("action", "continue")),
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON 解析失败，尝试正则提取: {e}")

        # 降级：正则提取关键字段
        is_scam = _extract_bool(llm_output, "is_scam", default=False)
        confidence = _extract_float(llm_output, "confidence", default=0.5)
        scam_type = _extract_str(llm_output, "scam_type", default="")
        reason = _extract_str(llm_output, "reason", default="解析失败")

        return {
            "is_scam": is_scam,
            "scam_type": scam_type,
            "confidence": confidence,
            "reason": reason,
            "action": "reject" if (is_scam and confidence >= self.confidence_threshold) else "uncertain",
        }

    def _fallback_detection(self, call_text: str) -> dict:
        """
        当 LLM 不可用时的降级检测策略。
        仅根据 RAG 检索结果的相似度来判断。

        参数:
            call_text: 来电文本

        返回:
            dict: 降级检测结果
        """
        try:
            results = self.retriever.retrieve(call_text, top_k=3, filter_fraud_only=True)
        except Exception:
            results = []

        if not results:
            return {
                "is_scam": False,
                "scam_type": "",
                "confidence": 0.0,
                "reason": "LLM 不可用且无匹配案例，无法判断",
                "action": "uncertain",
            }

        # 取最高相似度
        top_similarity = results[0]["similarity"]
        top_type = results[0]["metadata"].get("fraud_type", "")

        if top_similarity > 0.8:
            return {
                "is_scam": True,
                "scam_type": top_type,
                "confidence": top_similarity,
                "reason": f"与已知诈骗案例高度相似 (相似度={top_similarity:.0%})",
                "action": "reject",
            }
        else:
            return {
                "is_scam": False,
                "scam_type": "",
                "confidence": 1.0 - top_similarity,
                "reason": "降级模式: 知识库匹配度不足",
                "action": "uncertain",
            }


# ============================================================
# JSON 解析辅助函数（容错用）
# ============================================================

def _extract_bool(text: str, field: str, default: bool = False) -> bool:
    """从文本中用正则提取布尔字段的值。"""
    # 匹配 "is_scam": true 或 "is_scam": false
    pattern = rf'"{field}"\s*:\s*(true|false)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return default


def _extract_float(text: str, field: str, default: float = 0.5) -> float:
    """从文本中用正则提取浮点数字段的值。"""
    pattern = rf'"{field}"\s*:\s*([\d.]+)'
    match = re.search(pattern, text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))  # 截断到 0~1
        except ValueError:
            pass
    return default


def _extract_str(text: str, field: str, default: str = "") -> str:
    """从文本中用正则提取字符串字段的值。"""
    pattern = rf'"{field}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return default


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试诈骗检测 Agent:
        python -m src.agents.scam_detector

    需要配置有效的 DeepSeek API Key（在 config.yaml 中）。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.utils.logger import load_config

    config = load_config("config.yaml")

    detector = ScamDetector(config)

    # 模拟不同类型的来电
    test_cases = [
        ("诈骗", "您好，我是市公安局的，你涉嫌一起洗钱案件，请配合我们调查，把你的身份证号和银行卡号告诉我。"),
        ("诈骗", "恭喜您获得了我司十周年抽奖活动的一等奖，奖金30万元！请先缴纳个人所得税3000元到这个账户。"),
        ("外卖", "喂你好，我是美团外卖的，你的外卖到了，我现在在楼下，你下来拿一下还是放门卫？"),
        ("快递", "您好我是顺丰快递的，有您一个包裹到了，请问您现在在家吗？"),
        ("正常", "喂？老张吗？我是你表姐啊，妈住院了，你赶紧来市医院一趟！"),
    ]

    for category, text in test_cases:
        print("\n" + "=" * 60)
        print(f"[{category}] {text[:60]}...")
        print("=" * 60)

        result = detector.detect(text)

        print(f"  是否诈骗: {'⚠️ 是' if result['is_scam'] else '✅ 否'}")
        print(f"  诈骗类型: {result['scam_type'] or '无'}")
        print(f"  置信度:   {result['confidence']:.0%}")
        print(f"  判定理由: {result['reason']}")
        print(f"  处理动作: {result['action']}")
