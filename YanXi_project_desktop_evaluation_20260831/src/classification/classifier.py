"""
三级来电分类器
==============
关键词匹配(快速) → RAG检索(辅助) → LLM增强(兜底)

从 CC 项目整合而来，使用枚举常量替代字符串硬编码。
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.enums import CallType, ClassifyMethod
from src.classification.keywords import CATEGORY_KEYWORDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassifyResult:
    """分类结果"""
    type_id: str
    call_type: CallType
    confidence: float
    method: str  # 使用 ClassifyMethod 的值
    matched_keywords: list = field(default_factory=list)
    reason: str = ""


class ThreeTierClassifier:
    """
    三级分类器：关键词 → RAG → LLM

    使用方式:
        classifier = ThreeTierClassifier(config, retriever=hybrid_retriever)
        result = classifier.classify("我是美团外卖的，你的餐到了")
    """

    def __init__(self, config: dict, retriever=None, llm_client=None):
        self.config = config
        self.retriever = retriever
        self.llm_client = llm_client
        self._llm_enabled = llm_client is not None

    def classify(self, text: str) -> ClassifyResult:
        """
        三级分类入口。

        参数:
            text: 来电语音转文字

        返回:
            ClassifyResult: 分类结果
        """
        if not text or not text.strip():
            return ClassifyResult(
                type_id=CallType.MEANINGLESS.type_id,
                call_type=CallType.MEANINGLESS,
                confidence=0.0,
                method=ClassifyMethod.NONE.value,
            )

        # 第 0 步：无意义检测
        if is_meaningless(text):
            return ClassifyResult(
                type_id=CallType.MEANINGLESS.type_id,
                call_type=CallType.MEANINGLESS,
                confidence=0.95,
                method=ClassifyMethod.MEANINGLESS_DETECT.value,
            )

        # 第 1 步：关键词匹配
        kw_result = _keyword_match(text)
        if kw_result and kw_result.confidence >= 0.7:
            logger.info(f"分类: {kw_result.call_type.display_name} (关键词, 置信度={kw_result.confidence:.0%})")
            return kw_result

        # 关键词弱匹配 → 信任关键词
        if kw_result and kw_result.confidence >= 0.55:
            logger.info(f"分类: {kw_result.call_type.display_name} (关键词弱匹配, 置信度={kw_result.confidence:.0%})")
            return kw_result

        # 第 2 步：RAG 检索辅助
        rag_result = self._rag_classify(text, kw_result)
        if rag_result and rag_result.confidence >= 0.7:
            logger.info(f"分类: {rag_result.call_type.display_name} (RAG, 置信度={rag_result.confidence:.0%})")
            return rag_result

        # 第 3 步：LLM 增强
        if self._llm_enabled and (not kw_result or kw_result.confidence < 0.7):
            llm_result = self._llm_classify(text)
            if llm_result and llm_result.confidence >= 0.7:
                logger.info(f"分类: {llm_result.call_type.display_name} (LLM, 置信度={llm_result.confidence:.0%})")
                return llm_result

        # 兜底
        if kw_result:
            logger.info(f"分类: {kw_result.call_type.display_name} (关键词, 低置信度={kw_result.confidence:.0%})")
            return kw_result

        return ClassifyResult(
            type_id=CallType.GENERAL.type_id,
            call_type=CallType.GENERAL,
            confidence=0.3,
            method=ClassifyMethod.DEFAULT.value,
            reason="无法确定来电类型",
        )

    def _rag_classify(self, text: str, kw_result: Optional[ClassifyResult] = None) -> Optional[ClassifyResult]:
        """通过 RAG 检索辅助分类"""
        if self.retriever is None:
            return None

        try:
            results = self.retriever.retrieve(text, top_k=5)
            if not results:
                return None

            from collections import Counter
            type_ids = []
            for r in results:
                metadata = r.get("metadata", {})
                label = metadata.get("label", "")
                if label == "fraud":
                    type_ids.append("scam")
                else:
                    content = r.get("content", "")
                    for tid, keywords in CATEGORY_KEYWORDS.items():
                        if any(kw in content for kw in keywords[:3]):
                            type_ids.append(tid)
                            break
                    else:
                        type_ids.append("general")

            if not type_ids:
                return None

            counter = Counter(type_ids)
            top_type_id, count = counter.most_common(1)[0]
            confidence = min(0.85, 0.5 + (count / len(type_ids)) * 0.35)

            return ClassifyResult(
                type_id=top_type_id,
                call_type=CallType.from_id(top_type_id),
                confidence=confidence,
                method=ClassifyMethod.RAG.value,
                reason=f"检索到 {count}/{len(type_ids)} 条匹配",
            )
        except Exception as e:
            logger.debug(f"RAG 分类失败: {e}")
            return None

    def _llm_classify(self, text: str) -> Optional[ClassifyResult]:
        """通过 LLM 分类（最后一层兜底）"""
        if self.llm_client is None:
            return None

        try:
            type_list = "\n".join([
                f"- {ct.type_id}: {ct.name}"
                for ct in CallType if ct != CallType.MEANINGLESS
            ])

            prompt = f"""判断以下来电内容属于哪种类型，只输出类型ID。

来电: "{text}"

可选类型:
{type_list}
- general: 其他

只输出一个类型ID："""

            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )

            if not response or not response.strip():
                return None
            type_id = response.strip().lower()
            call_type = CallType.from_id(type_id)

            return ClassifyResult(
                type_id=call_type.type_id,
                call_type=call_type,
                confidence=0.9,
                method=ClassifyMethod.LLM.value,
            )
        except Exception as e:
            logger.warning(f"LLM 分类失败: {e}")
            return None


def _keyword_match(text: str) -> Optional[ClassifyResult]:
    """关键词快速匹配"""
    best_type_id = ""
    best_score = 0
    best_keywords = []

    for type_id, keywords in CATEGORY_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text]
        if not matched:
            continue

        if type_id in ("scam", "scam_risk"):
            score = min(0.98, 0.85 + len(matched) * 0.05)
        elif type_id in ("telemarketing", "game_promo"):
            score = min(0.90, 0.70 + len(matched) * 0.08) if len(matched) >= 2 else 0.55
        elif type_id in ("food_delivery", "express", "taxi_arrived"):
            score = min(0.92, 0.75 + len(matched) * 0.08) if len(matched) >= 2 else 0.60
        else:
            score = min(0.88, 0.60 + len(matched) * 0.10)

        if score > best_score:
            best_score = score
            best_type_id = type_id
            best_keywords = matched

    if best_score >= 0.6:
        return ClassifyResult(
            type_id=best_type_id,
            call_type=CallType.from_id(best_type_id),
            confidence=best_score,
            method=ClassifyMethod.KEYWORD.value,
            matched_keywords=best_keywords,
            reason=f"关键词匹配: {', '.join(best_keywords[:5])}",
        )

    return None


def is_meaningless(text: str) -> bool:
    """检测 STT 输出是否为无意义内容"""
    text = text.strip()
    if len(text) < 2:
        return True

    clean = re.sub(r'\s+', '', text)
    if len(clean) < 2:
        return True

    # 连续数字 >= 15 位
    if re.search(r'\d{15,}', clean):
        return True

    # 同一个单字重复 >= 3 次且占比 >= 40%
    for ch in set(clean):
        count = clean.count(ch)
        if count >= 3 and count / len(clean) >= 0.4:
            return True

    # 同一个 2-3 字词组重复 >= 3 次
    for wlen in [2, 3]:
        seen = set()
        for i in range(len(clean) - wlen + 1):
            word = clean[i:i + wlen]
            if word in seen:
                continue
            seen.add(word)
            if clean.count(word) >= 3 and len(word) * clean.count(word) >= len(clean) * 0.5:
                return True

    # 中文数字 + 阿拉伯数字占比 >= 60%
    numerals = set('一二三四五六七八九十百千万亿零两')
    if len(clean) >= 3:
        numeral_count = sum(1 for ch in clean if ch in numerals or ch.isdigit())
        if numeral_count / len(clean) >= 0.6:
            meaningful = {'送餐', '取餐', '取件', '打钱', '汇款', '快递', '外卖', '到了', '面试', '开会', '同学'}
            if not any(kw in clean for kw in meaningful):
                return True

    return False
