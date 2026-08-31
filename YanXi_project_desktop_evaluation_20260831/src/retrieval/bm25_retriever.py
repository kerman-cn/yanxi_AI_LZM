"""
BM25 关键词检索器
==================
第一路检索：基于 BM25 算法的稀疏检索，对中文来电文本做精确关键词匹配。

参考来源:
- CC 项目: SCAM_KEYWORD_WEIGHTS 关键词加权策略
- Terry 项目: RAG_TOP_K 可配置 top-k
- LZM 项目: n-gram 倒排索引中的关键词加权思路

设计理念:
  简洁高效 — 不构建复杂倒排索引，直接用 BM25 统计算法
  兼顾中文 — jieba 分词 + BM25，比纯 n-gram 更准确
"""

import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 关键词权重表（来自 CC/LZM 项目，用于辅助加权）
KEYWORD_WEIGHTS: Dict[str, float] = {
    # 高危词
    "转账": 3.0, "安全账户": 3.0, "保证金": 3.0, "手续费": 2.5,
    "验证码": 3.0, "银行卡号": 3.0, "密码": 2.5, "身份证号": 3.0,
    "通缉令": 3.0, "逮捕": 3.0, "涉嫌犯罪": 3.0, "洗钱": 3.0,
    "公安局": 2.5, "检察院": 2.5, "法院": 2.5, "公检法": 2.5,
    "资金冻结": 2.5, "冻结": 2.0, "冒充": 2.0, "诈骗": 2.0, "骗": 1.5,
    # 中危词
    "退款": 2.0, "赔偿": 2.0, "中奖": 2.5, "奖金": 2.0,
    "客服": 1.5, "注销": 2.0, "征信": 2.0, "逾期": 2.0,
    "稳赚": 2.5, "高收益": 2.5, "理财": 1.5, "投资": 1.5,
    "贷款": 2.0, "无抵押": 2.0, "低息": 2.0,
    "刷单": 2.5, "兼职": 1.5, "日赚": 2.5, "垫付": 2.5,
    "杀猪盘": 3.0, "网恋": 2.0, "博彩": 2.5,
    # 正常词（负权重，降低误判）
    "外卖": -1.5, "快递": -1.5, "包裹": -1.5, "美团": -1.5,
    "饿了么": -1.5, "顺丰": -1.5, "取餐": -1.5, "放门口": -1.0,
    "楼下": -1.0, "开会": -1.0, "吃饭": -1.0, "回家": -1.0,
}


def _tokenize(text: str) -> List[str]:
    """中文分词，优先 jieba，回退字符级"""
    try:
        import jieba
        # 清理空白
        clean = re.sub(r'\s+', '', text)
        if not clean:
            return []
        return list(jieba.cut(clean))
    except ImportError:
        # 回退：2-gram 字符级分词
        clean = re.sub(r'\s+', '', text)
        tokens = list(clean)
        for i in range(len(clean) - 1):
            tokens.append(clean[i:i + 2])
        return tokens


class BM25Retriever:
    """
    BM25 关键词检索器。

    基于 BM25Okapi 实现，结合 CC/LZM 的关键词加权策略，
    增强中文来电场景的检索效果。

    使用方式:
        retriever = BM25Retriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("查询文本", top_k=5)
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[dict] = []
        self._tokenized_docs: List[List[str]] = []

    def index_documents(self, documents: List[dict]) -> None:
        """
        构建 BM25 索引。

        参数:
            documents: 文档列表，每个包含 content/text
        """
        self._documents = documents
        self._tokenized_docs = []

        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            tokens = _tokenize(text)
            self._tokenized_docs.append(tokens)

        if self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)
            logger.info(f"BM25 索引构建完成: {len(documents)} 篇文档")
        else:
            self._bm25 = None
            logger.warning("BM25 索引为空")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        BM25 检索。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            list[dict]: 文档列表，含 id/content/similarity/metadata
        """
        if not self._bm25 or not self._documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # BM25 基础得分
        scores = self._bm25.get_scores(query_tokens)

        # 关键词加权调整（参考 CC/LZM）
        adjusted_scores = self._apply_keyword_weight(query, list(scores))

        # 排序取 top_k
        ranked = sorted(
            enumerate(adjusted_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            doc = self._documents[idx]
            # 归一化到 [0, 1]
            max_score = max(adjusted_scores) if max(adjusted_scores) > 0 else 1
            normalized = round(min(score / max_score, 1.0), 4)
            results.append({
                "id": doc.get("id", f"bm25_{idx}"),
                "content": doc.get("content", doc.get("text", "")),
                "similarity": normalized,
                "metadata": doc.get("metadata", {}),
            })

        return results

    def _apply_keyword_weight(self, query: str, bm25_scores: List[float]) -> List[float]:  # type: ignore[arg-type]
        """应用关键词加权调整（来自 CC/LZM 的策略）"""
        adjusted = list(bm25_scores)

        # 计算 query 中的关键词加权系数
        query_bonus = 0.0
        for kw, weight in KEYWORD_WEIGHTS.items():
            if kw in query:
                query_bonus += weight * 0.1  # 缩小影响避免过度

        # 对每个文档调整得分
        for idx, doc in enumerate(self._documents):
            text = doc.get("content", doc.get("text", ""))
            doc_bonus = 0.0
            for kw, weight in KEYWORD_WEIGHTS.items():
                if kw in query and kw in text:
                    doc_bonus += weight * 0.15
                    if weight >= 2.5:  # 高危词额外加分
                        doc_bonus += weight * 0.1

            if query_bonus != 0 or doc_bonus != 0:
                adjusted[idx] = bm25_scores[idx] * (1.0 + doc_bonus * 0.3)

        return adjusted

    def is_available(self) -> bool:
        """检查是否已索引"""
        return self._bm25 is not None and len(self._documents) > 0
