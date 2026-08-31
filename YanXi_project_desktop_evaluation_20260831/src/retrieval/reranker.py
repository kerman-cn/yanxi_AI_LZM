"""
重排序器
========
对融合后的检索结果进行二次排序。

参考来源:
- CC 项目: 语义相似度 + 关键词命中 + 原始相似度 综合公式
- KCN 项目: embedder 可选容错设计

设计理念:
  保留 embedder 优先 + 关键词降级的容错策略，
  直接用 embedder 算语义相似度 + 关键词命中做综合排序。
"""

from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    重排序器。

    使用 embedder 计算语义相似度，结合关键词命中率和原始相似度做综合排序。
    embedder 不可用时自动降级为纯关键词+原始相似度排序。

    使用方式:
        reranker = Reranker(embedder=embedder)
        reranked = reranker.rerank("查询", docs, top_n=5)
    """

    def __init__(
        self,
        embedder=None,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        # 原始相似度权重 = 1 - semantic_weight - keyword_weight
        self.orig_weight = max(0.05, 1.0 - semantic_weight - keyword_weight)

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_n: int = 5,
    ) -> List[dict]:
        """
        对文档列表重新排序。

        参数:
            query: 查询文本
            documents: 文档列表
            top_n: 返回数量

        返回:
            list[dict]: 重排后的 top_n 文档
        """
        if not documents:
            return []

        query_terms = set(query)

        # 计算语义相似度
        if self.embedder and len(documents) > 0:
            try:
                contents = [d.get("content", "")[:500] for d in documents]
                query_vec = self.embedder.encode(query)
                doc_vecs = self.embedder.encode(contents)
                import numpy as np
                similarities = np.dot(doc_vecs, query_vec)
                # 归一化到 [0, 1]
                sim_max = float(np.max(similarities)) if len(similarities) > 0 else 1.0
                if sim_max > 0:
                    similarities = similarities / sim_max
            except Exception as e:
                logger.warning(f"语义相似度计算失败: {e}")
                similarities = [0.5] * len(documents)
        else:
            similarities = [0.5] * len(documents)

        scored = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")

            # 关键词命中率
            if len(query_terms) > 0:
                term_hits = sum(1 for t in query_terms if t in content) / len(query_terms)
            else:
                term_hits = 0.0

            # 原始相似度
            orig_score = doc.get("similarity", 0.5)

            # 综合得分
            combined = (
                float(similarities[i]) * self.semantic_weight
                + term_hits * self.keyword_weight
                + orig_score * self.orig_weight
            )
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
