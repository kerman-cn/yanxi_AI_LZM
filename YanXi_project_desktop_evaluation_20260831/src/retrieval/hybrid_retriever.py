"""
双路混合检索器
==============
BM25 关键词检索 + ChromaDB 向量语义检索 → RRF 融合。

参考来源:
- CC 项目: RRF 融合 + 重排序管线
- Terry 项目: ChromaDB 纯语义检索 + L2 距离阈值
- LZM 项目: 混合检索中的关键词加权

设计理念:
  简洁双路架构（BM25 0.4 + 向量 0.6），
  去掉知识图谱这一路（通话助手场景下"大炮打蚊子"），
  用 BM25 替代 n-gram 倒排索引（更标准、更高效）。
"""

from typing import List

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.fusion import RRFFuser
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    双路混合检索器：BM25 + ChromaDB 向量。

    双路加权架构：BM25（权重 0.4）+ 向量（权重 0.6），RRF 融合。

    使用方式:
        hr = HybridRetriever(config, embedder)
        hr.index_documents(docs)          # 构建 BM25 索引
        docs = hr.retrieve("查询", top_k=5)  # 双路检索融合
    """

    def __init__(self, config: dict, embedder=None):
        self.config = config
        self.embedder = embedder

        # 初始化两路检索器
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(config, embedder)

        # RRF 融合器
        hybrid_cfg = config.get("hybrid_retrieval", {})
        self.fuser = RRFFuser(k_const=hybrid_cfg.get("rrf_k", 60))

        # 双路权重：BM25 0.4, 向量 0.6
        self.bm25_weight = hybrid_cfg.get("bm25_weight", 0.4)
        self.vector_weight = hybrid_cfg.get("vector_weight", 0.6)

        logger.info("HybridRetriever 初始化完成 (BM25 + 向量双路)")

    def index_documents(self, documents: List[dict]) -> None:
        """
        构建 BM25 索引（向量库索引通过 VectorRetriever.add_documents）。

        参数:
            documents: 文档列表
        """
        self.bm25.index_documents(documents)
        self.vector.add_documents(documents)
        logger.info(f"双路索引构建完成: {len(documents)} 篇文档")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        双路混合检索。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            list[dict]: 融合后的文档列表
        """
        # 两路并行检索
        bm25_docs = self.bm25.retrieve(query, top_k=top_k)
        vector_docs = self.vector.retrieve(query, top_k=top_k)

        # RRF 融合（双路加权）
        fused = self.fuser.fuse_two(
            docs_a=bm25_docs,
            docs_b=vector_docs,
            weight_a=self.bm25_weight,
            weight_b=self.vector_weight,
        )

        return fused[:top_k]

    def is_available(self) -> bool:
        """检查是否有任一路可用"""
        return self.bm25.is_available() or self.vector.is_available()
