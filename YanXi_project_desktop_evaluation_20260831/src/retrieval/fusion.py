"""
RRF 融合器
==========
Reciprocal Rank Fusion，将多路检索结果融合排序。

参考来源:
- CC 项目: RRFFuser 类 + doc_key() + k_const 参数
- KCN 项目: fuse_weighted() 加权融合

当前用于 BM25 + 向量 双路融合，保留扩展多路接口。
"""

from collections import defaultdict
from typing import List, Dict


def doc_key(doc: dict) -> str:
    """生成文档唯一标识"""
    return doc.get("id", "") or doc.get("content", "")[:100]


class RRFFuser:
    """
    RRF 融合器。

    使用方式:
        fuser = RRFFuser(k_const=60)
        fused = fuser.fuse_two(docs_a, docs_b, weight_a=0.4, weight_b=0.6)
    """

    def __init__(self, k_const: int = 60):
        self.k_const = k_const

    def fuse_two(
        self,
        docs_a: List[dict],
        docs_b: List[dict],
        weight_a: float = 0.4,
        weight_b: float = 0.6,
    ) -> List[dict]:
        """
        双路 RRF 加权融合。

        参数:
            docs_a: 第一路文档列表
            docs_b: 第二路文档列表
            weight_a: 第一路权重
            weight_b: 第二路权重

        返回:
            list[dict]: 融合后的文档列表，按 RRF 分数降序
        """
        scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, dict] = {}

        for rank, doc in enumerate(docs_a or []):
            key = doc_key(doc)
            scores[key] += weight_a / (self.k_const + rank + 1)
            store.setdefault(key, doc)

        for rank, doc in enumerate(docs_b or []):
            key = doc_key(doc)
            scores[key] += weight_b / (self.k_const + rank + 1)
            store.setdefault(key, doc)

        ranked_keys = sorted(store.keys(), key=lambda k: scores[k], reverse=True)
        return [store[k] for k in ranked_keys]

    def fuse(self, *doc_lists: List[dict]) -> List[dict]:
        """
        多路等权 RRF 融合（兼容旧接口）。

        参数:
            *doc_lists: 多个文档列表

        返回:
            list[dict]: 融合后的文档列表
        """
        scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, dict] = {}

        for docs in doc_lists:
            if not docs:
                continue
            for rank, doc in enumerate(docs):
                key = doc_key(doc)
                scores[key] += 1.0 / (self.k_const + rank + 1)
                store.setdefault(key, doc)

        ranked_keys = sorted(store.keys(), key=lambda k: scores[k], reverse=True)
        return [store[k] for k in ranked_keys]
