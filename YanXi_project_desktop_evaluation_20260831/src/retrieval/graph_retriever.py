"""
知识图谱检索器
==============
第二路检索：基于 NetworkX 知识图谱 + Louvain 社区检测。
从 KCN 项目整合而来。
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional

from src.core.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """
    知识图谱检索器。

    检索策略:
    1. 基于查询关键词匹配社区摘要
    2. 返回社区内最相关的通话记录
    3. 支持社区摘要向量检索（可选）

    使用方式:
        retriever = GraphRetriever(config)
        docs = retriever.retrieve("诈骗电话", k=5)
    """

    def __init__(self, config: dict, embedder=None):
        kg_cfg = config.get("knowledge_graph", {})
        self.config = config

        self.graph_path = str(PROJECT_ROOT / kg_cfg.get("graph_path", "data/graph/graph.pkl").lstrip("./"))
        self.communities_path = str(PROJECT_ROOT / kg_cfg.get("communities_path", "data/graph/communities.json").lstrip("./"))
        self.top_k = kg_cfg.get("graph_k_calls", 5)
        self.embedder = embedder
        self.summary_db = None

        self.graph = None
        self.communities = None
        self._load()

    def _load(self):
        """加载图谱和社区数据"""
        # 加载图谱
        graph_file = Path(self.graph_path)
        if graph_file.exists():
            try:
                if graph_file.suffix == '.pkl':
                    try:
                        with open(graph_file, 'rb') as f:
                            self.graph = pickle.load(f)
                        if self.graph is not None:
                            logger.info(f"加载图谱(pickle): {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
                    except (pickle.UnpicklingError, UnicodeDecodeError, EOFError):
                        with open(graph_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        self._build_graph_from_json(data)
                else:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._build_graph_from_json(data)
                    if self.graph is not None:
                        logger.info(f"加载图谱(JSON): {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
            except Exception as e:
                logger.warning(f"图谱加载失败: {e}")

        # 加载社区
        communities_file = Path(self.communities_path)
        if communities_file.exists():
            try:
                with open(communities_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.communities = {str(c.get("community_id", i)): c for i, c in enumerate(data)}
                else:
                    self.communities = data
                logger.info(f"加载社区: {len(self.communities)} 个")
            except Exception as e:
                logger.warning(f"社区加载失败: {e}")

    def _build_graph_from_json(self, data: dict) -> None:
        """从 JSON 数据构建 NetworkX 图"""
        import networkx as nx
        G = nx.Graph()
        for node in data.get('nodes', []):
            G.add_node(node['id'], **node.get('properties', {}))
        for rel in data.get('relations', []):
            G.add_edge(rel['source'], rel['target'])
        self.graph = G

    def retrieve(self, query: str, k: int | None = None) -> List[dict]:
        """
        图谱检索。

        参数:
            query: 查询文本
            k: 返回数量

        返回:
            list[dict]: 文档列表
        """
        k = k or self.top_k
        docs = []

        if self.communities is None:
            return docs

        # 找到相关社区
        matched = self._find_relevant_communities(query, k=3)
        if not matched:
            return self._keyword_match(query, k=k)

        # 从匹配社区收集文档
        for cid, score in matched:
            community = self.communities.get(str(cid), {})
            nodes = community.get('nodes', [])
            summary = community.get('summary', '')
            categories = community.get('categories', [])

            if nodes:
                for node_id in nodes[:5]:
                    doc = self._node_to_document(node_id, score)
                    if doc:
                        docs.append(doc)
            elif summary:
                cats_str = ', '.join(categories) if categories else ''
                content = f"[社区{cid}] {cats_str}\n{summary}" if cats_str else summary
                docs.append({
                    "id": f"community_{cid}",
                    "content": content,
                    "similarity": round(min(score / 10, 1.0), 4),
                    "metadata": {
                        "community_id": cid,
                        "categories": categories,
                        "score": score,
                    },
                })

            if len(docs) >= k:
                break

        logger.info(f"图谱检索返回 {len(docs)} 个文档")
        return docs[:k]

    @staticmethod
    def _tokenize(text: str) -> set:
        """分词"""
        import re
        tokens = set()
        for m in re.finditer(r'[\u4e00-\u9fff]{2,}', text):
            tokens.add(m.group())
        for m in re.finditer(r'[a-zA-Z]{2,}', text):
            tokens.add(m.group())
        return tokens

    def _find_relevant_communities(self, query: str, k: int = 3) -> List[tuple]:
        """找到最相关的社区"""
        query_lower = query.lower()
        query_words = self._tokenize(query_lower)
        scored = []

        for cid_str, community in (self.communities or {}).items():
            score = 0.0

            # 匹配摘要
            summary = community.get('summary', '').lower()
            if summary:
                for word in query_words:
                    if word in summary:
                        score += 1.0

            # 匹配类别
            categories = community.get('categories', [])
            for cat in categories:
                if cat.lower() in query_lower:
                    score += 2.0
                for word in query_words:
                    if word in cat.lower():
                        score += 0.5

            # 匹配样本通话
            sample_calls = community.get('sample_calls', [])
            for call in sample_calls:
                call_content = call.lower() if isinstance(call, str) else ''
                for word in query_words:
                    if word in call_content:
                        score += 0.3

            if score > 0:
                try:
                    cid = int(cid_str)
                except (ValueError, TypeError):
                    cid = cid_str
                scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _keyword_match(self, query: str, k: int = 5) -> List[dict]:
        """关键词匹配回退"""
        docs = []
        query_lower = query.lower()
        query_words = self._tokenize(query_lower)

        if self.communities is None:
            return docs

        scored = []
        for cid_str, community in (self.communities or {}).items():
            score = 0.0
            summary = community.get('summary', '').lower()
            if summary:
                for word in query_words:
                    if word in summary:
                        score += 1.0

            categories = community.get('categories', [])
            for cat in categories:
                if cat.lower() in query_lower:
                    score += 2.0

            if score > 0:
                cats_str = ', '.join(categories) if categories else ''
                content = f"[社区{cid_str}] {cats_str}\n{summary}" if cats_str else summary
                scored.append(({
                    "id": f"community_{cid_str}",
                    "content": content,
                    "similarity": round(min(score / 10, 1.0), 4),
                    "metadata": {"community_id": cid_str, "categories": categories, "score": score},
                }, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]

    def _node_to_document(self, node_id: str, community_score: float = 0.0) -> Optional[dict]:
        """将图节点转换为文档"""
        if self.graph is None or node_id not in self.graph.nodes:
            return None

        data = self.graph.nodes[node_id]
        content = data.get('text', '')
        if not content:
            return None

        return {
            "id": node_id,
            "content": content,
            "similarity": round(min(community_score / 10, 1.0), 4),
            "metadata": {
                "node_id": node_id,
                "category": data.get('category', ''),
                "community_score": community_score,
            },
        }

    def is_available(self) -> bool:
        """检查图谱检索是否可用"""
        return self.communities is not None
