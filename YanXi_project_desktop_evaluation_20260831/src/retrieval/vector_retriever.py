"""
向量语义检索器
==============
第一路检索：使用 BGE-large-zh 进行语义向量检索。
"""

from pathlib import Path
from typing import List

from src.core.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    向量语义检索器。

    使用方式:
        retriever = VectorRetriever(config, embedder)
        docs = retriever.retrieve("我是美团外卖的", top_k=5)
    """

    def __init__(self, config: dict, embedder=None):
        self.config = config
        self.embedder = embedder
        self._collection = None
        self._chroma_client = None

        rag_cfg = config.get("rag", {})
        persist_dir = rag_cfg.get("chroma_persist_dir", "./data/chroma_db")
        self.persist_dir = str(PROJECT_ROOT / persist_dir.lstrip("./"))
        self.collection_name = rag_cfg.get("collection_name", "yanxi_knowledge")
        self.top_k = rag_cfg.get("retrieval_top_k", 5)

    def _init_chroma(self):
        """懒加载 ChromaDB"""
        if self._collection is not None:
            return

        persist_path = Path(self.persist_dir)
        if not persist_path.exists() or not any(persist_path.iterdir()):
            logger.info("ChromaDB 未构建，向量检索暂不可用")
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_collection(self.collection_name)
            logger.info(f"ChromaDB 连接成功 (collection={self.collection_name})")
        except Exception as e:
            logger.warning(f"ChromaDB 连接失败: {e}")
            self._collection = None

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        向量语义检索。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            list[dict]: 文档列表，每个包含 id, content, similarity, metadata
        """
        k = top_k or self.top_k
        self._init_chroma()

        if self._collection is None or self.embedder is None:
            return []

        try:
            query_embedding = self.embedder.encode(query).tolist()

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            docs = []
            ids = results.get("ids")
            documents = results.get("documents")
            metadatas = results.get("metadatas")
            distances = results.get("distances")
            if ids and ids[0]:
                for i, doc_id in enumerate(ids[0]):
                    content = documents[0][i] if documents else ""
                    metadata = metadatas[0][i] if metadatas else {}
                    distance = distances[0][i] if distances else 0.0
                    similarity = round(1.0 / (1.0 + distance), 4)

                    docs.append({
                        "id": doc_id,
                        "content": content,
                        "similarity": similarity,
                        "metadata": metadata or {},
                    })

            return docs
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
            return []

    def add_documents(self, documents: List[dict]) -> None:
        """添加文档到向量库"""
        self._init_chroma()
        if self._collection is None or self.embedder is None:
            return

        try:
            texts = [d.get("content", d.get("text", "")) for d in documents]
            ids = [d.get("id", f"doc_{i}") for i, d in enumerate(documents)]
            metadatas = [d.get("metadata", {}) for d in documents]
            embeddings = self.embedder.encode(texts)

            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"添加 {len(documents)} 条文档到向量库")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")

    def is_available(self) -> bool:
        """检查向量检索是否可用"""
        self._init_chroma()
        return self._collection is not None and self.embedder is not None
