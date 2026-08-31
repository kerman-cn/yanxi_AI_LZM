"""
嵌入模型模块
============
统一使用 BGE-large-zh-v1.5 作为嵌入模型。
"""

import os
from typing import List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    嵌入模型封装。

    使用方式:
        embedder = Embedder(config)
        vec = embedder.encode("你好")
        vecs = embedder.encode(["文本1", "文本2"])
    """

    def __init__(self, config: dict):
        rag_cfg = config.get("rag", {})
        self.model_name = rag_cfg.get(
            "embedding_model", "BAAI/bge-large-zh-v1.5"
        )
        self.device = rag_cfg.get("embedding_device", "cpu")
        self._model = None

        # 设置 HuggingFace 镜像
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    def _load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"嵌入模型加载成功 (维度={dim})")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

    @property
    def model(self):
        self._load_model()
        return self._model

    def encode(self, texts: str | List[str], normalize: bool = True) -> np.ndarray:
        """
        将文本编码为向量。

        参数:
            texts: 单个文本或文本列表
            normalize: 是否归一化

        返回:
            np.ndarray: 向量
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings

    def get_dimension(self) -> int:
        """获取嵌入维度"""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
