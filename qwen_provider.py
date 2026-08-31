"""通义千问的 OpenAI 兼容客户端与 LangChain Embedding 适配器。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus").strip()
QWEN_EMBEDDING_MODEL = os.getenv(
    "QWEN_EMBEDDING_MODEL", "text-embedding-v4"
).strip()
QWEN_EMBEDDING_DIMENSIONS = int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "1024"))


def require_qwen_api_key() -> str:
    if not QWEN_API_KEY:
        raise RuntimeError(
            "未配置通义千问 API Key。请在项目 .env 中设置 DASHSCOPE_API_KEY。"
        )
    return QWEN_API_KEY


def create_qwen_client(api_key: Optional[str] = None) -> OpenAI:
    """创建百炼北京地域的 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=(api_key or require_qwen_api_key()),
        base_url=QWEN_BASE_URL,
        timeout=60.0,
        max_retries=2,
    )


class QwenEmbeddings(Embeddings):
    """将 text-embedding-v4 适配为 LangChain Embeddings 接口。"""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = QWEN_EMBEDDING_MODEL,
        dimensions: int = QWEN_EMBEDDING_DIMENSIONS,
        batch_size: int = 10,
    ) -> None:
        self.client = client or create_qwen_client()
        self.model = model
        self.dimensions = dimensions
        # text-embedding-v4 的同步接口单批最多 10 行。
        self.batch_size = max(1, min(batch_size, 10))

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
