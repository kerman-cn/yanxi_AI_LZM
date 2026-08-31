"""通义千问适配层的无网络测试。"""

import unittest
from types import SimpleNamespace

from qwen_provider import QwenEmbeddings


class _FakeEmbeddingEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
            for index, text in reversed(list(enumerate(kwargs["input"])))
        ]
        return SimpleNamespace(data=data)


class QwenProviderTests(unittest.TestCase):
    def test_embedding_batches_at_official_limit_and_preserves_order(self):
        endpoint = _FakeEmbeddingEndpoint()
        client = SimpleNamespace(embeddings=endpoint)
        embeddings = QwenEmbeddings(
            client=client, model="text-embedding-v4", dimensions=1024, batch_size=99
        )

        vectors = embeddings.embed_documents([f"文本{i}" for i in range(23)])

        self.assertEqual([len(call["input"]) for call in endpoint.calls], [10, 10, 3])
        self.assertEqual(len(vectors), 23)
        self.assertEqual(vectors[0][0], 0.0)
        self.assertEqual(vectors[10][0], 0.0)
        self.assertEqual(endpoint.calls[0]["dimensions"], 1024)

    def test_query_uses_single_embedding(self):
        endpoint = _FakeEmbeddingEndpoint()
        embeddings = QwenEmbeddings(
            client=SimpleNamespace(embeddings=endpoint), dimensions=256
        )
        vector = embeddings.embed_query("你好")
        self.assertEqual(vector, [0.0, 2.0])
        self.assertEqual(endpoint.calls[0]["input"], ["你好"])


if __name__ == "__main__":
    unittest.main()
