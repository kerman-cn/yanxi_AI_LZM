"""
双路混合检索模块

BM25 关键词检索 + ChromaDB 向量语义检索 → RRF 融合 → 语义重排序

参考:
- CC 项目: RRF 融合 + 重排序管线
- Terry 项目: ChromaDB 纯语义检索
- LZM 项目: 关键词加权策略
"""
