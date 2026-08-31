"""
n-gram 精确匹配检索器
=====================
第三路检索：从 LZM 项目整合的 n-gram 倒排索引 + 精确匹配。
"""

import re
from collections import defaultdict
from typing import List, Dict, Set, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 诈骗检测关键词权重表
KEYWORD_WEIGHTS: Dict[str, float] = {
    # 高危词
    "转账": 3.0, "安全账户": 3.0, "保证金": 3.0, "手续费": 2.5,
    "验证码": 3.0, "银行卡号": 3.0, "密码": 2.5, "身份证号": 3.0,
    "通缉令": 3.0, "逮捕": 3.0, "涉嫌犯罪": 3.0, "洗钱": 3.0,
    "公安局": 2.5, "检察院": 2.5, "法院": 2.5, "公检法": 2.5,
    "资金冻结": 2.5, "冻结": 2.0, "社保卡": 2.5, "医保": 2.0,
    "冒充": 2.0, "诈骗": 2.0, "骗": 1.5,
    # 中危词
    "退款": 2.0, "赔偿": 2.0, "中奖": 2.5, "奖金": 2.0,
    "个人所得税": 2.0, "公证费": 2.0, "入会费": 2.0,
    "客服": 1.5, "注销": 2.0, "征信": 2.0, "逾期": 2.0,
    "稳赚": 2.5, "高收益": 2.5, "理财": 1.5, "投资": 1.5,
    "贷款": 2.0, "无抵押": 2.0, "低息": 2.0,
    "刷单": 2.5, "兼职": 1.5, "日赚": 2.5, "垫付": 2.5,
    "杀猪盘": 3.0, "网恋": 2.0, "博彩": 2.5,
    # 正常词（负权重）
    "外卖": -1.5, "快递": -1.5, "包裹": -1.5, "美团": -1.5,
    "饿了么": -1.5, "顺丰": -1.5, "取餐": -1.5, "放门口": -1.0,
    "楼下": -1.0, "门卫": -1.0, "物业": -1.0, "开会": -1.0,
    "吃饭": -1.0, "回家": -1.0,
}


class NgramRetriever:
    """
    n-gram 精确匹配检索器。

    检索策略:
    1. 对 query 和文档做 n-gram（1~3 gram）分词
    2. 构建倒排索引
    3. 计算 Jaccard 相似度 + 关键词加权
    4. 按得分排序返回

    使用方式:
        retriever = NgramRetriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("我是美团外卖的", top_k=5)
    """

    def __init__(self, max_ngram: int = 3):
        self.max_ngram = max_ngram
        self._inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self._documents: List[dict] = []
        self._doc_ngrams: List[Set[str]] = []

    def index_documents(self, documents: List[dict]) -> None:
        """
        构建 n-gram 倒排索引。

        参数:
            documents: 文档列表，每个包含 content/text
        """
        self._documents = documents
        self._inverted_index.clear()
        self._doc_ngrams.clear()

        for idx, doc in enumerate(documents):
            text = doc.get("content", doc.get("text", ""))
            ngrams = self._extract_ngrams(text)
            self._doc_ngrams.append(ngrams)

            for ng in ngrams:
                self._inverted_index[ng].add(idx)

        logger.info(f"n-gram 索引构建完成: {len(documents)} 篇文档, {len(self._inverted_index)} 个 n-gram")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        n-gram 检索。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            list[dict]: 文档列表
        """
        if not self._documents:
            return []

        query_ngrams = self._extract_ngrams(query)
        if not query_ngrams:
            return []

        # 从倒排索引中找到候选文档
        candidate_ids: Set[int] = set()
        for ng in query_ngrams:
            if ng in self._inverted_index:
                candidate_ids.update(self._inverted_index[ng])

        if not candidate_ids:
            return []

        # 计算每个候选文档的得分
        scored = []
        for idx in candidate_ids:
            doc_ngrams = self._doc_ngrams[idx]

            # Jaccard 相似度
            intersection = len(query_ngrams & doc_ngrams)
            union = len(query_ngrams | doc_ngrams)
            jaccard = intersection / union if union > 0 else 0

            # 关键词加权
            weight_score = self._compute_keyword_weight(query, self._documents[idx])

            # 综合得分
            score = jaccard * 0.3 + weight_score * 0.7

            if score > 0:
                scored.append((score, self._documents[idx]))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k]:
            results.append({
                "id": doc.get("id", f"ngram_{id(doc)}"),
                "content": doc.get("content", doc.get("text", "")),
                "similarity": round(min(score, 1.0), 4),
                "metadata": doc.get("metadata", {}),
            })

        return results

    def _extract_ngrams(self, text: str) -> Set[str]:
        """提取 n-gram"""
        ngrams = set()
        # 清理文本
        clean = re.sub(r'\s+', '', text)

        # 1-gram（单字）
        if self.max_ngram >= 1:
            ngrams.update(clean)

        # 2-gram
        if self.max_ngram >= 2:
            for i in range(len(clean) - 1):
                ngrams.add(clean[i:i + 2])

        # 3-gram
        if self.max_ngram >= 3:
            for i in range(len(clean) - 2):
                ngrams.add(clean[i:i + 3])

        # 添加关键词表中的词作为额外 n-gram
        for kw in KEYWORD_WEIGHTS:
            if kw in text:
                ngrams.add(kw)

        return ngrams

    def _compute_keyword_weight(self, query: str, doc: dict) -> float:
        """计算关键词加权得分"""
        doc_text = doc.get("content", doc.get("text", ""))

        weight_score = 0.0
        max_possible = 0.0

        for kw, weight in KEYWORD_WEIGHTS.items():
            max_possible += abs(weight)
            if kw in query and kw in doc_text:
                weight_score += weight
                # 高危词额外加分
                if weight >= 2.5:
                    weight_score += weight * 0.5

        if max_possible > 0:
            normalized = max(0.0, weight_score) / max_possible
        else:
            normalized = 0.0

        return normalized

    def is_available(self) -> bool:
        """检查是否已索引"""
        return len(self._documents) > 0
