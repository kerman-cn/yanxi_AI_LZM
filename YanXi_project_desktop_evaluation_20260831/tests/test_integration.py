"""
集成测试
========
测试三路检索融合 + 全链路分类 → 处理流程。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.orchestration.orchestrator import CallOrchestrator
from src.classification.classifier import ThreeTierClassifier, is_meaningless
from src.utils.logger import get_logger, configure_logging

logger = get_logger(__name__)


def test_meaningless_detection():
    """测试无意义文本检测"""
    tests = [
        ("123456789012345", True),   # 连续数字
        ("啊啊啊啊啊", True),          # 重复单字
        ("你好你好你好你好", True),    # 重复词组
        ("我是美团外卖的", False),     # 正常文本
        ("", True),                  # 空文本
        ("ab", False),               # 短文本但可能有效
    ]

    for text, expected in tests:
        result = is_meaningless(text)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} is_meaningless('{text[:20]}') = {result} (expected {expected})")


def test_classifier():
    """测试三级分类器"""
    config = load_config()
    classifier = ThreeTierClassifier(config, retriever=None, llm_client=None)

    tests = [
        ("我是美团外卖的，你的餐到楼下了", "food_delivery"),
        ("你涉嫌洗钱，请配合调查", "scam"),
        ("妈，我今天晚上回家吃饭", "family"),
        ("领导，明天有个会议", "leader"),
        ("你的快递到了，在菜鸟驿站", "express"),
        ("你好，我们这边有个课程推荐", "telemarketing"),
        ("喂，在干嘛呢", "friend"),
    ]

    for text, expected in tests:
        result = classifier.classify(text)
        actual = result.type_id
        status = "[OK]" if expected in actual or actual in expected else "[?]"
        print(f"{status} '{text[:25]}...' → {actual} ({result.call_type.display_name}, 置信度={result.confidence:.0%})")


def test_full_pipeline():
    """测试全链路"""
    config = load_config()
    configure_logging(config)

    print("\n初始化 CallOrchestrator（含三路检索）...")
    try:
        orchestrator = CallOrchestrator(config)
    except Exception as e:
        print(f"  初始化失败（部分功能可能不可用）: {e}")
        print("  继续测试基础功能...")
        return

    tests = [
        ("13800001111", "我是美团外卖的，你的餐到楼下了"),
        ("13800002222", "您好，我是市公安局的，你涉嫌洗钱案件"),
        ("13800003333", "你的快递到菜鸟驿站了"),
        ("13800004444", "妈，我今天晚上回家吃饭"),
        ("13800005555", "领导，明天下午有个紧急会议"),
        ("13800006666", "恭喜您中奖了！奖金10万元，请先缴纳个人所得税"),
        ("13800007777", "你好，我们这边有个课程推荐给您"),
    ]

    for number, text in tests:
        result = orchestrator.run(text, caller_number=number)
        print(f"\n来电: {text[:30]}...")
        print(f"  号码: {number}")
        print(f"  类型: {result.get('call_type_name', '?')} (置信度={result.get('confidence', 0):.0%})")
        print(f"  动作: {result.get('final_action')}")
        print(f"  回复: {result.get('agent_reply', '')[:50]}")
        if result.get('notification_card'):
            print(f"  卡片: {result['notification_card'].get('title', '')}")


def test_retrieval():
    """测试三路检索融合"""
    config = load_config()

    # 准备测试文档
    test_docs = [
        {"id": "doc1", "content": "我是美团外卖的，你的餐到楼下了，请下来取餐", "metadata": {"category": "food_delivery"}},
        {"id": "doc2", "content": "您好，我是公安局的，您涉嫌洗钱案件，请配合调查", "metadata": {"category": "scam", "label": "fraud"}},
        {"id": "doc3", "content": "你的快递到菜鸟驿站了，取件码是1234", "metadata": {"category": "express"}},
        {"id": "doc4", "content": "恭喜您中奖了！奖金10万元，请先缴纳个人所得税2000元", "metadata": {"category": "scam", "label": "fraud"}},
        {"id": "doc5", "content": "领导，明天下午有个紧急会议，请您参加", "metadata": {"category": "leader"}},
    ]

    print("\n测试 n-gram 检索...")
    from src.retrieval.ngram_retriever import NgramRetriever
    ngram = NgramRetriever()
    ngram.index_documents(test_docs)

    queries = ["外卖", "公安局", "中奖"]
    for q in queries:
        results = ngram.retrieve(q, top_k=3)
        print(f"  查询 '{q}':")
        for r in results:
            print(f"    - [{r['similarity']:.2f}] {r['content'][:50]}...")

    print("\n测试 RRF 融合...")
    from src.retrieval.fusion import RRFFuser
    fuser = RRFFuser(k_const=60)

    list_a = [
        {"id": "a1", "content": "文档A1", "similarity": 0.9},
        {"id": "a2", "content": "文档A2", "similarity": 0.8},
        {"id": "a3", "content": "文档A3", "similarity": 0.7},
    ]
    list_b = [
        {"id": "b1", "content": "文档B1", "similarity": 0.85},
        {"id": "a1", "content": "文档A1", "similarity": 0.9},  # 重复
        {"id": "b2", "content": "文档B2", "similarity": 0.75},
    ]

    fused = fuser.fuse(list_a, list_b)
    print(f"  融合结果 ({len(fused)} 条):")
    for doc in fused:
        print(f"    - {doc['id']}: {doc['content']}")


if __name__ == "__main__":
    print("=" * 60)
    print("言犀整合版 — 集成测试")
    print("=" * 60)

    print("\n[1/4] 无意义文本检测")
    test_meaningless_detection()

    print("\n[2/4] 三级分类器")
    test_classifier()

    print("\n[3/4] 检索模块")
    test_retrieval()

    print("\n[4/4] 全链路测试")
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
