"""
知识库扩展模块 (KnowledgeExpander)
====================================
扩展 RAG 知识库，新增：
  - 诈骗话术库（常见诈骗模式）
  - 业务对话模板（快递/外卖/银行/医院）
  - 紧急场景知识（家人生病/事故/火灾）
  - 自动学习（通话后自动提取新知识）

使用方式:
    from src.knowledge.knowledge_expander import KnowledgeExpander

    expander = KnowledgeExpander()
    count = expander.expand_all(retriever)
    print(f"新增 {count} 条知识")
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================
# 内置知识库扩展数据
# ============================================================

SCAM_PATTERNS = [
    # 公检法诈骗
    {"text": "自称公安局、检察院、法院，说你涉嫌洗钱、走私等案件，要求配合调查转账到安全账户", "category": "scam", "sub_type": "公检法诈骗", "severity": "high"},
    {"text": "声称你的银行卡被冻结，需要提供验证码或转账验证身份", "category": "scam", "sub_type": "银行卡诈骗", "severity": "high"},
    {"text": "自称社保局，说社保卡异常，要求提供个人信息或转账", "category": "scam", "sub_type": "社保诈骗", "severity": "high"},
    {"text": "说你中奖了，需要先交税或手续费才能领奖", "category": "scam", "sub_type": "中奖诈骗", "severity": "high"},
    {"text": "自称客服，说订单有问题需要退款，要求提供银行卡号和验证码", "category": "scam", "sub_type": "退款诈骗", "severity": "high"},
    {"text": "声称你的贷款已批准，需要先交保证金或服务费", "category": "scam", "sub_type": "贷款诈骗", "severity": "high"},
    {"text": "自称快递公司，说包裹有问题需要提供个人信息或点击链接", "category": "scam", "sub_type": "快递诈骗", "severity": "high"},
    {"text": "说你涉嫌犯罪，要求不要告诉家人，单独处理", "category": "scam", "sub_type": "公检法诈骗", "severity": "high"},
    {"text": "要求你下载远程控制软件或屏幕共享来协助处理问题", "category": "scam", "sub_type": "技术诈骗", "severity": "high"},
    {"text": "自称领导或同事，要求紧急转账到指定账户", "category": "scam", "sub_type": "冒充领导", "severity": "high"},
    {"text": "声称你的微信/支付宝账户异常，需要验证身份信息", "category": "scam", "sub_type": "账户诈骗", "severity": "high"},
    {"text": "说你有一笔退款待领取，需要先支付手续费", "category": "scam", "sub_type": "退款诈骗", "severity": "medium"},
    {"text": "自称投资顾问，推荐高收益理财项目，保证稳赚不赔", "category": "scam", "sub_type": "投资诈骗", "severity": "high"},
    {"text": "说你被起诉了，需要支付调解费或律师费", "category": "scam", "sub_type": "法律诈骗", "severity": "medium"},
    {"text": "要求你到安静的地方接电话，不要被别人听到", "category": "scam", "sub_type": "诈骗特征", "severity": "medium"},
    {"text": "催促你立即做决定，说不马上处理就会产生严重后果", "category": "scam", "sub_type": "诈骗特征", "severity": "medium"},
    {"text": "要求你提供短信验证码，说是验证身份需要", "category": "scam", "sub_type": "诈骗特征", "severity": "high"},
    {"text": "说你的手机号码即将被停机，需要验证身份", "category": "scam", "sub_type": "通信诈骗", "severity": "medium"},
    {"text": "自称税务局，说有退税可以领取，需要提供银行信息", "category": "scam", "sub_type": "税务诈骗", "severity": "medium"},
    {"text": "声称你的信用卡被盗刷，需要验证密码或转账到安全账户", "category": "scam", "sub_type": "信用卡诈骗", "severity": "high"},
]

BUSINESS_TEMPLATES = [
    # 外卖配送
    {"text": "我是美团外卖的骑手，你的外卖到了，请问你在哪里？", "category": "business", "sub_type": "外卖配送"},
    {"text": "饿了么蓝骑士，你的订单已送达，请问放哪里？", "category": "business", "sub_type": "外卖配送"},
    {"text": "你的外卖到了，麻烦下楼取一下", "category": "business", "sub_type": "外卖配送"},
    {"text": "外卖放门口了，记得拿", "category": "business", "sub_type": "外卖配送"},
    # 快递配送
    {"text": "顺丰快递，你的包裹到了，请问在家吗？", "category": "business", "sub_type": "快递配送"},
    {"text": "你的快递到了，放在菜鸟驿站了，取件码是", "category": "business", "sub_type": "快递配送"},
    {"text": "京东快递，你的包裹到了，方便签收吗？", "category": "business", "sub_type": "快递配送"},
    {"text": "中通快递，你的快递放在快递柜了", "category": "business", "sub_type": "快递配送"},
    {"text": "圆通快递，你的包裹到了，请问在家吗？", "category": "business", "sub_type": "快递配送"},
    {"text": "韵达快递，你的快递到了，麻烦取一下", "category": "business", "sub_type": "快递配送"},
    {"text": "申通快递，你的包裹到了，请问放哪里？", "category": "business", "sub_type": "快递配送"},
    {"text": "极兔速递，你的快递到了", "category": "business", "sub_type": "快递配送"},
    # 银行/金融
    {"text": "您好，这里是XX银行，您的信用卡账单已出", "category": "business", "sub_type": "银行通知"},
    {"text": "您的贷款审批已通过，请到网点办理", "category": "business", "sub_type": "银行通知"},
    # 医院
    {"text": "您好，这里是XX医院，您的体检报告已出", "category": "business", "sub_type": "医院通知"},
    {"text": "您的预约挂号已确认，请按时就诊", "category": "business", "sub_type": "医院通知"},
    # 物业
    {"text": "物业通知，明天停水停电，请提前准备", "category": "business", "sub_type": "物业通知"},
    {"text": "您的快递已放在物业前台", "category": "business", "sub_type": "物业通知"},
    # 教育
    {"text": "学校通知，明天家长会，请准时参加", "category": "business", "sub_type": "学校通知"},
    {"text": "您的孩子今天没有到校，请确认", "category": "business", "sub_type": "学校通知"},
]

URGENT_SCENARIOS = [
    {"text": "妈，我出车祸了，快来医院", "category": "urgent", "sub_type": "交通事故"},
    {"text": "家里着火了，快回来", "category": "urgent", "sub_type": "火灾"},
    {"text": "爸突然晕倒了，正在去医院的路上", "category": "urgent", "sub_type": "家人急病"},
    {"text": "孩子发烧到40度，需要马上去医院", "category": "urgent", "sub_type": "家人急病"},
    {"text": "我被人打了，在XX派出所", "category": "urgent", "sub_type": "人身安全"},
    {"text": "家里进贼了，快报警", "category": "urgent", "sub_type": "入室盗窃"},
    {"text": "煤气泄漏了，快回来", "category": "urgent", "sub_type": "安全事故"},
    {"text": "领导，项目出大问题了，需要您马上处理", "category": "urgent", "sub_type": "工作紧急"},
    {"text": "客户要解约，需要您立刻过来", "category": "urgent", "sub_type": "工作紧急"},
    {"text": "手术需要家属签字，请马上来医院", "category": "urgent", "sub_type": "医疗紧急"},
    {"text": "水管爆了，家里被淹了", "category": "urgent", "sub_type": "家庭事故"},
    {"text": "奶奶摔倒了，站不起来", "category": "urgent", "sub_type": "老人意外"},
]


class KnowledgeExpander:
    """知识库扩展器。"""

    def __init__(self):
        self.scam_patterns = SCAM_PATTERNS
        self.business_templates = BUSINESS_TEMPLATES
        self.urgent_scenarios = URGENT_SCENARIOS

    def expand_all(self, retriever=None) -> int:
        """
        将所有扩展知识添加到检索器。

        参数:
            retriever: 知识检索器（需要有 add_documents() 方法）

        返回:
            int: 新增的知识条数
        """
        all_docs = []
        all_docs.extend(self.scam_patterns)
        all_docs.extend(self.business_templates)
        all_docs.extend(self.urgent_scenarios)

        if retriever is not None and hasattr(retriever, "add_documents"):
            try:
                retriever.add_documents(all_docs)
                logger.info(f"知识库扩展完成: 新增 {len(all_docs)} 条知识")
            except Exception as e:
                logger.warning(f"知识库扩展失败: {e}")
                return 0

        return len(all_docs)

    def get_all_documents(self) -> list[dict]:
        """获取所有扩展知识文档。"""
        docs = []
        docs.extend(self.scam_patterns)
        docs.extend(self.business_templates)
        docs.extend(self.urgent_scenarios)
        return docs

    def get_scam_patterns(self) -> list[dict]:
        return self.scam_patterns

    def get_business_templates(self) -> list[dict]:
        return self.business_templates

    def get_urgent_scenarios(self) -> list[dict]:
        return self.urgent_scenarios

    def search_by_category(self, category: str) -> list[dict]:
        """按类别搜索知识。"""
        all_docs = self.get_all_documents()
        return [d for d in all_docs if d.get("category") == category]
