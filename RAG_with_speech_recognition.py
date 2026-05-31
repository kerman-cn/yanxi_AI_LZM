import whisper
import edge_tts
import asyncio
import pygame
import os
import shutil
from typing import Dict, Tuple

# ================== 核心依赖导入 ==================
# 原有RAG依赖
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
# 多Agent大模型依赖
from zhipuai import ZhipuAI

# ================== 全局配置区（全参数可配置，无硬编码）==================
# --- 语音核心配置 ---
WHISPER_MODEL = "small"  # 测试可改为tiny/base，正式用small
VOICE = "zh-CN-XiaoxiaoNeural"
INPUT_CALL_AUDIO = "test.mp3"
OUTPUT_REPLY_AUDIO = "ai_reply.mp3"

# --- 智谱AI核心配置（RAG+多Agent大模型共用）---
ZHIPU_API_KEY = "df0ae242b97e4e92beed7bbed62f504e.bOfZ4YoZ3xYVlthi"
LLM_MODEL_NAME = "glm-4-flash"  # 智谱免费高速模型，无额度压力
KNOWLEDGE_FILE_PATH = "./knowledge.txt"
SIMILARITY_THRESHOLD = 1.0
CHROMA_DB_PATH = "./chroma_call_db"

# --- 基础话术兜底配置 ---
# 仅当语音中无明确地点时，用这个兜底默认地点
DEFAULT_EXPRESS_LOCATION = "北门"
SCAM_REPLY = "您好，机主不需要相关服务，请勿再次来电，本次通话已做风险标记。"
DEFAULT_REPLY = "您好，机主现在暂时不方便接听电话，请稍后再联系。"
TRANSFER_REPLY = "您好，您的需求我无法自动处理，已为您转接人工，机主会在空闲时间尽快给您回电，请您耐心等待。"


# ================== 【模块1：ASR语音识别引擎】==================
class ASREngine:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model = None  # 懒加载模型，避免启动慢

    def run(self, audio_path: str) -> str:
        """
        对外唯一接口：输入音频文件路径，返回识别后的纯文本
        完全保留原有语音识别逻辑和异常处理
        """
        # 音频文件存在性校验
        if not os.path.exists(audio_path):
            current_dir = os.getcwd()
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"❌ 找不到语音输入文件！\n"
                f"📂 目标文件路径：{audio_path}\n"
                f"📍 当前工作文件夹：{current_dir}\n\n"
                f"💡 解决方法：\n"
                f"1. 把录音文件命名为 '{audio_path}'\n"
                f"2. 复制到上述当前工作文件夹中\n"
                f"{'=' * 60}"
            )

        print("🎤 【1/4】正在识别来电语音内容...")
        # 懒加载Whisper模型
        if self.model is None:
            print("   (首次运行正在加载语音识别模型，耗时取决于模型大小，请稍候...)")
            self.model = whisper.load_model(self.model_name)

        try:
            result = self.model.transcribe(
                audio_path,
                language="zh",
                initial_prompt="这是一通电话的通话录音，内容是外卖、快递、诈骗推销相关的普通话句子。",
                fp16=False,
                temperature=0.0
            )
            call_text = result["text"].strip()
            print(f"✅ 语音识别完成，来电内容：{call_text}")
            return call_text
        except Exception as e:
            raise RuntimeError(f"❌ 语音识别失败：{str(e)}，请检查音频文件格式是否正确、FFmpeg是否安装")


# ================== 【模块2：RAG规则检索引擎】==================
class RAGRetriever:
    def __init__(self, vector_db: Chroma, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.vector_db = vector_db
        self.similarity_threshold = similarity_threshold

    def run(self, call_text: str) -> str:
        """
        对外唯一接口：输入来电文本，返回匹配的规则文本
        完全保留原有RAG匹配逻辑和输出格式
        """
        print("\n🔍 【2/4】正在执行RAG规则匹配...")
        total_docs = self.vector_db._collection.count()
        results_with_score = self.vector_db.similarity_search_with_score(
            call_text,
            k=total_docs
        )

        valid_matches = []
        for doc, score in results_with_score:
            if score < self.similarity_threshold:
                valid_matches.append({
                    "content": doc.page_content,
                    "type": doc.metadata["call_type"],
                    "rule": doc.metadata["handle_rule"],
                    "score": score
                })

        if not valid_matches:
            print("⚠️  未匹配到有效规则，Agent将按通用常识处理")
            return ""

        # 拼接匹配到的规则，传给Agent作为参考
        rag_hint = "\n".join([
            f"匹配规则{i + 1}：内容={m['content']}，类型={m['type']}，处理规则={m['rule']}，匹配度={m['score']:.4f}"
            for i, m in enumerate(valid_matches[:3])  # 只取前3条最匹配的，避免token过长
        ])
        print(f"✅ 匹配到{len(valid_matches)}条有效规则，取Top3传给Agent")
        return rag_hint


# ================== 【模块3：多Agent角色定义】==================
# ------------------ 3.1 风险防控Agent ------------------
class RiskControlAgent:
    """
    风险防控Agent：核心职责是来电风险识别与分级
    输入：来电文本、RAG匹配的风险规则
    输出：风险等级、风险类型、处理建议、拦截话术
    """

    def __init__(self, client: ZhipuAI, model_name: str = LLM_MODEL_NAME):
        self.client = client
        self.model_name = model_name
        # 完全保留原有提示词
        self.system_prompt = """
# 角色：通话风险防控专家
## 核心职责
你是专门负责电话来电风险识别的专家，仅针对来电内容做风险判断，不处理具体业务，严格按照规则输出结果，禁止模糊边界。

## 【重要！话术规则】
所有输出的reply_content都是**直接播放给来电方听的**，不是告诉机主的！
- 绝对禁止出现"您收到的信息是诈骗"、"请不要转账"这类对机主说的话
- 绝对禁止出现"我们会立即处理"这类不符合代接身份的话
- 语气要求：
  - 诈骗（高风险）：语气强硬、直接揭穿，不给对方继续说话的机会
  - 营销推销（中风险）：礼貌但坚决拒绝，明确表示不需要
  - 转人工（低风险）：清晰告知机主无法接听，建议稍后再打或短信留言

## 核心定义（必须严格遵守）
### 什么是【正常业务-无风险】（仅以下场景属于无风险）
只有**机主已产生订单、针对该订单进行的一对一沟通**，才属于正常业务，包括：
1.  外卖/快递：已下单的商品配送通知、取件通知、配送异常沟通、取件码告知
2.  亲友/同学：私人关系来电
3.  学校/老师/公务：针对机主的官方通知、事务沟通
4.  其他：机主明确预期的、一对一的非营销类来电

### 什么是【营销推销-中风险】（以下场景全量命中，无例外）
无论内容是否和外卖/快递相关，只要符合以下任意一条，均属于中风险营销推销：
1.  品牌促销、优惠活动、新品推广、套餐广告（如肯德基疯狂星期四、奶茶买一送一等）
2.  无对应订单的餐饮、商品、服务推广
3.  贷款理财、保险推销、课程推广、房产中介等商业广告
4.  非一对一的、广撒网的营销类内容

### 什么是【高风险】
明确的诈骗、电信欺诈、违法违规内容、恶意骚扰，包括但不限于：
1.  冒充公检法要求转账、提供银行卡信息
2.  冒充客服说订单异常要求退款
3.  冒充熟人借钱、要求帮忙转账
4.  中奖诈骗、刷单诈骗、杀猪盘等

## 风险等级与处理建议规则（严格一一对应）
1.  高风险：直接拦截，输出标准诈骗拦截话术，终止流程
2.  中风险：直接拦截，输出标准推销拦截话术，终止流程
3.  低风险：身份不明、需求模糊、无法确认是否为正常业务的来电，建议转人工
4.  无风险：流转至业务处理Agent做后续处理

## 标准话术模板（必须严格遵循，可微调语气但不能改变核心意思）
- 高风险（诈骗）："你好，你正在进行的是诈骗行为，我已经挂断电话并报警处理。"
- 中风险（营销推销）："您好，机主不需要相关服务，请勿再次来电。"
- 低风险（转人工）："您好，机主现在正在上课，不方便接听电话，麻烦您稍后再打或者短信留言，谢谢。"

## 输出要求：必须严格按照JSON格式输出，禁止额外内容，字段如下：
{
    "risk_level": "高风险/中风险/低风险/无风险",
    "risk_type": "诈骗/营销推销/骚扰/正常业务/无法识别",
    "handle_suggestion": "拦截/流转业务处理/转人工",
    "reply_content": "对应的回复话术，拦截和转人工场景必填，其他场景可空",
    "reason": "判断依据，必须简洁明确，严格对照上面的核心定义"
}
        """

    def run(self, call_text: str, rag_rule_hint: str = "") -> Dict:
        """执行风险判断，异常兜底返回默认无风险结果"""
        try:
            user_prompt = f"""
            来电内容：{call_text}
            参考匹配规则：{rag_rule_hint if rag_rule_hint else "无匹配规则，按系统提示的核心定义严格判断"}
            请严格按照系统提示的规则，输出JSON格式的风险判断结果。
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 低温保证判断稳定性，禁止模糊判断
                response_format={"type": "json_object"}
            )
            result = eval(response.choices[0].message.content.strip())
            return result
        except Exception as e:
            print(f"⚠️  风险防控Agent调用异常：{str(e)}，使用兜底逻辑")
            # 异常兜底：无法判断时转人工，避免误放行
            return {
                "risk_level": "低风险",
                "risk_type": "无法识别",
                "handle_suggestion": "转人工",
                "reply_content": "您好，机主现在正在上课，不方便接听电话，麻烦您稍后再打或者短信留言，谢谢。",
                "reason": "系统异常，使用兜底逻辑"
            }


# ------------------ 3.2 业务处理Agent（【核心优化】新增智能异常场景处理）------------------
class BusinessProcessAgent:
    """
    业务处理Agent：核心职责是处理无风险的正常来电业务
    输入：来电文本、RAG匹配的业务规则
    输出：业务类型、处理结果、代接回复话术
    """

    def __init__(self, client: ZhipuAI, model_name: str = LLM_MODEL_NAME,
                 default_location: str = DEFAULT_EXPRESS_LOCATION):
        self.client = client
        self.model_name = model_name
        self.default_location = default_location  # 兜底默认地点
        # 【核心优化】重写提示词，新增外卖/快递异常场景智能处理规则
        self.system_prompt = f"""
# 角色：通话业务代接处理专家
## 核心职责
你是专门负责上课时段电话代接的业务专家，仅处理已确认无风险的正常来电，不做风险判断，严格按照机主预设规则，结合来电的实际内容，生成精准、礼貌、智能的代接话术。
## 核心基础规则
1.  机主当前状态：正在上课，无法接听电话，所有话术必须符合该场景，语气礼貌、简洁、清晰
2.  核心要求：必须先提取来电文本中的所有关键信息，再结合信息生成针对性回复，禁止使用固定不变的话术
3.  关键信息提取要求：
    - 外卖类：必须提取菜品名称、价格、订单变动原因、配送地点、配送员要求等所有关键信息
    - 快递类：必须提取快递公司、快递类型、取件码、存放地点、异常原因、配送员要求等所有关键信息
    - 亲友/公务类：必须提取来电人身份、核心诉求、紧急程度等关键信息
4.  精简规则：禁止完整复述来电方已经明确告知的信息，只需提及核心事件即可，无需重复细节
5.  容错规则：自动修正语音识别的同音字错误（如「一站」修正为「驿站」）

## 【新增】外卖场景智能处理规则（优先级最高）
### 正常配送场景（无异常）
- 话术核心：告知机主正在上课，确认配送地点，告知会同步信息，礼貌收尾
- 示例：来电说「外卖到南门了」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您放在南门就可以了，我会同步给机主，感谢您的配合。

### 外卖异常场景（智能处理）
1.  **菜品售罄/没货**：自动同意更换**价格相近的同类型套餐**，不需要询问机主
    - 示例：来电说「您点的宫保鸡丁没了」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您帮忙更换一份价格相近的同类型套餐就可以了，放在{self.default_location}指定地点，我会同步给机主，感谢您的配合。
2.  **配送延迟**：自动确认延迟，告知会同步信息
    - 示例：来电说「您的外卖要晚20分钟送到」，回复：您好，机主现在正在上课，不方便接听电话，我知道了，会同步给机主，麻烦您尽快送达，感谢您的配合。
3.  **无法送达指定地点**：自动同意放在就近的方便地点
    - 示例：来电说「北门不让进，我在西门等你」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您放在西门就可以了，我会同步给机主，感谢您的配合。
4.  **需要加钱/补差价**：自动同意补差价，告知会同步信息
    - 示例：来电说「您点的饮料涨价了，需要补2块钱」，回复：您好，机主现在正在上课，不方便接听电话，没问题，您直接加在订单里就可以了，放在{self.default_location}指定地点，我会同步给机主，感谢您的配合。
5.  **订单取消/退款**：自动同意取消，告知会同步信息
    - 示例：来电说「您的订单商家取消了，给您退款」，回复：您好，机主现在正在上课，不方便接听电话，我知道了，会同步给机主，感谢您的告知。

## 【新增】快递场景智能处理规则（优先级最高）
### 正常配送场景（无异常）
- 话术核心：告知机主正在上课，确认存放地点，告知会同步信息，礼貌收尾
- 示例：来电说「快递放东门丰巢了，取件码1234」，回复：您好，机主现在正在上课，不方便接听电话，我已收到您的存放信息，会同步给机主，感谢您的配合。

### 快递异常场景（智能处理）
1.  **快递柜满了**：自动建议放在附近的驿站/便利店
    - 示例：来电说「丰巢满了，放不了」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您放在附近的菜鸟驿站就可以了，我会同步给机主，感谢您的配合。
2.  **上门无人**：自动同意次日再送或放在驿站
    - 示例：来电说「家里没人，快递怎么办」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您明天这个时间再送一次，或者放在附近的菜鸟驿站都可以，我会同步给机主，感谢您的配合。
3.  **大件快递需要签收**：自动确认送货时间，建议放在物业
    - 示例：来电说「您有一个大件快递，需要本人签收」，回复：您好，机主现在正在上课，不方便接听电话，麻烦您放在小区物业就可以了，或者下午5点以后再送，我会同步给机主，感谢您的配合。
4.  **快递破损/丢失**：自动告知会同步机主处理
    - 示例：来电说「您的快递有点破损」，回复：您好，机主现在正在上课，不方便接听电话，我知道了，会同步给机主，他会联系您处理的，感谢您的告知。

## 其他场景处理规则
### 场景1：亲友/同学私人关系来电
话术核心：告知机主正在上课，空闲后会尽快回电，可提醒对方有急事可短信留言

### 场景2：学校/老师/公务通知类来电
话术核心：告知机主正在上课，会第一时间同步信息给机主，麻烦对方可将关键信息短信告知，机主空闲后会尽快联系

### 场景3：其他合规正常业务来电
- 能明确处理的，结合场景生成礼貌的代接话术
- 无法明确处理、需求复杂的，标记为「无法处理」，建议转人工

## 输出要求：必须严格按照JSON格式输出，禁止额外内容，字段如下：
    {{
        "business_type": "外卖/快递/亲友来电/公务通知/其他业务",
        "handle_result": "已处理/无法处理",
        "reply_content": "生成的代接回复话术，必填",
        "extract_info": "提取的所有来电关键信息，必填",
        "reason": "处理依据，简洁明确"
    }}
        """

    def run(self, call_text: str, rag_rule_hint: str = "") -> Dict:
        """执行业务处理，异常兜底返回默认回复"""
        try:
            user_prompt = f"""
            来电内容：{call_text}
            参考匹配规则：{rag_rule_hint if rag_rule_hint else "无匹配规则，按系统提示的场景规则处理"}
            请严格按照系统提示的规则，输出JSON格式的业务处理结果。
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # 略微提高温度，支持个性化回复，同时保证稳定性
                response_format={"type": "json_object"}
            )
            result = eval(response.choices[0].message.content.strip())
            return result
        except Exception as e:
            print(f"⚠️  业务处理Agent调用异常：{str(e)}，使用兜底逻辑")
            return {
                "business_type": "其他业务",
                "handle_result": "已处理",
                "reply_content": DEFAULT_REPLY,
                "extract_info": "无",
                "reason": "系统异常，使用兜底默认回复"
            }


# ------------------ 3.3 人工转接Agent ------------------
class ManualTransferAgent:
    """
    人工转接Agent：核心职责是处理无法自动处理的来电，生成转接话术
    输入：来电文本、前置处理结果
    输出：转接标记、转接话术、通话记录信息
    """

    def __init__(self, client: ZhipuAI, model_name: str = LLM_MODEL_NAME):
        self.client = client
        self.model_name = model_name
        # 完全保留原有提示词
        self.system_prompt = """
# 角色：人工转接客服专员
## 核心职责
你是负责电话人工转接的专员，仅针对无法自动处理的来电，生成规范的转接话术，记录通话关键信息，不做其他业务处理。
## 工作规则
1.  转接场景：风险无法判断、业务无法自动处理、来电人明确要求转人工
2.  话术要求：礼貌告知无法自动处理，已转接人工，机主会尽快回电，语气友好不生硬
3.  记录要求：提取来电的核心需求、关键信息，方便机主后续回电查看
4.  输出要求：必须严格按照JSON格式输出，禁止额外内容，字段如下：
    {
        "transfer_flag": true,
        "reply_content": "生成的转接话术，必填",
        "record_info": "提取的来电关键信息，必填",
        "reason": "转接原因，简洁明确"
    }
        """

    def run(self, call_text: str, pre_process_result: Dict) -> Dict:
        """执行人工转接处理，异常兜底返回固定转接话术"""
        try:
            user_prompt = f"""
            来电内容：{call_text}
            前置处理结果：{pre_process_result}
            请严格按照系统提示的规则，输出JSON格式的人工转接处理结果。
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            result = eval(response.choices[0].message.content.strip())
            return result
        except Exception as e:
            print(f"⚠️  人工转接Agent调用异常：{str(e)}，使用兜底逻辑")
            return {
                "transfer_flag": True,
                "reply_content": TRANSFER_REPLY,
                "record_info": f"来电内容：{call_text}",
                "reason": "系统异常，兜底转接人工"
            }


# ================== 【模块4：多Agent调度器】==================
class MultiAgentScheduler:
    """
    多Agent调度器：严格遵循「先风险判断→再业务处理→必要时转人工」的执行顺序
    统一管理三个Agent的调用、状态流转、结果汇总
    """

    def __init__(self, risk_agent: RiskControlAgent, business_agent: BusinessProcessAgent,
                 transfer_agent: ManualTransferAgent):
        self.risk_agent = risk_agent
        self.business_agent = business_agent
        self.transfer_agent = transfer_agent
        # 全流程执行日志，用于调试和结果追溯
        self.process_log = []

    def _add_log(self, step: str, content: str):
        """添加流程日志，统一格式"""
        self.process_log.append(f"【{step}】{content}")
        print(f"📌 {step}：{content}")

    def run(self, call_text: str, rag_match_result: str = "") -> Tuple[str, Dict]:
        """
        核心调度执行入口，严格按顺序执行
        :param call_text: 语音识别后的来电文本
        :param rag_match_result: RAG检索到的匹配规则
        :return: 最终回复话术，全流程处理结果汇总
        """
        self.process_log = []
        self._add_log("调度启动", f"收到来电内容：{call_text}")
        final_result = {}
        final_reply = DEFAULT_REPLY

        # ------------------ 第一步：执行风险防控Agent判断 ------------------
        self._add_log("步骤1", "调用风险防控Agent进行风险识别")
        risk_result = self.risk_agent.run(call_text, rag_match_result)
        final_result["risk_result"] = risk_result
        self._add_log("风险判断结果",
                      f"风险等级：{risk_result['risk_level']}，处理建议：{risk_result['handle_suggestion']}")

        # 高风险直接拦截，终止流程
        if risk_result["handle_suggestion"] == "拦截":
            final_reply = risk_result["reply_content"] if risk_result["reply_content"] else SCAM_REPLY
            self._add_log("流程终止", "高风险拦截，直接输出拦截话术")
            return final_reply, final_result

        # 建议转人工，直接进入转接流程
        if risk_result["handle_suggestion"] == "转人工":
            self._add_log("流程跳转", "风险防控Agent建议转人工，进入人工转接流程")
            transfer_result = self.transfer_agent.run(call_text, {"risk_result": risk_result})
            final_result["transfer_result"] = transfer_result
            final_reply = transfer_result["reply_content"]
            self._add_log("流程终止", "人工转接处理完成")
            return final_reply, final_result

        # ------------------ 第二步：无风险/低风险，执行业务处理Agent ------------------
        if risk_result["handle_suggestion"] == "流转业务处理":
            self._add_log("步骤2", "调用业务处理Agent进行业务代接处理")
            business_result = self.business_agent.run(call_text, rag_match_result)
            final_result["business_result"] = business_result
            self._add_log("业务处理结果",
                          f"业务类型：{business_result['business_type']}，处理结果：{business_result['handle_result']}")
            self._add_log("提取的关键信息", business_result["extract_info"])

            # 业务已处理，输出回复话术
            if business_result["handle_result"] == "已处理":
                final_reply = business_result["reply_content"]
                self._add_log("流程终止", "业务处理完成，输出代接话术")
                return final_reply, final_result

            # 业务无法处理，进入人工转接流程
            if business_result["handle_result"] == "无法处理":
                self._add_log("流程跳转", "业务无法处理，进入人工转接流程")
                transfer_result = self.transfer_agent.run(call_text, {"risk_result": risk_result,
                                                                      "business_result": business_result})
                final_result["transfer_result"] = transfer_result
                final_reply = transfer_result["reply_content"]
                self._add_log("流程终止", "人工转接处理完成")
                return final_reply, final_result

        # 极端兜底：所有流程都未命中，返回默认回复
        self._add_log("兜底处理", "无匹配流程，返回默认回复")
        return final_reply, final_result


# ================== 【模块5：TTS语音合成引擎】==================
class TTSEngine:
    def __init__(self, voice: str = VOICE, output_path: str = OUTPUT_REPLY_AUDIO):
        self.voice = voice
        self.output_path = output_path
        # pygame混音器初始化，增加异常捕获避免重复初始化报错
        try:
            pygame.mixer.init()
        except pygame.error:
            pygame.mixer.quit()
            pygame.mixer.init()

    async def run(self, reply_text: str) -> str:
        """
        对外唯一接口：输入回复文本，生成语音并播放，返回音频文件路径
        完全保留原有语音合成和播放逻辑
        """
        print("\n🔊 【3/4】正在生成AI代接回复语音...")
        try:
            communicate = edge_tts.Communicate(reply_text, self.voice, rate="+0%", volume="+0%")
            await communicate.save(self.output_path)

            print("▶️  正在播放AI代接回复...")
            pygame.mixer.music.load(self.output_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)  # 避免CPU占用过高

            print(f"✅ 回复播放完成，语音文件已保存：{self.output_path}")
            return self.output_path
        except Exception as e:
            raise RuntimeError(f"❌ 语音合成/播放失败：{str(e)}")


# ================== 【系统初始化模块】==================
def init_system() -> Tuple[ASREngine, RAGRetriever, MultiAgentScheduler, TTSEngine]:
    """
    系统全量初始化：构建向量库、初始化所有模块实例
    完全保留原有初始化逻辑和输出格式
    """
    print("\n" + "=" * 80)
    print("📞 正在初始化 上课时段通话代管多Agent AI 系统...")
    print("=" * 80)

    # 1. 知识库文件校验
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        current_dir = os.getcwd()
        raise FileNotFoundError(
            f"\n❌ 找不到知识库文件：{KNOWLEDGE_FILE_PATH}\n   请确保该文件在当前文件夹：{current_dir}"
        )

    # 2. 智谱AI客户端初始化
    try:
        print("🔗 正在初始化智谱AI大模型客户端...")
        zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
        print("✅ 智谱AI客户端初始化成功")
    except Exception as e:
        raise RuntimeError(f"❌ 智谱AI客户端初始化失败，请检查API_KEY是否正确：{str(e)}")

    # 3. 向量库构建
    try:
        print("📖 正在解析通话规则知识库...")
        loader = TextLoader(KNOWLEDGE_FILE_PATH, encoding='utf-8')
        docs = loader.load()
        raw_lines = docs[0].page_content.split('\n')

        valid_rules = []
        document_list = []
        for line in raw_lines:
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            call_content = parts[0].strip()
            call_type = parts[1].strip()
            handle_rule = parts[2].strip()

            valid_rules.append({
                "content": call_content,
                "type": call_type,
                "rule": handle_rule
            })
            document_list.append(
                Document(
                    page_content=call_content,
                    metadata={"call_type": call_type, "handle_rule": handle_rule}
                )
            )

        if len(document_list) == 0:
            raise ValueError("❌ knowledge.txt 中没有有效的通话规则")

        print("🔍 正在构建通话规则向量库...")
        embeddings = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY)

        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )

        # 批量入库，避免单条插入性能问题
        batch_size = 32
        total_count = len(document_list)
        for i in range(0, total_count, batch_size):
            batch = document_list[i: i + batch_size]
            vector_db.add_documents(batch)

        print(f"✅ 向量库构建成功！共加载 {total_count} 条通话规则")
        print(f"   - 外卖/快递规则：{len([r for r in valid_rules if r['type'] == '[外卖/快递]'])} 条")
        print(f"   - 诈骗/推销规则：{len([r for r in valid_rules if r['type'] == '[诈骗/推销]'])} 条")

    except Exception as e:
        raise RuntimeError(f"❌ 向量库构建失败：{str(e)}")

    # 4. 所有模块实例化
    try:
        print("🤖 正在初始化多Agent实例...")
        # 初始化基础模块
        asr_engine = ASREngine(model_name=WHISPER_MODEL)
        rag_retriever = RAGRetriever(vector_db=vector_db, similarity_threshold=SIMILARITY_THRESHOLD)
        tts_engine = TTSEngine(voice=VOICE, output_path=OUTPUT_REPLY_AUDIO)

        # 初始化Agent
        risk_agent = RiskControlAgent(zhipu_client, LLM_MODEL_NAME)
        business_agent = BusinessProcessAgent(zhipu_client, LLM_MODEL_NAME, DEFAULT_EXPRESS_LOCATION)
        transfer_agent = ManualTransferAgent(zhipu_client, LLM_MODEL_NAME)

        # 初始化调度器
        scheduler = MultiAgentScheduler(risk_agent, business_agent, transfer_agent)
        print("✅ 多Agent与调度器初始化成功")
        print("=" * 80 + "\n")

        return asr_engine, rag_retriever, scheduler, tts_engine
    except Exception as e:
        raise RuntimeError(f"❌ 多Agent初始化失败：{str(e)}")


# ================== 【主程序：全链路闭环】==================
async def main():
    asr_engine = None
    tts_engine = None
    zhipu_client = None

    try:
        # 1. 系统全量初始化
        asr_engine, rag_retriever, scheduler, tts_engine = init_system()
        print("📞 多Agent通话代管AI已就绪，开始处理来电...")

        # 2. 语音识别
        call_content = asr_engine.run(INPUT_CALL_AUDIO)
        if not call_content:
            raise ValueError("❌ 语音识别结果为空，请检查音频文件是否有内容")

        # 3. RAG规则匹配
        rag_hint = rag_retriever.run(call_content)

        # 4. 多Agent调度执行核心逻辑
        print("\n🤖 【4/4】正在执行多Agent调度处理...")
        final_reply, process_result = scheduler.run(call_content, rag_hint)

        # 5. 语音合成与播放
        await tts_engine.run(final_reply)

        # 6. 最终结果汇总输出
        print("\n" + "=" * 80)
        print("🎉 【全流程完成】多Agent通话代管执行成功！")
        print(f"📌 最终来电内容：{call_content}")
        print(f"📢 AI最终回复：{final_reply}")
        print("\n📋 全流程执行日志：")
        for log in scheduler.process_log:
            print(f"   {log}")
        print("\n📊 详细处理结果：")
        for key, value in process_result.items():
            print(f"   {key}：{value}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
    finally:
        # 资源释放
        if tts_engine:
            pygame.mixer.music.unload()
        # 注意：zhipuai 2.0+ 客户端没有 _client.close() 方法，这里保留兼容逻辑
        try:
            if 'zhipu_client' in locals() and zhipu_client:
                zhipu_client._client.close()
        except:
            pass


# ================== 程序入口（依赖自动安装+异常兼容）==================
if __name__ == "__main__":
    # 依赖自动检查与安装（完全保留原有逻辑）
    required_packages = [
        "whisper", "edge-tts", "pygame", "langchain",
        "langchain-community", "langchain-chroma", "zhipuai", "chromadb"
    ]
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"💡 检测到缺少依赖包：{missing_packages}，正在自动安装...")
        import subprocess

        try:
            subprocess.check_call(["pip", "install"] + missing_packages)
            print("✅ 依赖安装完成，请重新运行程序！")
        except Exception as e:
            print(f"❌ 依赖安装失败：{str(e)}，请手动执行 pip install {' '.join(missing_packages)}")
        exit()

    # Windows系统asyncio事件循环兼容
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 启动主程序
    asyncio.run(main())