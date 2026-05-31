"""
上课时段 AI 电话代接系统 — 多 Agent 协作版（麦克风输入版）
支持：麦克风录音 → 语音识别 → RAG规则匹配 → 风险/意图多Agent分析 → 智能回复 → TTS语音播放
"""
import whisper
import edge_tts
import asyncio
import pygame
import os
import json
import shutil
import re
import hashlib
import tempfile
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from zhipuai import ZhipuAI

# ===================== 全局配置 =====================
WHISPER_MODEL   = "turbo"
VOICE           = "zh-CN-XiaoxiaoNeural"
OUTPUT_AUDIO    = "ai_reply.mp3"

ZHIPU_API_KEY   = "df0ae242b97e4e92beed7bbed62f504e.bOfZ4YoZ3xYVlthi"
LLM_MODEL       = "glm-4-flash"
KNOWLEDGE_FILE  = "./knowledge.txt"
CHROMA_DB_PATH  = "./chroma_call_db"
SIMILARITY_THRESHOLD = 0.90

DEFAULT_LOCATION   = "北门"
SCAM_REPLY         = "你好，你正在进行的是诈骗行为，本次通话已全程录音，我将立即挂断电话并报警。"
MARKETING_REPLY    = "您好，机主不需要相关服务，请勿再次来电。"
NOT_AVAILABLE_REPLY = "您好，机主现在不方便接听电话，请稍后再联系。"
TRANSFER_REPLY     = "您好，您的情况我已记录下来，机主下课后会第一时间给您回电，请您耐心等待。"

# 麦克风录音配置
SAMPLE_RATE   = 44100
RECORD_MAX_SEC = 120
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "call_agent_mic")


# ===================== 数据结构 =====================
@dataclass
class ProcessLog:
    step: str
    content: str

@dataclass
class CallContext:
    call_text: str
    rag_hint: str = ""
    risk_result: dict = field(default_factory=dict)
    business_result: dict = field(default_factory=dict)
    transfer_result: dict = field(default_factory=dict)
    final_reply: str = ""
    logs: list = field(default_factory=list)


# ===================== 模块0：麦克风录音 =====================
class MicRecorder:
    """从电脑麦克风录音，保存为 WAV 文件"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.device_id = None
        self.device_name = ""

    def setup(self):
        """首次选择麦克风设备（只在程序启动时调用一次）"""
        import sounddevice as sd

        print("\n" + "=" * 60)
        print(" 可用音频输入设备：")
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                input_devices.append(i)
                print(f"   [{i}] {d['name']}")
        print("=" * 60)

        if len(input_devices) > 1:
            try:
                choice = input(f"请选择输入设备编号 (默认 {input_devices[0]}): ").strip()
                self.device_id = int(choice) if choice else input_devices[0]
            except (ValueError, IndexError):
                self.device_id = input_devices[0]
        elif len(input_devices) == 0:
            raise RuntimeError("未检测到音频输入设备，请检查麦克风连接")
        else:
            self.device_id = input_devices[0]

        self.device_name = devices[self.device_id]["name"]
        print(f" 已选择：{self.device_name}\n")

    def record(self, output_path: str = None) -> str:
        """按回车开始录音，再按回车停止，返回录音文件路径"""
        import sounddevice as sd
        import soundfile as sf

        if self.device_id is None:
            raise RuntimeError("请先调用 setup() 选择麦克风设备")

        if output_path is None:
            os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
            output_path = os.path.join(TEMP_AUDIO_DIR, "mic_input.wav")

        print(f" 使用设备：{self.device_name}")
        input(f" 按【Enter】开始录音（最长 {RECORD_MAX_SEC} 秒）...")

        # 等一下让用户松开 Enter，避免残留换行符被停止线程读到
        import time, sys
        time.sleep(0.5)
        self._flush_stdin()

        print("\n  正在录音中... 按【Enter】停止录音")
        audio_data = []
        recording = [True]

        def callback(indata, frames, time, status):
            if status:
                print(f"  录音状态异常：{status}")
            if recording[0]:
                audio_data.append(indata.copy())

        import threading
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.device_id,
            callback=callback,
        )
        stream.start()

        stop_event = threading.Event()

        def wait_enter():
            self._flush_stdin()
            input()
            stop_event.set()

        t = threading.Thread(target=wait_enter, daemon=True)
        t.start()

        start_time = time.time()
        max_duration = RECORD_MAX_SEC
        while not stop_event.is_set() and (time.time() - start_time) < max_duration:
            elapsed = time.time() - start_time
            print(f"\r  录制中... {elapsed:.0f} 秒（按 Enter 停止）", end="", flush=True)
            time.sleep(0.1)

        recording[0] = False
        stream.stop()
        stream.close()

        print()  # 换行

        total_samples = sum(len(chunk) for chunk in audio_data)
        duration = total_samples / self.sample_rate

        if total_samples < self.sample_rate * 0.5:
            raise RuntimeError(f"录音太短（{duration:.1f}秒），请重新运行")

        full_audio = __import__('numpy').concatenate(audio_data)
        sf.write(output_path, full_audio, self.sample_rate)

        print(f"  录音完成：{duration:.1f} 秒，已保存至 {output_path}")
        return output_path

    @staticmethod
    def _flush_stdin():
        """清空标准输入缓冲区，防止残留的换行符被 input() 读到"""
        import sys
        if sys.platform == "win32":
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except ImportError:
                pass


# ===================== 模块1：ASR 语音识别 =====================
class ASREngine:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model = None

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"找不到音频文件：{audio_path}")

        print("\n 【1/4】正在识别来电语音内容...")
        if self.model is None:
            print("  正在加载 Whisper 语音识别模型，请稍后…")
            self.model = whisper.load_model(self.model_name)

        result = self.model.transcribe(
            audio_path,
            language="zh",
            initial_prompt="这是一通电话的通话录音，内容是外卖、快递、诈骗推销相关的中文句子。",
            fp16=False,
            temperature=0.0,
        )
        text = result["text"].strip()
        print(f" 识别完成：{text}")
        return text

    def run(self, user_input: str) -> str:
        """统一接口：音频路径 → 识别文本"""
        return self.transcribe(user_input)


# ===================== 模块2：RAG 规则检索引擎（混合模式） =====================
class RAGRetriever:
    def __init__(self, vector_db: Chroma, documents: list = None,
                 threshold: float = SIMILARITY_THRESHOLD):
        self.vector_db = vector_db
        self.threshold = threshold
        self.documents = documents or []
        self._ngram_index = {}
        if self.documents:
            self._build_ngram_index()

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set:
        cleaned = re.sub(r'[^一-鿿0-9a-zA-Z]', '', text)
        if len(cleaned) < n:
            return set()
        return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}

    def _build_ngram_index(self):
        for n in (2, 3, 4, 5):
            for idx, doc in enumerate(self.documents):
                for ng in self._char_ngrams(doc.page_content, n):
                    key = f"{n}_{ng}"
                    if key not in self._ngram_index:
                        self._ngram_index[key] = set()
                    self._ngram_index[key].add(idx)

    def _ngram_match_score(self, call_text: str, doc_text: str) -> float:
        total = 0.0
        for n, weight in ((2, 0.15), (3, 0.25), (4, 0.35), (5, 0.25)):
            c_ng = self._char_ngrams(call_text, n)
            d_ng = self._char_ngrams(doc_text, n)
            if not c_ng or not d_ng:
                continue
            overlap = c_ng & d_ng
            total += (len(overlap) / max(len(c_ng), len(d_ng))) * weight
        return total

    def search(self, call_text: str) -> str:
        print("\n 【2/4】正在 RAG 规则匹配（混合：n-gram + 向量）...")

        candidate_scores = {}
        for n, weight in ((2, 1), (3, 2), (4, 3), (5, 4)):
            for ng in self._char_ngrams(call_text, n):
                key = f"{n}_{ng}"
                if key in self._ngram_index:
                    for doc_idx in self._ngram_index[key]:
                        candidate_scores[doc_idx] = candidate_scores.get(doc_idx, 0) + weight

        ngram_matches = []
        if candidate_scores:
            for doc_idx, _ in sorted(candidate_scores.items(),
                                     key=lambda x: x[1], reverse=True)[:30]:
                doc = self.documents[doc_idx]
                score = self._ngram_match_score(call_text, doc.page_content)
                if score > 0.04:
                    ngram_matches.append({
                        "content": doc.page_content,
                        "type": doc.metadata.get("call_type", "未知"),
                        "rule": doc.metadata.get("handle_rule", ""),
                        "score": score,
                        "source": "ngram",
                    })

        ngram_matches.sort(key=lambda x: x["score"], reverse=True)

        total = self.vector_db._collection.count()
        vec_results = self.vector_db.similarity_search_with_score(
            call_text, k=min(total, 10)
        )
        vector_matches = []
        for doc, score in vec_results:
            if score < self.threshold:
                vector_matches.append({
                    "content": doc.page_content,
                    "type": doc.metadata.get("call_type", "未知"),
                    "rule": doc.metadata.get("handle_rule", ""),
                    "score": 1.0 - score,
                    "source": "vector",
                })

        merged = {}
        for m in ngram_matches:
            merged[m["content"]] = m

        for m in vector_matches:
            key = m["content"]
            if key in merged:
                merged[key]["score"] = merged[key]["score"] * 0.6 + m["score"] * 0.4 + 0.15
                merged[key]["source"] = "ngram+vector"
            else:
                merged[key] = m

        combined = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        if not combined:
            print("  未匹配到有效规则，Agent 将按通用常识处理")
            return ""

        valid = [m for m in combined if m["score"] > 0.08][:5]

        hint = "\n".join(
            f"匹配规则{i+1}：内容={m['content']}，类型={m['type']}，"
            f"处理规则={m['rule']}，匹配度={m['score']:.4f}({m['source']})"
            for i, m in enumerate(valid)
        )
        print(f" ngram候选{len(ngram_matches)}条 + 向量{len(vector_matches)}条"
              f" → 合并得{len(valid)}条，取 Top{len(valid)} 传给 Agent")
        return hint

    def run(self, user_input: str) -> str:
        """统一接口：来电文本 → RAG匹配提示"""
        return self.search(user_input)


# ===================== 模块3：深度场景分析 Agent =====================
class DeepSceneAnalyzer:

    SCENE_PROMPT = """
# 角色：通话场景深度分析专家
## 核心职责
你需要在风险/业务判断之前，先对来电内容做精细化的多维度分析，为后续 Agent 提供更准确的上下文。
不要做风险判断，只做场景分析。

## 分析维度（必须逐项输出）
1. **场景类别**：外卖配送/快递物流/餐饮商家/亲友来电/学校公务/金融机构/政府机关/平台客服/营销推销/未知来电
   - 亲友来电：父母、爷爷奶奶、亲戚、男女朋友、同学、室友、朋友之间的私人对话，即使提到"学校""上课"也是亲友！
   - 学校公务：老师/辅导员/教务处等校方人员的正式通知，不是学生之间的对话
2. **紧急程度**：紧急（涉及人身安全/限时/催促多次）/ 一般（正常通知）/ 不急（非时效性）
3. **话术模式**：正常沟通 / 催促施压 / 恐吓威胁 / 利诱引导 / 含糊不清 / 专业套话
4. **语义锚点**：提取来电中的核心关键词（地点、订单号、机构名、金额、时间节点等）
5. **矛盾信号**：是否存在"内外部矛盾"（如声称是快递但要求转账、声称是客服但说不出订单号）
6. **信息完整度**：来电是否包含足够信息做判断（如外卖需有地点/菜品、快递需有公司/取件方式）

## 重要提示
- 外卖/快递类来电：关键特征是有具体的地点、订单/快递信息，一般不会涉及额外付费或个人信息索取
- 诈骗/推销类来电：常见特征是利诱+链接+付费、恐吓+转账、假冒客服+验证码、中奖+手续费
- 注意识别"先正常后异常"的混合型话术：如一开始正常询问快递，后续引导转账
- 注意识别"冒充类"话术：冒充快递员/外卖员实施的新型诈骗

## 输出 JSON 格式
{
    "scene_category": "场景类别",
    "urgency": "紧急/一般/不急",
    "speech_pattern": "正常沟通/催促施压/恐吓威胁/利诱引导/含糊不清/专业套话",
    "key_info": {"地点": "", "订单号": "", "金额": "", "机构": "", "时间": "", "其他": ""},
    "contradiction_flag": true/false,
    "contradiction_detail": "矛盾说明（无则空）",
    "info_completeness": "完整/一般/不足",
    "analysis_note": "综合分析备注"
}
"""

    def __init__(self, client: ZhipuAI):
        self.client = client

    def analyze(self, call_text: str, rag_hint: str = "") -> dict:
        print("    深度场景分析中...")
        try:
            user_prompt = f"来电内容：{call_text}\n参考规则：{rag_hint or '无'}\n请逐项完成场景分析，输出JSON。"
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.SCENE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content.strip())
            return result
        except Exception as e:
            print(f"     场景分析异常：{e}")
            return {"scene_category": "未知", "urgency": "一般", "speech_pattern": "正常沟通",
                    "contradiction_flag": False, "contradiction_detail": "", "analysis_note": "分析失败，使用兜底"}

    def run(self, user_input: str) -> str:
        """统一接口：JSON(call_text, rag_hint) → JSON场景分析"""
        data = json.loads(user_input)
        result = self.analyze(data.get("call_text", ""), data.get("rag_hint", ""))
        return json.dumps(result, ensure_ascii=False)


# ===================== 模块4：风险防控 Agent（增强版）=====================
class RiskControlAgent:

    RISK_PROMPT = """
# 角色：通话风险防控专家
## 最高优先级铁律（违反则结果无效）

### 铁律1：1级-安全的场景（以下场景必须判1级，不可升级）
- 外卖配送（送达/延迟/地址变更/找不到位置/外卖柜故障）
- 外卖异常处理（菜品售罄换套餐/补差价/餐品洒漏重做退款/拼好饭缺货）
- 快递配送（派送/到付/上门取件/驿站提醒/二次派送/已入柜通知/大件签收）
- 亲友来电（家人/亲戚/同学/朋友/男女朋友的私人通话）
- 只要不要求转账、不索要验证码、不推链接、不推销新产品 → 就是1级

### 铁律2：亲友来电必须判1级（除非有诈骗信号）
- 父母子女、爷爷奶奶、男女朋友、同学朋友、室友等私人关系通话 → 1级-安全
- 朋友借钱、同学借笔记、约打球约吃饭 → 都是1级-安全（正常人际交往！）
- 只有出现"转给我/打钱到XX账户/提供验证码/点击链接"才升级

### 铁律3：推销不是诈骗（除非要密码/验证码/转账到个人账户）
- 房产/保险/课程/信用卡/理财/健身房/装修/汽车促销 → 3级-中风险，不是5级
- 只有索要密码/验证码/要求转账到个人账户的才升级为5级诈骗

### 铁律4：订单异常处理不是营销
- 补差价/餐洒了重做/拼好饭缺货换套餐 → 都是1级-安全，不是3级营销
- 来电方处理已有订单的问题 = 1级；想让机主买新东西 = 3级

---以下为详细规则---

## 最关键区分规则（必须严格遵守）
**正常外卖/快递来电**：来电方是配送员/驿站，告知机主已有订单的配送状态、取件信息。
**营销推销来电**：来电方是商家/品牌，推广产品、促销活动、套餐广告，无论是否与餐饮相关。
**核心判据**：这通电话是"为已有订单服务"还是"想让机主产生新消费"？前者安全，后者至少中风险。

## 营销推销典型话术（全量命中3级-中风险，无例外，不可降级）
- 品牌促销："肯德基疯狂星期四"、"麦当劳优惠"、"奶茶买一送一"、"新品上市"
- 活动推广："限时优惠"、"活动期间"、"打折促销"、"免费领取"
- 会员/积分："积分兑换"、"会员福利"、"免费体验"
- 课程/保险/贷款/房产等商业推广
- 任何含"推荐/推广/促销/优惠/打折/免费领/活动"的话术
- **即使来电方是知名餐饮品牌，只要内容是促销推广而非配送通知，就是3级-中风险**

【重要区分：营销推销 vs 订单异常处理】
- "肯德基疯狂星期四三个汉堡9块9" → 营销，3级-中风险（这是让机主去消费的广告）
- "您点的拼好饭没了，是否更换其他套餐？" → 正常订单异常，1级-安全（已经下单的餐品缺货）
- "您的外卖套餐要不要升级成豪华版？" → 营销，3级-中风险（主动推销升级）
- "您点的宫保鸡丁售罄，换个同价位的可以吗？" → 正常订单异常，1级-安全（已有订单商品缺货）
- "需要补三块钱差价可以吗？" → 正常订单异常，1级-安全（价格调整是订单服务的一部分）
- "餐洒了一半，重做还是退款？" → 正常订单异常，1级-安全（商家处理配送问题）
- 判断标准：来电人是在处理机主已有订单的问题，还是在推销新商品/服务？前者1级，后者3级

【亲友间借钱/求助的正确处理】
- 朋友/同学/亲戚借钱 → 2级-低风险，转人工（这是正常人际交往，不是诈骗！）
- 只有出现明显诈骗信号（要验证码/要求转账到陌生账户/恐吓威胁）才判4级或5级
- 普通借钱、借笔记、帮忙等日常请求 → 2级，转人工让机主自己决定

【推销与诈骗的区分】
- 房产推销、保险推销、课程推销、健身房、装修、汽车促销 → 3级-中风险，拒绝
- 信用卡办理、理财投资 → 3级-中风险，拒绝（除非要求密码/验证码才升级为5级）
- 5级诈骗必须存在：要密码/验证码/转账到个人账户/恐吓/利诱+矛盾信号

## 混合话术识别
1. "先正常后异常"：开场似配送通知，后引导转账/扫码/验证码 → 5级-高危
2. "冒充类"：声称是配送员但无订单信息、要求额外付费 → 4级-高风险
3. "信息矛盾"：身份与信息不匹配 → 4级-高风险

## 风险等级（严格对应，不可跨级）
5级-高危 | 明确诈骗（转账/验证码/冒充公检法/安全账户/中奖付费/银行卡问题）
  → 处理=拦截，话术="你好，你正在进行的是诈骗行为，本次通话已全程录音，我将立即挂断电话并报警。"
4级-高风险 | 高度疑似诈骗（恐吓/利诱+矛盾信号/冒充配送员无订单信息）
  → 处理=拦截，话术="您好，机主不需要相关服务，请勿再次来电，本次通话已做风险标记。"
3级-中风险 | 营销推广/品牌促销/广告推销/任何让机主产生新消费的来电
  → 处理=拒绝，话术="您好，机主不需要相关服务，请勿再次来电。"
  【注意】外卖订单的"菜品售罄换套餐"、"拼好饭没了"属于正常订单异常处理，不是营销，应归为1级
  → 处理=拒绝，话术="您好，机主不需要相关服务，请勿再次来电。"
2级-低风险 | 身份不明/信息不足但无明显风险信号；或亲友间借钱/求助等需机主自行判断的场景
  → 处理=转人工
1级-安全 | 为机主已有订单服务的正常配送/取件/售后通知（与已有订单直接相关）
  → 处理=流转业务处理
  【明确属于1级-安全的场景，不可降级为2级或误判为3级】：
  - 快递破损/丢失询问处理、快递超时未取提醒、到付确认、上门取件预约、二次派送
  - 外卖缺货换套餐、配送延迟、地址变更、补差价、餐品洒漏/破损重做、外卖柜故障
  - 只要来电是为机主已有订单服务的，即使信息不够完整，也归为1级-安全

## 1级-安全必须同时满足（缺一不可）
1. 机主已存在对应订单
2. 来电仅涉及该订单的配送/取件/售后
3. 不涉及任何促销推广、额外付费、信息索取、链接/扫码

## 输出 JSON
{
    "risk_level": "5级-高危/4级-高风险/3级-中风险/2级-低风险/1级-安全",
    "risk_type": "诈骗/冒充诈骗/混合话术诈骗/营销推销/骚扰/无法识别/正常外卖/正常快递/正常亲友/正常公务",
    "handle_suggestion": "拦截/拒绝/转人工/流转业务处理",
    "reply_content": "对应的回复话术（拦截和拒绝场景必填）",
    "risk_score": 0-100,
    "reason": "判断依据，必须说明为何排除其他等级"
}
"""

    def __init__(self, client: ZhipuAI):
        self.client = client

    def assess(self, call_text: str, scene_analysis: dict, rag_hint: str = "") -> dict:
        print("     风险防控评估中...")
        try:
            user_prompt = (
                f"来电内容：{call_text}\n"
                f"深度场景分析：{json.dumps(scene_analysis, ensure_ascii=False)}\n"
                f"RAG规则参考：{rag_hint or '无'}\n"
                f"请结合场景分析结果，做精细化风险判断，输出JSON。"
            )
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.RISK_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content.strip())
            result = self._fix_risk(call_text, result)
            return result
        except Exception as e:
            print(f"     风险判断异常：{e}")
            return {"risk_level": "2级-低风险", "risk_type": "无法识别",
                    "handle_suggestion": "转人工",
                    "reply_content": NOT_AVAILABLE_REPLY, "reason": "系统异常兜底"}

    def run(self, user_input: str) -> str:
        """统一接口：JSON(call_text, scene_analysis, rag_hint) → JSON风险判断"""
        data = json.loads(user_input)
        result = self.assess(
            data.get("call_text", ""),
            data.get("scene_analysis", {}),
            data.get("rag_hint", "")
        )
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _fix_risk(call_text: str, result: dict) -> dict:
        if any(w in call_text for w in ["补差价", "补的钱", "补钱", "补三块", "补两块", "补一块", "补五块"]):
            if result.get("risk_level", "").startswith(("3", "4", "5")):
                result["risk_level"] = "1级-安全"
                result["risk_type"] = "正常外卖"
                result["handle_suggestion"] = "流转业务处理"
                result["reason"] = "[关键词修正] 补差价属于订单异常处理，强制1级-安全"

        if any(w in call_text for w in ["借我", "借你", "借钱", "借点钱", "手头紧"]):
            if result.get("risk_level", "").startswith(("3", "4", "5")):
                result["risk_level"] = "2级-低风险"
                result["risk_type"] = "正常亲友"
                result["handle_suggestion"] = "转人工"
                result["reason"] = "[关键词修正] 亲友借钱属于正常人际交往，强制降为2级"

        return result


# ===================== 模块5：业务处理 Agent（增强版）====================
class BusinessProcessAgent:

    BUSINESS_PROMPT = f"""
# 角色：通话业务代接处理专家
## 最高优先级：模板防混用
- "帮他换套餐"只能用于菜品售罄/缺货！外卖柜故障/确认地址/找不到位置/配送延迟等场景严禁用！
- 回复内容必须对应来电场景，不要乱套模板！

## 核心职责
替正在上课的机主接听电话，生成直接播放给来电方听的语音回复。

## 关键视角规则
- 你是代接者，你在跟来电者通话
- 来电者是配送员/快递员/亲友/老师等——他们是"您"
- 你说的话是直接播放给来电者听的——你就是替机主接电话的人
- 严禁出现"麻烦您取餐""麻烦您取件"——配送员是送餐/送件的人，不是取的人

## 外卖配送（正常送达）
- 话术："您好，机主正在上课不方便接电话，放在{DEFAULT_LOCATION}就行，谢谢。"
- 核心：确认收到，告知放置地点，致谢。不要多余的话。

## 外卖异常（自动决策）
1. **菜品售罄/拼好饭没了** → "您好，机主正在上课不方便接电话，麻烦帮他换一个价格相近的同类套餐，谢谢。"
2. **配送延迟** → "好的收到，会同步给机主，谢谢。"
3. **地址问题/找不到位置** → 给出指引或同意放在来电方建议的位置
4. **补差价/退款/取消** → 同意，告知加在订单里
5. **餐品破损/洒漏** → 同意重做或退款
6. **外卖柜故障** → 同意放在替代位置，如"好的，放在保安亭就行，谢谢。"

## 严禁模板混用
- "帮他换套餐"【仅限】菜品售罄/缺货/拼好饭没了场景！
- 外卖柜故障、确认地址、找不到位置 → 不能用"换套餐"模板！
- 补差价 → "好的，补差价可以，加在订单里就行，谢谢。"
- 餐品洒漏/破损 → "好的，麻烦重做一份/退款就行，谢谢。"

## 快递配送（正常送达/取件提醒）
- 话术："好的收到，机主会尽快去取，谢谢提醒。"
- 送达通知："您好，机主正在上课不方便接电话，放在驿站就行，谢谢。"

## 快递异常（自动决策）
1. **超时未取提醒** → "好的收到，机主会尽快去取，谢谢提醒。"
2. **破损/丢失** → "好的，会同步给机主处理，谢谢告知。"
3. **快递柜/驿站满** → 建议替代位置
4. **货到付款** → "好的，放驿站就行，机主会通过APP支付，谢谢。"
5. **上门取件预约** → "好的，按预约时间上门取件即可，会同步给机主。"
6. **二次派送** → "好的，放驿站/快递柜就行，谢谢。"
7. **大件需本人签收** → "机主正在上课，麻烦改到下午5点后派送或放物业。"
8. **已放入快递柜/丰巢** → "好的收到，谢谢。"（不需要再说放驿站，因为已经放了）
9. **面单地址不清需核对** → 礼貌核对信息

## 亲友/同学来电
- 【核心规则】你就假装是机主本人！不要扮演机主的同学/室友/朋友。你说的话就是机主在说。
  例：来电问"什么时候回来吃饭" → 你说"爸，我正在上课，下课就回家，你们先吃吧"
  例：来电说"赵先生，生日快乐" → 你说"谢谢！我正在上课，下课给您回电话"
  例：来电说"宝贝，妈妈想你了" → 你说"妈，我正上课呢，下课给你打回去"
- 【关键-角色定位】来电者提到的称呼/名字/尊称，是在叫机主（也就是你），判断来电者是谁：
  来电者称呼机主"宝贝/儿子/闺女" → 来电者是父母辈
  来电者称呼机主"赵先生" → 来电者可能是朋友、同事、长辈，需根据语气判断
  来电者自称"妈/爸/奶奶/二姨" → 直接用来电者自称的称呼
- 【关键-祝福/问候的处理】来电者的祝福/问候是给你的（机主），感谢并告知在上课即可
  来电"生日快乐" → 你说"谢谢！我正在上课，下课给您回电话"
  来电"注意身体" → 你说"谢谢关心，我正上课呢，下课聊"
- 【场景判断】根据来电内容选择合适的称呼：
- 来电说"妈做了…" → 称呼"妈"；来电说"爸给你…" → 称呼"爸"
- 来电说"奶奶…" → 称呼"奶奶"；来电说"二姨…" → 称呼"二姨"
- 来电说"兄弟/哥们" → 称呼"兄弟"；来电说"同学" → 称呼"同学"
- 实在判断不了身份 → "您好，我正在上课，下课给您回电话"
- 告知正在上课，下课回电，急事可短信

## 学校/老师来电
- 称呼"老师"，告知机主正在上课，会同步信息

## 输出 JSON
{{
    "business_type": "外卖/快递/亲友来电/公务通知/其他业务",
    "handle_result": "已处理/无法处理",
    "reply_content": "播放给来电方的语音文字（必填，注意你是对来电者说话",
    "extract_info": "提取的关键信息（必填）",
    "reason": "处理依据简述"
}}
"""

    def __init__(self, client: ZhipuAI):
        self.client = client

    def process(self, call_text: str, scene_analysis: dict, rag_hint: str = "") -> dict:
        print("    业务处理中...")
        try:
            user_prompt = (
                f"来电内容：{call_text}\n"
                f"场景分析：{json.dumps(scene_analysis, ensure_ascii=False)}\n"
                f"参考规则：{rag_hint or '无'}\n"
                f"请生成智能代接话术，输出JSON。"
            )
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.BUSINESS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content.strip())
            result["reply_content"] = self._fix_reply(call_text, result["reply_content"])
            return result
        except Exception as e:
            print(f"     业务处理异常：{e}")
            return {"business_type": "其他", "handle_result": "已处理",
                    "reply_content": NOT_AVAILABLE_REPLY, "extract_info": "无",
                    "reason": "系统异常兜底"}

    def run(self, user_input: str) -> str:
        """统一接口：JSON(call_text, scene_analysis, rag_hint) → JSON业务处理"""
        data = json.loads(user_input)
        result = self.process(
            data.get("call_text", ""),
            data.get("scene_analysis", {}),
            data.get("rag_hint", "")
        )
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _fix_reply(call_text: str, reply: str) -> str:
        if any(w in call_text for w in ["外卖柜", "柜子打不开", "柜子坏了", "打不开了", "打不开"]):
            if any(w in reply for w in ["换套餐", "换个", "同类", "价格相近"]):
                loc = "保安亭" if "保安" in call_text else ("前台" if "前台" in call_text else "门口")
                return f"好的，放在{loc}就行，谢谢。"

        if any(w in call_text for w in ["超时", "退回", "快到期", "不取.*退", "没取"]):
            if "催" in reply:
                return "好的收到，机主会尽快去驿站取件，谢谢提醒。"

        if any(w in call_text for w in ["补差价", "补的钱", "补钱", "补.*差价"]):
            if "拒绝" in reply or "不" in reply or "警惕" in reply:
                return "好的，补差价可以，加在订单里就行，谢谢。"

        return reply


# ===================== 模块6：人工转接 Agent =====================
class ManualTransferAgent:

    TRANSFER_PROMPT = """
# 角色：通话转接处理专家
## 职责
当来电无法自动处理时，生成转接话术。注意：你是在跟来电者说话，"转接"的意思是转告机主（机主就是那个"人工"），不是转给什么客服中心。

## 关键规则
- 对来电者说话，告知其消息会被转达给机主
- 严禁说"转接人工客服"、"转接人工服务"——机主就是人工
- 正确说法："已记录您的信息，机主下课后会第一时间给您回电"
- 快递破损等需要机主亲自处理的：告知来电者机主稍后会联系处理

## 输出 JSON
{
    "transfer_flag": true,
    "reply_content": "转接话术（对来电者说的话）",
    "record_info": "来电关键信息摘要",
    "reason": "转接原因"
}
"""

    def __init__(self, client: ZhipuAI):
        self.client = client

    def transfer(self, call_text: str, context: dict = None) -> dict:
        print("    人工转接处理中...")
        try:
            user_prompt = f"来电内容：{call_text}\n前置处理：{json.dumps(context or {}, ensure_ascii=False)}"
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.TRANSFER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"     转接异常：{e}")
            return {"transfer_flag": True, "reply_content": TRANSFER_REPLY,
                    "record_info": call_text, "reason": "系统异常兜底"}

    def run(self, user_input: str) -> str:
        """统一接口：JSON(call_text, context) → JSON转接结果"""
        data = json.loads(user_input)
        result = self.transfer(data.get("call_text", ""), data.get("context"))
        return json.dumps(result, ensure_ascii=False)


# ===================== 模块7：多Agent调度器（增强版）====================
class CallAgentScheduler:
    def __init__(self, scene_analyzer: DeepSceneAnalyzer, risk_agent: RiskControlAgent,
                 business_agent: BusinessProcessAgent, transfer_agent: ManualTransferAgent):
        self.scene_analyzer = scene_analyzer
        self.risk_agent = risk_agent
        self.business_agent = business_agent
        self.transfer_agent = transfer_agent

    def handle(self, call_text: str, rag_hint: str = "") -> CallContext:
        ctx = CallContext(call_text=call_text, rag_hint=rag_hint)

        def log(step, content):
            ctx.logs.append(ProcessLog(step, content))
            print(f" {step}：{content}")

        log("调度启动", f"来电：{call_text}")

        # 统一接口串联：每个模块 run(user_input: str) → str
        scene_input = json.dumps({"call_text": call_text, "rag_hint": rag_hint}, ensure_ascii=False)
        scene_json = self.scene_analyzer.run(scene_input)
        scene = json.loads(scene_json)
        log("场景分析", f"类别={scene.get('scene_category')}, 紧急度={scene.get('urgency')}, 话术={scene.get('speech_pattern')}")

        risk_input = json.dumps({"call_text": call_text, "scene_analysis": scene, "rag_hint": rag_hint}, ensure_ascii=False)
        risk_json = self.risk_agent.run(risk_input)
        risk = json.loads(risk_json)
        ctx.risk_result = risk
        log("风险判断", f"等级={risk.get('risk_level')}, 类型={risk.get('risk_type')}, 建议={risk.get('handle_suggestion')}")

        if risk.get("handle_suggestion") in ("拦截", "拒绝"):
            reply = risk.get("reply_content", "")
            rlevel = risk.get("risk_level", "")
            if rlevel.startswith(("5", "4")):
                if "诈骗" not in reply or "报警" not in reply:
                    reply = SCAM_REPLY
            if rlevel.startswith("3") and not reply:
                reply = MARKETING_REPLY
            ctx.final_reply = reply or SCAM_REPLY
            log("流程终止", "风险拦截，直接输出拦截/拒绝话术")
            return ctx

        if risk.get("handle_suggestion") == "转人工":
            log("流程跳转", "建议转人工")
            transfer_input = json.dumps({"call_text": call_text, "context": {"risk": risk, "scene": scene}}, ensure_ascii=False)
            transfer_json = self.transfer_agent.run(transfer_input)
            transfer = json.loads(transfer_json)
            ctx.transfer_result = transfer
            ctx.final_reply = transfer.get("reply_content", TRANSFER_REPLY)
            log("流程终止", "人工转接完成")
            return ctx

        log("步骤3", "流转至业务处理Agent")
        biz_input = json.dumps({"call_text": call_text, "scene_analysis": scene, "rag_hint": rag_hint}, ensure_ascii=False)
        biz_json = self.business_agent.run(biz_input)
        biz = json.loads(biz_json)
        ctx.business_result = biz
        log("业务结果", f"类型={biz.get('business_type')}, 结果={biz.get('handle_result')}")
        log("提取信息", biz.get("extract_info", "无"))

        if biz.get("handle_result") == "已处理":
            ctx.final_reply = biz.get("reply_content", NOT_AVAILABLE_REPLY)
            log("流程终止", "业务处理完成")
            return ctx

        log("流程跳转", "业务无法处理，转人工")
        transfer_input = json.dumps({"call_text": call_text, "context": {"risk": risk, "scene": scene, "biz": biz}}, ensure_ascii=False)
        transfer_json = self.transfer_agent.run(transfer_input)
        transfer = json.loads(transfer_json)
        ctx.transfer_result = transfer
        ctx.final_reply = transfer.get("reply_content", TRANSFER_REPLY)
        log("流程终止", "人工转接完成")
        return ctx


# ===================== 模块8：TTS 语音合成 =====================
class TTSEngine:
    def __init__(self, voice: str = VOICE, output_path: str = OUTPUT_AUDIO):
        self.voice = voice
        self.output_path = output_path
        try:
            pygame.mixer.init()
        except pygame.error:
            pygame.mixer.quit()
            pygame.mixer.init()

    async def speak(self, text: str) -> str:
        print("\n 【3/4】正在生成 AI 回复语音...")
        # 先释放 pygame 对旧音频文件的占用，否则 Windows 下无法覆盖写入
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate="+0%", volume="+0%")
            await asyncio.wait_for(communicate.save(self.output_path), timeout=30)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"  TTS 语音生成失败（网络原因）：{e}")
            print(f"  将跳过语音，直接显示文字回复。")
            return ""

        try:
            print("  正在播放 AI 回复...")
            pygame.mixer.music.load(self.output_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            print(f" 播放完成，语音已保存：{self.output_path}")
        except Exception as e:
            print(f" 播放失败：{e}，语音文件已保存：{self.output_path}")
        return self.output_path

    def run(self, user_input: str) -> str:
        """统一接口：回复文本 → 音频路径（内部启动事件循环）"""
        return asyncio.run(self.speak(user_input))


# ===================== 系统初始化 =====================
def init_system():
    print("\n" + "=" * 70)
    print(" 正在初始化 AI 电话代接多 Agent 系统...")
    print("=" * 70)

    if not os.path.exists(KNOWLEDGE_FILE):
        raise FileNotFoundError(f"找不到知识库：{KNOWLEDGE_FILE}")

    print(" 初始化智谱 AI 客户端...")
    client = ZhipuAI(api_key=ZHIPU_API_KEY)

    print(" 构建通话规则向量库...")
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    docs = loader.load()
    raw_lines = docs[0].page_content.split("\n")

    document_list = []
    for line in raw_lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        document_list.append(Document(
            page_content=parts[0].strip(),
            metadata={"call_type": parts[1].strip(), "handle_rule": parts[2].strip()},
        ))

    embeddings = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY)

    knowledge_hash = hashlib.md5(docs[0].page_content.encode()).hexdigest()
    hash_file = os.path.join(CHROMA_DB_PATH, ".knowledge_hash")
    need_rebuild = True

    if os.path.exists(CHROMA_DB_PATH) and os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                old_hash = f.read().strip()
            if old_hash == knowledge_hash:
                need_rebuild = False
                print(" 向量库已是最新，直接加载…")
        except Exception:
            pass

    if need_rebuild:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        for i in range(0, len(document_list), 32):
            vector_db.add_documents(document_list[i:i + 32])
        with open(hash_file, "w") as f:
            f.write(knowledge_hash)
        print(f" 向量库构建完成：{len(document_list)} 条规则")
    else:
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print(f" 向量库加载完成：{vector_db._collection.count()} 条规则")

    print(" 初始化多 Agent...")
    asr = ASREngine()
    rag = RAGRetriever(vector_db, documents=document_list)
    tts = TTSEngine()

    scene_agent = DeepSceneAnalyzer(client)
    risk_agent = RiskControlAgent(client)
    biz_agent = BusinessProcessAgent(client)
    transfer_agent = ManualTransferAgent(client)
    scheduler = CallAgentScheduler(scene_agent, risk_agent, biz_agent, transfer_agent)

    print(" 系统初始化完成")
    print("=" * 70 + "\n")
    return asr, rag, scheduler, tts


# ===================== 主程序 =====================
async def main():
    print("\n" + "=" * 60)
    print(" AI 电话代接系统 — 麦克风输入模式（循环测试）")
    print(" 输入 q 退出，按 Enter 继续下一轮录音")
    print("=" * 60)

    # 系统只初始化一次
    asr_obj, rag, scheduler, tts = init_system()
    recorder = MicRecorder()
    recorder.setup()

    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'=' * 60}")
        print(f" 第 {round_num} 轮 — 按 Enter 开始录音，输入 q 退出")
        print(f"{'=' * 60}")
        choice = input().strip()
        if choice.lower() == "q":
            print(" 已退出。")
            break

        try:
            # 录音
            audio_path = recorder.record()

            # 1. 语音识别
            call_text = asr_obj.transcribe(audio_path)

            # 2. RAG 规则匹配
            rag_hint = rag.search(call_text)

            # 3. 多Agent调度处理
            print("\n 【4/4】多 Agent 调度处理中...")
            ctx = scheduler.handle(call_text, rag_hint)

            # 4. TTS 语音合成播放
            await tts.speak(ctx.final_reply)

            # 5. 全流程汇总
            print("\n" + "=" * 70)
            print(" 全流程处理完成！")
            print(f" 来电内容：{call_text}")
            print(f" AI 最终回复：{ctx.final_reply}")
            print(f"\n 执行日志（共 {len(ctx.logs)} 步）：")
            for entry in ctx.logs:
                print(f"   [{entry.step}] {entry.content}")
            print(f"\n 处理详情：")
            if ctx.risk_result:
                print(f"   风险：{json.dumps(ctx.risk_result, ensure_ascii=False, indent=2)}")
            if ctx.business_result:
                print(f"   业务：{json.dumps(ctx.business_result, ensure_ascii=False, indent=2)}")
            if ctx.transfer_result:
                print(f"   转接：{json.dumps(ctx.transfer_result, ensure_ascii=False, indent=2)}")
            print("=" * 70)

        except FileNotFoundError as e:
            print(f"\n {e}")
        except Exception as e:
            print(f"\n 运行出错：{e}")
            import traceback
            traceback.print_exc()
            print(" 继续下一轮...")

    try:
        pygame.mixer.music.unload()
    except:
        pass


if __name__ == "__main__":
    required = ["whisper", "edge_tts", "pygame", "langchain", "langchain-community",
                "langchain-chroma", "zhipuai", "chromadb", "sounddevice", "soundfile", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f" 缺少依赖：{missing}，正在安装...")
        import subprocess
        subprocess.check_call(["pip", "install"] + missing)
        print(" 安装完成，请重新运行程序。")
        exit()

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
