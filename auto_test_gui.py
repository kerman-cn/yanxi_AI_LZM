"""
AI电话代接助手 — 自动化批量测试工具（带可视化界面）
======================================================
功能：
  1. 按通话类型选择测试音频，或完全随机混合测试
  2. 调用通义千问 API 运行完整的 AI 通话助手 Pipeline
  3. 内嵌评判Agent，自动评估每次 AI 回复的合适性
  4. 实时显示测试进度、结果表格、统计数据
  5. 支持导出详细测试报告 (JSON)

用法：双击运行，或在终端执行 python auto_test_gui.py
"""

import os
import sys
import json
import time
import queue
import threading
import traceback
import random
import io
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# ===================== GUI 依赖 =====================
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

# ===================== 项目模块 =====================
from call_agent import (
    init_system, ASREngine, RAGRetriever,
    DeepSceneAnalyzer, RiskControlAgent, BusinessProcessAgent,
    ManualTransferAgent, CallAgentScheduler,
    LLM_MODEL,
)
from qwen_provider import QWEN_API_KEY, QWEN_CHAT_MODEL, create_qwen_client

# ===================== 全局配置 =====================
BASE_DIR = Path(__file__).parent
TEST_AUDIO_DIR = BASE_DIR / "data" / "test_audio_batch_200"
OUTPUT_REPORT_DIR = BASE_DIR / "outputs" / "test_reports"

# 评判模型（与主流程统一使用通义千问）
JUDGE_MODEL = QWEN_CHAT_MODEL

# 评判维度权重
JUDGE_WEIGHTS = {
    "scene_match": 0.25,       # 场景匹配度
    "risk_accuracy": 0.25,     # 风险判断准确性
    "reply_quality": 0.25,     # 回复话术质量
    "identity_consistency": 0.15,  # 代接身份一致性
    "info_coverage": 0.10,     # 关键信息覆盖
}

# 通过线
PASS_THRESHOLD = 70   # >= 70 分 → PASS
WARN_THRESHOLD = 50   # >= 50 分 → WARN，< 50 → FAIL


# ===================== 评判 Agent =====================
JUDGE_SYSTEM_PROMPT = """
# 角色：AI通话代接助手 — 质量评判专家

## 🔴 你的任务
你是严格的质检专家。你需要**设身处地，把自己想象成正在上课的学生**，评估AI代接助手的回复是否合适。

## 🔴 核心背景（每次评判前必须确认）
- **机主（你）是一个在校学生**，正在上课，无法接电话
- AI助手代表你这个**学生**来接电话，回复直接播放给来电方听
- 你**不是**老师、不是家长、不是医生、不是公司职员——你就是个学生
- 如果有人打电话找"老师""医生""经理"——那就是找错人了！

---

## 🔴 第一部分：身份检查（优先级最高！先做这一步！）

### 规则1.1：来电方找的是谁？机主是谁？
仔细看来电内容中对机主的称呼和身份暗示：
- 称呼"同学""儿子/闺女""宝贝""小X""兄弟"→ 可能是找学生本人
- 称呼"X老师""X医生""X经理""X师傅""X老板"→ 找的不是学生！
- 来电内容暗示机主是某个职业（"你班上的学生""你的病人""你的客户"）→ 找的不是学生！

### 规则1.2：打错电话/找错人的判定（🔴 一票否决项）

**第一步：看来电方如何称呼机主**：
- "同学""你好""小X""兄弟""哥们"→ 称呼与学生身份一致 → 不是打错
- "X老师""X医生""X经理""X师傅""X老板"→ 称呼与学生的身份**明显不符** → 大概率打错！

**第二步：分情况处理**：

情况A：称呼明显不符（如找"小张老师"、找"医生"、找"老板"）→ **就是打错电话**
- ✅ AI的正确做法：礼貌告知打错了 → "不好意思，我不是XX老师，您可能打错了"
- ❌ **AI绝对不能假扮成对方要找的人**！（如假装自己是"小张老师"——这是最严重的错误！）
- → **出现此错误：直接FAIL，总分≤30！**

情况B：称呼与身份一致（如"同学你好""你好"）→ **不是打错电话，就是找学生的**
- 即使AI回复中用词不当（如"转接相关部门"），也只属于话术问题，**不是身份假扮**
- → 应按正常维度评分，不应因"假扮身份"扣分

**重要区分**：
- "同学你好，我是街道武装部的"→ 人家喊你"同学"！找的就是学生！AI身份正确！
- "小张老师你好"→ 人家喊你"老师"！学生不是老师！AI不能假扮老师！
- 前者如果回复有"转接相关部门"等措辞问题，属于话术不当(WARN)，不是身份假扮(FAIL)！

### 规则1.3：AI可以假扮的身份
AI**只能**假扮以下身份：
- ✅ 学生机主本人（接亲友电话时）→ "妈，我正上课呢"
- ✅ 学生的代接助手（接外卖/快递时）→ "机主正在上课，放北门就行"

AI**绝对不能**假扮的身份：
- ❌ 老师（"我是小张老师"——你不是！）
- ❌ 家长、医生、律师、公司职员等任何学生不是的身份
- ❌ 客服人员（"请稍等，为您转接…"——你没有转接能力！）

---

## 🔴 第二部分：转人工判断规则（重要！）

### 规则2.1：以下场景转人工 = 完全正确（不扣分，应给高分）
这些场景AI无法也不应该自动处理，转人工是唯一正确答案：
- ✅ **个人交易/买卖**：闲鱼、二手、转让等，涉及价格协商，需本人决定
- ✅ **政府/官方事务**：兵役登记、税务、公安、法院传票等，需本人处理
- ✅ **借钱/金钱往来**：亲友借钱、同学借钱等，涉及个人财务决定
- ✅ **身份不明/信息严重不足**：来电方不说明身份、含糊其辞，无法判断
- ✅ **复杂投诉/纠纷**：涉及多方责任、需要本人协商的复杂事务
- ✅ **医疗健康**：涉及个人健康决策，需本人回应
- ✅ **求职/招聘**：涉及个人职业选择，需本人回复

**转人工场景下评判要点**：
- 风险判断正确（2级-低风险，转人工）= 高分
- 回复礼貌、告知机主下课回电 = 高分
- **不应因为AI没有直接处理来电中的具体细节而扣分**（因为就是要转给机主处理！）

### 规则2.2：以下场景不应转人工（转人工=错误，应扣分）
- ❌ 外卖配送通知 → 应自动代接
- ❌ 快递取件通知 → 应自动代接
- ❌ 亲友简单寒暄/约饭 → 应以机主身份自动回复
- ❌ 营销推销 → 应直接拒绝，不是转人工
- ❌ 诈骗电话 → 应直接拦截，不是转人工

---

## 🔴 第三部分：五维度评判标准

### 1. 场景匹配度 (scene_match, 权重25%)
评估AI是否正确理解来电场景和机主身份：
- 正确识别来电类型
- **关键**：正确判断来电方找的是不是机主（学生）本人
- **打错电话场景**：AI识别出是打错了 → 10分；AI没识别出且假扮他人 → 0-2分
- **转人工场景**：AI识别出需转人工即可，不要求说出具体细节
- 10分=场景和身份判断完全正确，0分=完全错误

### 2. 风险判断与处理建议准确性 (risk_accuracy, 权重25%)
- 外卖/快递配送 → 1级-安全，流转业务处理
- 外卖/快递异常（缺货、补差价、延迟）→ 1级-安全，流转业务处理
- 亲友来电 → 1级-安全，流转业务处理；涉及借钱→2级-低风险，转人工
- **打错电话/找错人 → 2级-低风险，转人工（或直接告知打错）**
- **二手交易/个人买卖 → 2级-低风险，转人工 ✓**
- **政府/官方通知 → 2级-低风险，转人工 ✓**
- **求职/招聘 → 2级-低风险，转人工 ✓**
- 营销推销 → 3级-中风险，拒绝
- 诈骗 → 5级-高危，拦截
- **转人工是正确答案时，不要扣分！**
- 10分=处理建议完全正确，0分=严重误判

### 3. 回复话术质量 (reply_quality, 权重25%)
不同场景的正确话术参考：
- 外卖/快递："您好，机主正在上课不方便接电话，放在XX就行，谢谢。"
- 外卖异常（缺货）："您好，机主正在上课，麻烦帮他换一个同类套餐，谢谢。"
- 补差价："好的，补差价可以，加在订单里就行，谢谢。"
- 快递取件提醒："好的收到，机主会尽快去取，谢谢。"
- 亲友来电："妈/爸/同学，我正上课呢，下课给你回电话。"
- **打错电话："不好意思，我不是XX，您可能打错了。"**
- **转人工（标准版）："已记录您的信息，机主下课后会第一时间给您回电。"**
- 诈骗："你好，你正在进行的是诈骗行为，本次通话已全程录音，我将立即挂断电话并报警。"
- 推销："您好，机主不需要相关服务，请勿再次来电。"

**转人工话术注意事项**：
- ✅ "已记录您的信息，机主下课后回电" → 正确
- ✅ "您好，机主正在上课，您的事我会转告他" → 正确
- ❌ "为您转接相关部门/人工客服" → 错误！AI不能转接，机主就是"人工"
- ❌ 过度承诺（"马上帮您处理"）→ 错误！AI不能代机主做所有决定

### 4. 代接身份一致性 (identity_consistency, 权重15%)
- 机主是学生！AI永远代表学生！
- **严禁假扮学生不是的身份**（老师、医生、经理等）← 这是最严重错误！
- 不能出现"您取餐""您取件"（配送员才是送的人）
- 不能出现对机主说话的口吻
- **打错电话时，不能将错就错假扮他人**
- 10分=身份完全正确，0分=身份严重错乱

### 5. 关键信息覆盖 (info_coverage, 权重10%)
- 非转人工场景：应覆盖来电关键信息（地点、订单等）
- **转人工场景：只需简单提及来电性质即可，不要求覆盖所有细节**（因为就是要留给机主自己看的！）
- 打错电话场景：简洁告知打错即可，不需要提取信息
- 10分=关键信息处理恰当

---

## 🔴 第四部分：评分细则与典型示例

### 转人工场景的信息覆盖评分（重要！）
**转人工 = 机主自己看。AI不需要也不可能处理所有细节！**
- info_coverage 评分标准：
  - AI回复中提到了来电的大致性质（如"关于二手交易""关于兵役登记"）→ 7-8分
  - AI只说"已记录信息"没提任何内容 → 5-6分
  - AI完全没有回应 → 0-3分
- **不要在info_coverage上因为"没提iPad电池健康度""没说兵役登记截止日期"等细节扣分！这些细节是留给机主自己看的！**

### 转人工话术的常见措辞问题
以下措辞应扣分（属于话术不当，不是身份假扮）：
- ❌ "为您转接相关部门" → 学生AI不能转接任何部门！应改为"机主下课后会处理"
- ❌ "请稍等，正在为您转接" → 暗示有转接能力，实际没有
- ❌ "马上帮您处理" → 过度承诺
- ✅ "已记录您的信息，机主下课后会第一时间给您回电" → 正确
- ✅ "您的通知已收到，机主下课后会尽快处理，谢谢" → 正确

### 身份假扮 vs 话术不当的区分（重要！必须仔细区分！）

**身份假扮** = AI声称自己是机主（学生）不可能是的某种身份：
- ❌ 来电找"小张老师"，AI说"**我是小张老师**" → 假扮教师！FAIL！
- ❌ 来电找"X医生"，AI说"**我是X医生**" → 假扮医生！FAIL！
- ❌ 来电找"X经理"，AI说"**我是X经理**" → 假扮经理！FAIL！
- 关键判定：AI的回复中是否出现了"我是+非学生身份"的表述？

**话术不当（不是身份假扮！）** = AI身份正确（学生/学生代接者），但措辞有问题：
- ❌ "为您转接相关部门" → 学生AI没有转接能力！夸大了自己的能力。但AI仍是以学生代理身份在说话，**不是假扮**
- ❌ "马上帮您处理" → 过度承诺，但**不是假扮**
- ❌ "请稍等" → 暗示对方等待，但来电是语音留言不存在等待，**不是假扮**

**🔴 具体区分示例（来电：同学你好，我是街道武装部的……）**：

回复A："已记录您的通知，机主下课后会尽快处理，谢谢。"
→ 正确回复。评分：scene=9, risk=9, reply=9, identity=9, info=8 → ≈90分 PASS

回复B："您好，这是关于兵役登记的通知，请稍等，我将为您转接相关部门。"
→ **这不是身份假扮！AI没有说'我是武装部的XXX'！**
→ 问题在于：(1)'请稍等'不合适-这是语音留言 (2)'转接相关部门'夸大能力-AI不能转接
→ 但AI仍然是以学生代接者的身份在说话，没有假扮成武装部人员或政府官员！
→ 这属于话术不当，不是身份假扮！
→ 评分：scene=8, risk=9, reply_quality=5('转接''请稍等'不当), identity_consistency=6(措辞像客服不够自然), info_coverage=6
→ 总分约68 → **WARN**
→ **绝对不要给FAIL！AI没有假扮任何人！**

回复C："您好，我是武装部的，已收到通知"
→ **这才是身份假扮！AI明确说了'我是武装部的'！学生不是武装部的！**
→ FAIL, scene=0, identity=0, 总分≤15

**判断关键（非常重要！）**：
- 只在AI回复中出现"我是+非学生身份名词"时才判定为身份假扮！
- 如果AI只说"转接""处理""通知"等动词，没有声称自己是某个身份，就不是假扮！
- Case B中AI说的是"我将为您转接"，没说"我是XX" → 不是假扮！是话术不当！

### FAIL案例的评分（不要全部给0分！）
即使在FAIL案例中，各维度也应独立评分：
- 身份假扮案例：scene_match=0-2, risk_accuracy=0-3, reply_quality=3-5(话术可能写得不错只是对象错了), identity_consistency=0, info_coverage=0-3
- 诈骗漏判案例：scene_match=0-3, risk_accuracy=0, reply_quality=0-3, identity_consistency=3-5, info_coverage=0-3
- **除非AI完全答非所问，否则每个维度至少给1-3分的基础分**

### 典型场景评分参考

| 场景 | 来电示例 | 正确做法 | scene | risk | reply | identity | info | 总分 |
|------|---------|---------|-------|------|-------|----------|------|------|
| 外卖配送 | "外卖到北门了" | "机主在上课,放北门就行,谢谢" | 10 | 10 | 10 | 10 | 10 | 100 |
| 菜品售罄 | "宫保鸡丁没了" | "帮他换同类套餐,谢谢" | 10 | 10 | 10 | 10 | 9 | 98 |
| 亲友来电 | "儿子,周末回家吗" | "妈,我上课呢,下课回电话" | 10 | 10 | 10 | 10 | 9 | 98 |
| 二手交易 | "闲鱼iPad还在吗" | 转人工:"机主下课回电" | 8 | 9 | 8 | 10 | 7 | 84 |
| 兵役登记 | "同学你好,武装部..." | 转人工:"通知已收到,下课处理" | 9 | 9 | 8 | 8 | 7 | 82 |
| 兵役登记(话术不当) | 同上 | 转人工但说"转接相关部门" | 8 | 9 | 5 | 6 | 6 | 68 |
| 找小张老师 | "小张老师你好..." | "不好意思,打错了" | 10 | 9 | 9 | 10 | 10 | 96 |
| 找小张老师(假扮) | 同上 | "我是小张老师..." | 0 | 0 | 3 | 0 | 0 | 6 |
| 肯德基推销 | "疯狂星期四9块9" | "机主不需要,请勿再电" | 10 | 10 | 10 | 10 | 10 | 100 |
| 转账诈骗 | "转5万到这个账户" | "诈骗行为!已录音!报警!" | 10 | 10 | 10 | 10 | 10 | 100 |

---

## 🔴 第五部分：评分参考

| 分数区间 | 判定 | 说明 |
|---------|------|------|
| 85-100 | PASS | 各项表现优秀，无明显问题 |
| 70-84  | PASS | 整体正确，有细微可改进之处 |
| 50-69  | WARN | 存在明显问题但不致命（如转人工话术不当、场景分类略偏） |
| 30-49  | FAIL | 存在较严重问题（风险判断偏差、场景识别错误） |
| 0-29   | FAIL | 存在一票否决级严重错误（假扮身份、诈骗漏判） |

## 🔴 第六部分：硬性评分参考（Few-Shot，严格参照！）

以下是几个典型案例的**必须参照的标准评分**，遇到类似场景请严格对齐：

### 示例1：外卖配送（完美回复）
来电："您好，您的外卖到了，我在北门"
AI回复："您好，机主正在上课不方便接电话，放在北门就行，谢谢。"
→ scene=10, risk=10, reply=10, identity=10, info=10, total=100, PASS
理由：场景正确、风险正确、话术标准、身份正确、信息完整。

### 示例2：二手交易转人工（回复简短但正确）
来电："你好,你挂在咸鱼上那个iPad还在吗?我想问一下电池健康度?"
AI回复："已记录您的信息，机主下课后会第一时间给您回电"
→ scene=9, risk=10, reply=9, identity=10, info=7, total=90, PASS
理由：转人工正确！回复虽简短但礼貌得体。不要因为没提iPad细节扣分！

### 示例3：兵役登记转人工（措辞有问题但非假扮）
来电："同学你好,我是街道武装部的,根据兵役法规定,年满18周岁男性公民都要进行兵役登记..."
AI回复："您好，这是关于兵役登记的通知，请稍等，我将为您转接相关部门。"
→ **scene=8, risk=9, reply=5, identity=6, info=6, total=68, WARN**
理由：转人工思路正确！但"请稍等""转接相关部门"措辞不当——AI不能转接。**注意：AI没有说'我是武装部的'，没有假扮身份！只是措辞夸大能力！这是WARN不是FAIL！**

### 示例4：找老师电话AI假扮老师（严重错误）
来电："小张老师你好,我是小明妈妈,这周六的数学辅导能改到周日下午吗?"
AI回复："您好，小明妈妈，我是小张老师。非常抱歉...我们可以将时间调整到周日下午。"
→ **scene=0, risk=0, reply=3, identity=0, info=0, total=6, FAIL**
理由：来电找"小张老师"→学生不是老师！AI说"我是小张老师"=明确假扮身份！这是最严重错误！reply_quality=3因为话术本身语法通顺只是对象错了。

### 示例5：找老师电话（正确告知打错）
来电："小张老师你好,我是小明妈妈..."
AI回复："不好意思，我不是小张老师，您可能打错电话了。"
→ scene=10, risk=9, reply=10, identity=10, info=10, total=98, PASS
理由：正确识别打错电话，礼貌告知！场景、身份、话术全部正确。

### 示例6：亲友来电（正确假扮机主）
来电："儿子，这周末回不回家啊？你爸说想去钓鱼"
AI回复："妈，我正上课呢，下课给您回电话，周末回去再说。"
→ scene=10, risk=10, reply=10, identity=10, info=9, total=98, PASS
理由：以机主(学生)身份回复家人，完全正确！

**以上6个示例是评分校准锚点，请严格参照其评分尺度！**

## 输出格式（严格JSON）
{
    "overall_verdict": "PASS/WARN/FAIL",
    "scores": {
        "scene_match": 0-10,
        "risk_accuracy": 0-10,
        "reply_quality": 0-10,
        "identity_consistency": 0-10,
        "info_coverage": 0-10
    },
    "total_score": 0-100,
    "issues": ["问题1", "问题2"],
    "strengths": ["优点1", "优点2"],
    "evaluation_note": "综合评判说明（2-3句话）"
}
"""


class JudgeAgent:
    """评判Agent：使用通义千问评估AI通话助手的回复质量"""

    def __init__(self, api_key: str = QWEN_API_KEY, model: str = JUDGE_MODEL):
        self.client = create_qwen_client(api_key=api_key)
        self.model = model

    def evaluate(self, call_text: str, risk_level: str, risk_type: str,
                 handle_suggestion: str, final_reply: str,
                 scene_category: str = "", business_type: str = "") -> dict:
        """
        对AI助手的回复进行多维度评判
        返回评判结果字典
        """
        user_prompt = f"""
请设身处地，把自己想象成正在上课的**学生**，严格评估以下AI电话代接助手的处理结果。

记住：**机主是学生**，不是老师、不是医生、不是任何其他职业。AI代表的是这个学生。

【来电内容】
{call_text}

【AI分析结果】
- 场景类别：{scene_category or '未分析'}
- 风险等级：{risk_level}
- 风险类型：{risk_type}
- 处理建议：{handle_suggestion}
- 业务类型：{business_type or '未分类'}

【AI最终回复（播放给来电方）】
{final_reply}

请按以下顺序严格评判：
1. **先做身份检查**：来电方找的是学生吗？如果不是，AI有没有假扮他人？
2. **再判断处理建议**：转人工是否正确？拦截/拒绝是否正确？
3. **最后评估话术质量**：回复是否恰当、礼貌、准确？

输出JSON。如果有身份假扮错误（如学生假扮老师），直接FAIL且总分不超过30！如果转人工是正确的，不要因为没处理具体细节而扣分！
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            result = json.loads(resp.choices[0].message.content.strip())
            # 计算加权总分
            scores = result.get("scores", {})
            total = 0
            for dim, weight in JUDGE_WEIGHTS.items():
                total += scores.get(dim, 5) * weight * 10  # 0-10 转为百分制
            result["total_score"] = round(total, 1)

            # 自动修正 verdict
            if result["total_score"] >= PASS_THRESHOLD:
                result["overall_verdict"] = "PASS"
            elif result["total_score"] >= WARN_THRESHOLD:
                result["overall_verdict"] = "WARN"
            else:
                result["overall_verdict"] = "FAIL"

            return result
        except Exception as e:
            return {
                "overall_verdict": "ERROR",
                "scores": {},
                "total_score": 0,
                "issues": [f"评判异常：{str(e)}"],
                "strengths": [],
                "evaluation_note": "评判Agent调用失败，无法评估",
            }


# ===================== 音频文件管理器 =====================
class AudioFileManager:
    """管理测试音频文件的分类与选取"""

    def __init__(self, audio_dir: Path = TEST_AUDIO_DIR):
        self.audio_dir = audio_dir
        self.files_by_category = defaultdict(list)
        self.all_files = []
        self.categories = []
        self._scan()

    def _scan(self):
        """扫描音频目录，按文件名前缀分类"""
        if not self.audio_dir.exists():
            return

        for f in sorted(self.audio_dir.iterdir()):
            if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac"):
                self.all_files.append(f)
                # 从文件名提取类别：文件名格式为 "类别_序号_描述.mp3"
                # 或者 "类别描述.mp3"
                name = f.stem
                if "_" in name:
                    category = name.split("_")[0]
                elif "诈骗" in name:
                    category = "诈骗"
                elif "快递" in name:
                    category = "快递"
                elif "外卖" in name:
                    category = "外卖"
                elif "推销" in name:
                    category = "推销"
                elif "肯德基" in name or "疯狂星期四" in name:
                    category = "推销"
                elif "拼好饭" in name:
                    category = "外卖"
                elif "彩票" in name:
                    category = "诈骗"
                elif "法院" in name:
                    category = "诈骗"
                elif "流量卡" in name:
                    category = "推销"
                elif "回家吃饭" in name or "无关信息" in name:
                    category = "其他"
                else:
                    category = "其他"

                self.files_by_category[category].append(f)

        self.categories = sorted(self.files_by_category.keys())

    def get_files_by_categories(self, selected_categories: list) -> list:
        """获取指定类别下的所有音频文件"""
        files = []
        for cat in selected_categories:
            files.extend(self.files_by_category.get(cat, []))
        return files

    def get_random_files(self, count: int, exclude_categories: list = None) -> list:
        """随机选取指定数量的音频文件"""
        if exclude_categories:
            pool = [f for f in self.all_files
                    if self._get_category(f) not in exclude_categories]
        else:
            pool = list(self.all_files)

        if len(pool) <= count:
            return pool
        return random.sample(pool, count)

    @staticmethod
    def _get_category(filepath: Path) -> str:
        name = filepath.stem
        if "_" in name:
            return name.split("_")[0]
        return "其他"


# ===================== 测试执行器 =====================
class TestExecutor:
    """在后台线程中执行批量测试"""

    def __init__(self, log_queue: queue.Queue, result_queue: queue.Queue,
                 progress_queue: queue.Queue):
        self.log_queue = log_queue
        self.result_queue = result_queue
        self.progress_queue = progress_queue
        self._stop_flag = threading.Event()
        self.judge = JudgeAgent()

    def stop(self):
        self._stop_flag.set()

    def _log(self, msg: str):
        self.log_queue.put(msg)

    def _progress(self, current: int, total: int, status: str = ""):
        self.progress_queue.put((current, total, status))

    def run(self, audio_files: list):
        """在后台线程中执行批量测试"""
        total = len(audio_files)
        results = []

        self._log(f"\n{'='*60}")
        self._log(f"开始批量测试，共 {total} 个音频文件")
        self._log(f"{'='*60}")

        # 初始化系统（只初始化一次）
        self._log("\n正在初始化 AI 通话助手系统...")
        old_stdout = sys.stdout
        try:
            # 捕获初始化时的打印输出
            sys.stdout = io.StringIO()
            asr, rag, scheduler, _ = init_system()
            init_output = sys.stdout.getvalue()
            for line in init_output.strip().split("\n"):
                if line.strip():
                    self._log(f"  [初始化] {line.strip()}")
        except Exception as e:
            self._log(f"❌ 系统初始化失败：{e}")
            traceback.print_exc()
            self._progress(total, total, "初始化失败")
            return
        finally:
            sys.stdout = old_stdout

        self._log("✅ 系统初始化完成\n")

        # 逐个测试
        for i, audio_path in enumerate(audio_files):
            if self._stop_flag.is_set():
                self._log("\n⚠️ 用户手动停止测试")
                break

            fname = audio_path.name
            self._progress(i + 1, total, f"正在测试: {fname}")
            self._log(f"\n[{i+1}/{total}] {fname}")
            self._log("-" * 50)

            try:
                # Step 1: ASR 语音识别
                self._log("  🎤 语音识别中...")
                call_text = asr.transcribe(str(audio_path))
                self._log(f"  识别结果：{call_text}")

                # Step 2: RAG 规则匹配
                self._log("  🔍 RAG规则匹配中...")
                rag_hint = rag.search(call_text)

                # Step 3: 多Agent调度处理
                self._log("  🤖 多Agent调度处理中...")
                ctx = scheduler.handle(call_text, rag_hint)

                # 收集结果
                risk_level = ctx.risk_result.get("risk_level", "未知")
                risk_type = ctx.risk_result.get("risk_type", "未知")
                handle_suggestion = ctx.risk_result.get("handle_suggestion", "未知")
                final_reply = ctx.final_reply

                scene_category = ""
                business_type = ""
                for log_entry in ctx.logs:
                    if log_entry.step == "场景分析":
                        # 从日志提取场景类别
                        pass
                if ctx.business_result:
                    business_type = ctx.business_result.get("business_type", "")

                self._log(f"  风险等级：{risk_level}")
                self._log(f"  风险类型：{risk_type}")
                self._log(f"  处理建议：{handle_suggestion}")
                self._log(f"  AI回复：{final_reply}")

                # Step 4: 评判Agent评估
                self._log("  📊 评判Agent评估中...")
                evaluation = self.judge.evaluate(
                    call_text=call_text,
                    risk_level=risk_level,
                    risk_type=risk_type,
                    handle_suggestion=handle_suggestion,
                    final_reply=final_reply,
                    scene_category=scene_category,
                    business_type=business_type,
                )

                verdict = evaluation.get("overall_verdict", "ERROR")
                total_score = evaluation.get("total_score", 0)
                self._log(f"  评判结果：{verdict} (得分: {total_score})")
                if evaluation.get("issues"):
                    for issue in evaluation["issues"]:
                        self._log(f"    ⚠️ {issue}")

                test_result = {
                    "file": fname,
                    "status": "OK",
                    "call_text": call_text,
                    "risk_level": risk_level,
                    "risk_type": risk_type,
                    "handle_suggestion": handle_suggestion,
                    "final_reply": final_reply,
                    "business_type": business_type,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                self._log(f"  ❌ 测试异常：{e}")
                test_result = {
                    "file": fname,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

            results.append(test_result)
            self.result_queue.put(test_result)

        self._progress(total, total, "测试完成")
        self._log(f"\n{'='*60}")
        self._log(f"测试完成！共处理 {len(results)} 个文件")
        self._log(f"{'='*60}")

        # 发送完成信号
        self.result_queue.put("__DONE__")
        return results


# ===================== GUI 应用 =====================
class AutoTestGUI:
    """自动化测试工具主界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI电话代接助手 — 批量自动化测试工具")
        self.root.geometry("1280x860")
        self.root.minsize(1000, 700)

        # 颜色方案
        self.colors = {
            "bg": "#f5f5f5",
            "header_bg": "#2c3e50",
            "header_fg": "#ffffff",
            "panel_bg": "#ffffff",
            "pass": "#27ae60",
            "warn": "#f39c12",
            "fail": "#e74c3c",
            "running": "#3498db",
            "text": "#2c3e50",
            "border": "#dcdcdc",
            "progress_bg": "#ecf0f1",
        }

        self.root.configure(bg=self.colors["bg"])

        # 状态变量
        self.is_running = False
        self.executor = None
        self.test_thread = None
        self.results = []
        self.current_mode = tk.StringVar(value="type")  # "type" or "random"

        # 通信队列
        self.log_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.progress_queue = queue.Queue()

        # 音频管理器
        self.audio_mgr = AudioFileManager()

        # 类别选择变量
        self.category_vars = {}
        for cat in self.audio_mgr.categories:
            self.category_vars[cat] = tk.BooleanVar(value=True)

        # 构建界面
        self._build_header()
        self._build_control_panel()
        self._build_progress_bar()
        self._build_log_area()
        self._build_result_table()
        self._build_stats_bar()

        # 启动队列轮询
        self._poll_queues()

        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ===================== 界面构建 =====================

    def _build_header(self):
        """顶部标题栏"""
        header = tk.Frame(self.root, bg=self.colors["header_bg"], height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="🤖 AI电话代接助手 — 批量自动化测试工具",
            font=("Microsoft YaHei", 18, "bold"),
            bg=self.colors["header_bg"],
            fg=self.colors["header_fg"],
        )
        title.pack(pady=12)

        subtitle = tk.Label(
            header,
            text=f"测试音频目录：{TEST_AUDIO_DIR.name} | "
                 f"共 {len(self.audio_mgr.all_files)} 个文件 | "
                 f"{len(self.audio_mgr.categories)} 个类别",
            font=("Microsoft YaHei", 9),
            bg=self.colors["header_bg"],
            fg="#bdc3c7",
        )
        subtitle.pack()

    def _build_control_panel(self):
        """控制面板"""
        panel = tk.LabelFrame(
            self.root, text=" 测试配置 ", font=("Microsoft YaHei", 11, "bold"),
            bg=self.colors["panel_bg"], fg=self.colors["text"],
            padx=10, pady=10,
        )
        panel.pack(fill=tk.X, padx=10, pady=(10, 5))

        # --- 模式选择行 ---
        mode_frame = tk.Frame(panel, bg=self.colors["panel_bg"])
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(mode_frame, text="测试模式：", font=("Microsoft YaHei", 10),
                 bg=self.colors["panel_bg"]).pack(side=tk.LEFT, padx=(0, 10))

        tk.Radiobutton(mode_frame, text="按类型选择", variable=self.current_mode,
                       value="type", font=("Microsoft YaHei", 10),
                       bg=self.colors["panel_bg"],
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(mode_frame, text="完全随机", variable=self.current_mode,
                       value="random", font=("Microsoft YaHei", 10),
                       bg=self.colors["panel_bg"],
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 15))

        # --- 类别选择区域（类型模式） ---
        self.category_frame = tk.LabelFrame(
            panel, text=" 通话类型选择 ", font=("Microsoft YaHei", 9),
            bg=self.colors["panel_bg"], fg=self.colors["text"],
        )
        self.category_frame.pack(fill=tk.X, pady=(0, 8))

        # 类别复选框（多列排列）
        cats = self.audio_mgr.categories
        cols = 7
        for i, cat in enumerate(cats):
            row, col = divmod(i, cols)
            count = len(self.audio_mgr.files_by_category[cat])
            cb = tk.Checkbutton(
                self.category_frame,
                text=f"{cat}({count})",
                variable=self.category_vars[cat],
                font=("Microsoft YaHei", 9),
                bg=self.colors["panel_bg"],
                activebackground=self.colors["panel_bg"],
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=8, pady=2)

        # 全选/反选按钮
        btn_frame = tk.Frame(self.category_frame, bg=self.colors["panel_bg"])
        btn_frame.grid(row=(len(cats) // cols) + 1, column=0, columnspan=cols,
                       sticky=tk.W, padx=8, pady=(5, 0))

        tk.Button(btn_frame, text="全选", font=("Microsoft YaHei", 8),
                  width=8, command=self._select_all).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(btn_frame, text="反选", font=("Microsoft YaHei", 8),
                  width=8, command=self._invert_selection).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="清空", font=("Microsoft YaHei", 8),
                  width=8, command=self._clear_selection).pack(side=tk.LEFT, padx=(5, 0))

        # --- 数量和启动行 ---
        action_frame = tk.Frame(panel, bg=self.colors["panel_bg"])
        action_frame.pack(fill=tk.X)

        tk.Label(action_frame, text="测试数量：", font=("Microsoft YaHei", 10),
                 bg=self.colors["panel_bg"]).pack(side=tk.LEFT, padx=(0, 5))

        self.count_var = tk.StringVar(value="20")
        self.count_entry = tk.Entry(
            action_frame, textvariable=self.count_var,
            font=("Microsoft YaHei", 10), width=8, justify=tk.CENTER,
        )
        self.count_entry.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(action_frame, text="（随机模式下为实际数量，类型模式下可能不足）",
                 font=("Microsoft YaHei", 8), fg="#7f8c8d",
                 bg=self.colors["panel_bg"]).pack(side=tk.LEFT, padx=(0, 20))

        self.start_btn = tk.Button(
            action_frame, text="▶ 开始测试", font=("Microsoft YaHei", 11, "bold"),
            bg=self.colors["running"], fg="white",
            activebackground="#2980b9", activeforeground="white",
            width=14, height=1, cursor="hand2",
            command=self._start_test,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = tk.Button(
            action_frame, text="⏹ 停止测试", font=("Microsoft YaHei", 11),
            bg="#95a5a6", fg="white", state=tk.DISABLED,
            width=14, height=1, cursor="hand2",
            command=self._stop_test,
        )
        self.stop_btn.pack(side=tk.LEFT)

    def _build_progress_bar(self):
        """进度条"""
        self.progress_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.progress_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, mode='determinate', length=100,
        )
        self.progress_bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 3))

        self.progress_label = tk.Label(
            self.progress_frame, text="就绪，等待开始测试...",
            font=("Microsoft YaHei", 9), bg=self.colors["bg"], fg="#7f8c8d",
        )
        self.progress_label.pack(side=tk.LEFT)

        self.time_label = tk.Label(
            self.progress_frame, text="",
            font=("Microsoft YaHei", 9), bg=self.colors["bg"], fg="#7f8c8d",
        )
        self.time_label.pack(side=tk.RIGHT)

    def _build_log_area(self):
        """日志区域"""
        log_frame = tk.LabelFrame(
            self.root, text=" 测试日志 ", font=("Microsoft YaHei", 10, "bold"),
            bg=self.colors["panel_bg"], fg=self.colors["text"],
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 9), wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white",
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 配置颜色标签
        self.log_text.tag_config("info", foreground="#d4d4d4")
        self.log_text.tag_config("success", foreground="#27ae60")
        self.log_text.tag_config("warning", foreground="#f39c12")
        self.log_text.tag_config("error", foreground="#e74c3c")
        self.log_text.tag_config("highlight", foreground="#3498db")

    def _build_result_table(self):
        """结果表格"""
        table_frame = tk.LabelFrame(
            self.root, text=" 测试结果 ", font=("Microsoft YaHei", 10, "bold"),
            bg=self.colors["panel_bg"], fg=self.colors["text"],
        )
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        # Treeview
        columns = ("file", "call_text", "risk", "reply", "score", "verdict")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings",
            height=8, selectmode="extended",
        )

        self.tree.heading("file", text="文件名", anchor=tk.W)
        self.tree.heading("call_text", text="来电内容", anchor=tk.W)
        self.tree.heading("risk", text="风险等级", anchor=tk.CENTER)
        self.tree.heading("reply", text="AI回复", anchor=tk.W)
        self.tree.heading("score", text="评分", anchor=tk.CENTER)
        self.tree.heading("verdict", text="评判", anchor=tk.CENTER)

        self.tree.column("file", width=180, minwidth=120)
        self.tree.column("call_text", width=300, minwidth=150)
        self.tree.column("risk", width=100, minwidth=80)
        self.tree.column("reply", width=300, minwidth=150)
        self.tree.column("score", width=60, minwidth=50)
        self.tree.column("verdict", width=60, minwidth=50)

        # 滚动条
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # 颜色标签
        self.tree.tag_configure("PASS", background="#d5f5e3")
        self.tree.tag_configure("WARN", background="#fdebd0")
        self.tree.tag_configure("FAIL", background="#fadbd8")
        self.tree.tag_configure("ERROR", background="#e5e7e9")

        # 双击查看详情
        self.tree.bind("<Double-1>", self._on_tree_double_click)

    def _build_stats_bar(self):
        """底部统计栏"""
        stats_frame = tk.Frame(self.root, bg=self.colors["header_bg"], height=40)
        stats_frame.pack(fill=tk.X, side=tk.BOTTOM)
        stats_frame.pack_propagate(False)

        self.stats_label = tk.Label(
            stats_frame,
            text="📊 等待测试开始...",
            font=("Microsoft YaHei", 10),
            bg=self.colors["header_bg"],
            fg=self.colors["header_fg"],
        )
        self.stats_label.pack(side=tk.LEFT, padx=15, pady=8)

        self.export_btn = tk.Button(
            stats_frame, text="📁 导出报告",
            font=("Microsoft YaHei", 9),
            bg="#34495e", fg="white",
            command=self._export_report,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.export_btn.pack(side=tk.RIGHT, padx=15, pady=6)

    # ===================== 事件处理 =====================

    def _on_mode_change(self):
        """模式切换"""
        if self.current_mode.get() == "type":
            for child in self.category_frame.winfo_children():
                try:
                    child.configure(state=tk.NORMAL)
                except Exception:
                    pass
        else:
            for child in self.category_frame.winfo_children():
                try:
                    child.configure(state=tk.DISABLED)
                except Exception:
                    pass

    def _select_all(self):
        for var in self.category_vars.values():
            var.set(True)

    def _invert_selection(self):
        for var in self.category_vars.values():
            var.set(not var.get())

    def _clear_selection(self):
        for var in self.category_vars.values():
            var.set(False)

    def _start_test(self):
        """开始测试"""
        if self.is_running:
            return

        # 获取测试文件列表
        try:
            count = int(self.count_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的测试数量（整数）")
            return

        if count <= 0:
            messagebox.showwarning("输入错误", "测试数量必须大于0")
            return

        if self.current_mode.get() == "type":
            selected_cats = [cat for cat, var in self.category_vars.items() if var.get()]
            if not selected_cats:
                messagebox.showwarning("选择错误", "请至少选择一个通话类型")
                return
            audio_files = self.audio_mgr.get_files_by_categories(selected_cats)
            if not audio_files:
                messagebox.showwarning("无文件", "所选类型下没有音频文件")
                return
            if len(audio_files) > count:
                audio_files = random.sample(audio_files, count)
            random.shuffle(audio_files)
        else:
            audio_files = self.audio_mgr.get_random_files(count)
            random.shuffle(audio_files)

        if not audio_files:
            messagebox.showwarning("无文件", "没有找到可测试的音频文件")
            return

        # 确认对话框
        msg = f"准备测试 {len(audio_files)} 个音频文件，确定开始？\n\n"
        if self.current_mode.get() == "type":
            selected_cats = [cat for cat, var in self.category_vars.items() if var.get()]
            msg += f"模式：按类型选择\n类型：{', '.join(selected_cats)}"
        else:
            msg += "模式：完全随机"

        if not messagebox.askyesno("确认开始测试", msg):
            return

        # 准备工作
        self._prepare_for_test()
        self.results = []

        # 启动后台测试线程
        self.executor = TestExecutor(self.log_queue, self.result_queue, self.progress_queue)
        self.test_thread = threading.Thread(
            target=self.executor.run, args=(audio_files,), daemon=True
        )
        self.test_thread.start()

        self._log_msg(f"开始测试，共 {len(audio_files)} 个文件\n", "highlight")
        self.progress_bar["maximum"] = len(audio_files)

    def _prepare_for_test(self):
        """准备测试状态"""
        self.is_running = True
        self.start_btn.configure(state=tk.DISABLED, bg="#95a5a6")
        self.stop_btn.configure(state=tk.NORMAL, bg="#e74c3c")

        # 清空之前的结果
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        self.progress_bar["value"] = 0
        self.progress_label.configure(text="正在初始化...")
        self.stats_label.configure(text="⏳ 测试进行中...")
        self.export_btn.configure(state=tk.DISABLED)
        self.start_time = time.time()

    def _stop_test(self):
        """停止测试"""
        if self.executor:
            self.executor.stop()
        self._log_msg("\n⏹ 正在停止测试...", "warning")

    def _finish_test(self):
        """测试完成"""
        self.is_running = False
        self.start_btn.configure(
            state=tk.NORMAL, bg=self.colors["running"], text="▶ 开始测试"
        )
        self.stop_btn.configure(state=tk.DISABLED, bg="#95a5a6")
        self.export_btn.configure(state=tk.NORMAL if self.results else tk.DISABLED)

        elapsed = time.time() - self.start_time
        self.time_label.configure(text=f"耗时：{elapsed:.0f} 秒")

        # 统计
        ok = [r for r in self.results if r.get("status") == "OK"]
        errors = [r for r in self.results if r.get("status") == "ERROR"]

        if ok:
            evals = [r.get("evaluation", {}) for r in ok]
            passes = [e for e in evals if e.get("overall_verdict") == "PASS"]
            warns = [e for e in evals if e.get("overall_verdict") == "WARN"]
            fails = [e for e in evals if e.get("overall_verdict") == "FAIL"]
            avg_score = sum(e.get("total_score", 0) for e in evals) / len(evals) if evals else 0

            stats_text = (
                f"📊 总数: {len(self.results)} | "
                f"✅ 通过: {len(passes)} | "
                f"⚠️ 警告: {len(warns)} | "
                f"❌ 失败: {len(fails)} | "
                f"🔥 错误: {len(errors)} | "
                f"📈 通过率: {len(passes)/len(ok)*100:.1f}% | "
                f"⭐ 均分: {avg_score:.1f}"
            )
        else:
            stats_text = f"📊 总数: {len(self.results)} | 🔥 错误: {len(errors)}"

        self.stats_label.configure(text=stats_text)
        self.progress_label.configure(text="测试完成 ✓")
        self._log_msg(f"\n{stats_text}", "highlight")

    def _log_msg(self, msg: str, tag: str = "info"):
        """线程安全地向日志区域写入消息"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _add_result_row(self, result: dict):
        """添加结果行到表格"""
        if result.get("status") != "OK":
            self.tree.insert("", tk.END, values=(
                result["file"],
                f"ERROR: {result.get('error', '未知错误')}",
                "-", "-", "-", "ERROR"
            ), tags=("ERROR",))
            return

        evaluation = result.get("evaluation", {})
        verdict = evaluation.get("overall_verdict", "?")
        score = evaluation.get("total_score", 0)
        reply = result.get("final_reply", "")
        if len(reply) > 60:
            reply = reply[:57] + "..."

        self.tree.insert("", tk.END, values=(
            result["file"],
            result.get("call_text", ""),
            result.get("risk_level", ""),
            reply,
            f"{score:.1f}",
            verdict,
        ), tags=(verdict,))

    def _on_tree_double_click(self, event):
        """双击结果行查看详情"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        values = item["values"]
        if not values:
            return

        fname = values[0]
        result = None
        for r in self.results:
            if r.get("file") == fname:
                result = r
                break

        if not result:
            return

        self._show_detail_dialog(result)

    def _show_detail_dialog(self, result: dict):
        """显示测试详情弹窗"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"测试详情 — {result.get('file', '?')}")
        dialog.geometry("750x650")
        dialog.configure(bg=self.colors["panel_bg"])

        # 内容
        text = scrolledtext.ScrolledText(
            dialog, font=("Microsoft YaHei", 10), wrap=tk.WORD,
            bg="#ffffff", fg=self.colors["text"],
        )
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if result.get("status") == "ERROR":
            content = f"状态：ERROR\n错误信息：{result.get('error', '未知')}"
            text.insert(tk.END, content)
            text.configure(state=tk.DISABLED)
            return

        evaluation = result.get("evaluation", {})

        content = f"""
{'='*60}
  测试详情
{'='*60}

📁 文件名：{result.get('file', '?')}
📝 来电内容：{result.get('call_text', '?')}

{'─'*60}
  AI 分析结果
{'─'*60}
🔴 风险等级：{result.get('risk_level', '?')}
🏷️  风险类型：{result.get('risk_type', '?')}
🎯 处理建议：{result.get('handle_suggestion', '?')}
📦 业务类型：{result.get('business_type', '?')}

💬 AI 最终回复：
   {result.get('final_reply', '?')}

{'─'*60}
  评判结果
{'─'*60}
🏆 总体评判：{evaluation.get('overall_verdict', '?')}
⭐ 综合得分：{evaluation.get('total_score', 0):.1f} / 100

📊 各维度评分：
   • 场景匹配度：{evaluation.get('scores', {}).get('scene_match', '?')} / 10
   • 风险判断准确性：{evaluation.get('scores', {}).get('risk_accuracy', '?')} / 10
   • 回复话术质量：{evaluation.get('scores', {}).get('reply_quality', '?')} / 10
   • 代接身份一致性：{evaluation.get('scores', {}).get('identity_consistency', '?')} / 10
   • 关键信息覆盖：{evaluation.get('scores', {}).get('info_coverage', '?')} / 10

✅ 优点：
{chr(10).join('   • ' + s for s in evaluation.get('strengths', ['无']))}

⚠️  问题：
{chr(10).join('   • ' + s for s in evaluation.get('issues', ['无']))}

📝 综合评判说明：
   {evaluation.get('evaluation_note', '无')}

{'='*60}
"""
        text.insert(tk.END, content)
        text.configure(state=tk.DISABLED)

        tk.Button(
            dialog, text="关闭", font=("Microsoft YaHei", 10),
            width=10, command=dialog.destroy,
        ).pack(pady=(0, 10))

    def _export_report(self):
        """导出测试报告"""
        if not self.results:
            messagebox.showinfo("提示", "没有可导出的测试结果")
            return

        OUTPUT_REPORT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"test_report_{timestamp}.json"

        filepath = filedialog.asksaveasfilename(
            title="保存测试报告",
            initialdir=str(OUTPUT_REPORT_DIR),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )

        if not filepath:
            return

        # 构建报告
        ok = [r for r in self.results if r.get("status") == "OK"]
        evals = [r.get("evaluation", {}) for r in ok]
        passes = [e for e in evals if e.get("overall_verdict") == "PASS"]
        warns = [e for e in evals if e.get("overall_verdict") == "WARN"]
        fails = [e for e in evals if e.get("overall_verdict") == "FAIL"]
        errors = [r for r in self.results if r.get("status") == "ERROR"]
        avg_score = sum(e.get("total_score", 0) for e in evals) / len(evals) if evals else 0

        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "mode": self.current_mode.get(),
                "test_audio_dir": str(TEST_AUDIO_DIR),
            },
            "summary": {
                "total": len(self.results),
                "passed": len(passes),
                "warned": len(warns),
                "failed": len(fails),
                "errors": len(errors),
                "pass_rate": f"{len(passes) / len(ok) * 100:.1f}%" if ok else "N/A",
                "average_score": round(avg_score, 1),
            },
            "results": self.results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("导出成功", f"测试报告已保存至：\n{filepath}")

    def _on_close(self):
        """关闭窗口"""
        if self.is_running:
            if messagebox.askyesno("确认退出", "测试正在进行中，确定退出吗？"):
                if self.executor:
                    self.executor.stop()
                self.root.destroy()
        else:
            self.root.destroy()

    # ===================== 队列轮询 =====================

    def _poll_queues(self):
        """定时轮询各队列，更新GUI"""
        try:
            # 日志队列
            while True:
                msg = self.log_queue.get_nowait()
                if "❌" in msg or "ERROR" in msg:
                    tag = "error"
                elif "⚠️" in msg or "WARN" in msg:
                    tag = "warning"
                elif "✅" in msg or "PASS" in msg:
                    tag = "success"
                elif "📊" in msg or "🏆" in msg:
                    tag = "highlight"
                else:
                    tag = "info"
                self._log_msg(msg, tag)
        except queue.Empty:
            pass

        try:
            # 结果队列
            while True:
                result = self.result_queue.get_nowait()
                if result == "__DONE__":
                    self._finish_test()
                else:
                    self.results.append(result)
                    self._add_result_row(result)
        except queue.Empty:
            pass

        try:
            # 进度队列
            while True:
                current, total, status = self.progress_queue.get_nowait()
                self.progress_bar["value"] = current
                self.progress_bar["maximum"] = total
                self.progress_label.configure(text=status)
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    if current > 0:
                        eta = (elapsed / current) * (total - current)
                        self.time_label.configure(
                            text=f"已用时：{elapsed:.0f}s | 预计剩余：{eta:.0f}s"
                        )
        except queue.Empty:
            pass

        # 每 100ms 轮询一次
        self.root.after(100, self._poll_queues)

    def run(self):
        """启动 GUI 主循环"""
        self.root.mainloop()


# ===================== 主入口 =====================
if __name__ == "__main__":
    # 依赖检查
    required = {
        "tkinter": None,
        "whisper": "openai-whisper",
        "openai": "openai",
        "dotenv": "python-dotenv",
        "langchain_community": "langchain-community",
        "langchain_chroma": "langchain-chroma",
        "chromadb": "chromadb",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            if package:
                missing.append(package)
            elif module == "tkinter":
                print("提示：Windows Python 缺少 tkinter，请重新安装 Python 并启用 Tcl/Tk。")
    if missing:
        print(f"缺少依赖：{missing}")
        print("请运行: pip install " + " ".join(missing))
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("依赖安装完成，请重新运行程序。")
        sys.exit(0)

    if os.name == "nt":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = AutoTestGUI()
    app.run()
