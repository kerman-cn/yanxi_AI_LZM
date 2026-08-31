# 言犀 (YanXi) — AI 智能通话助手

> **2026-08-31 桌面主程序与评测台分享版**：请先阅读 [分享版运行说明](SHARING_README.md)。
> 本目录是独立新增的修改版，不替换仓库根目录原工程，也不修改同学分享的上游仓库。
> 当前桌面程序分析来电内容并给出建议，不会真实接听、拨打、转接或挂断电话。

> 你的 AI 私人秘书，自动处理来电：拦截诈骗、代接外卖快递、转接重要来电。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 它能做什么

言犀是一个 AI 驱动的智能通话助手，可以自动接听电话并根据来电内容做出智能决策：

| 场景 | 示例来电 | 言犀行为 |
|------|---------|---------|
| 🛡️ 诈骗拦截 | "我是市公安局，你涉嫌洗钱" | **自动拒接** + 发送诈骗预警卡片 |
| 🍕 外卖配送 | "我是美团外卖，餐到楼下了" | **代接回复** "放门口就好" + 推送通知卡片 |
| 📦 快递送达 | "你的快递到菜鸟驿站了" | **代接回复** + 推送取件提醒 |
| 👨‍👩‍👧 家人来电 | "妈，我今晚回家吃饭" | **代接回复** + 通知机主 |
| 💼 领导来电 | "明天下午有紧急会议" | **转接机主**（重要来电不拦截） |

## 核心亮点

```
┌──────────────────────────────────────────────────────────┐
│                      来电文本                              │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│              三级分类器（关键词 → RAG → LLM）                │
│  先用关键词快速匹配 → 不确定时检索知识库 → 仍不确定交 LLM     │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│                  双路混合检索                               │
│  BM25 关键词检索 (0.4)  +  BGE-large-zh 向量检索 (0.6)     │
│              ↓ RRF 融合 + 语义重排序                        │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│                LangGraph StateGraph 编排                   │
│  分类 → 查画像 → 判断机主状态 → 路由处理 → 生成通知          │
└──────────┬───────────────────────────────────────────────┘
           ▼
     ┌────┴────┬────────┬────────┐
     ▼         ▼        ▼        ▼
  诈骗拦截  业务处理  紧急转接  普通代接
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek / 智谱GLM / 通义千问 | 统一抽象层，支持运行时切换和自动降级 |
| **嵌入模型** | BAAI/bge-large-zh-v1.5 | 中文语义向量，本地运行 |
| **向量数据库** | ChromaDB | 持久化语义检索 |
| **关键词检索** | BM25 + jieba 分词 | 稀疏检索，精确关键词匹配 |
| **编排框架** | LangGraph StateGraph | 多节点条件路由 |
| **语音识别** | faster-whisper large-v3 | 本地 STT，无需 API |
| **语音合成** | Edge TTS | 免费 TTS |
| **API 服务** | FastAPI + Pydantic v2 | RESTful 接口 |

## 快速开始

### 桌面可视化入口（新增）

双击 `启动言犀桌面版.bat`，或运行 `.venv\Scripts\python.exe desktop.py`。
支持首次配置并保存 API Key、麦克风输入、单个/批量录音导入、文字分析、结果展示与语音朗读。
语音方案参考 PythonProject2 的 sounddevice + Whisper turbo。详细设置、依赖、隐私说明和测试见 [DESKTOP_README.md](./DESKTOP_README.md)。

### 独立 AI 管家评测台（新增）

双击 `启动言犀评测台.bat`，或运行 `.venv\Scripts\python.exe evaluation.py`。
支持录音集批量评测、独立大模型四维评分、生成新来电并去重、合成语音端到端测试、JSON/CSV/HTML报告。
首次启动单独配置API Key，评测数据与日常管家隔离。使用方法、评审边界和分享说明见 [EVALUATION_README.md](./EVALUATION_README.md)。

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/YanXi_Integration.git
cd YanXi_Integration

# 安装依赖（需要 Python 3.10+）
pip install poetry
poetry install
```

### 2. 配置 API Key

只需配置 **1 个** LLM API Key（推荐 DeepSeek，便宜好用）：

```bash
# 创建 .env 文件，写入你的 Key
echo DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx > .env
```

> 支持的 LLM：`DEEPSEEK_API_KEY` / `ZHIPU_API_KEY` / `QWEN_API_KEY`
>
> Embedding、STT、TTS 全部本地运行，无需额外 API。

### 3. 运行

```bash
# 文本交互模式
python src/main.py

# 测试模式（一键跑 7 个测试用例）
python src/main.py --test

# 启动 API 服务
python src/api.py
```

### 4. 使用

启动后进入 `>>` 交互界面：

```
>> 13800001111 我是美团外卖的，你的餐到楼下了

[结果] 外卖配送 (置信度=92%)
  动作: notify_card
  言犀: 好的，请放在门口，谢谢！
  卡片: 外卖配送 - 13800001111 来电，外卖已送达

>> 13800002222 我是市公安局的，你涉嫌洗钱

[结果] 诈骗拦截 (置信度=95%)
  动作: reject_call
  卡片: ⚠️ 诈骗预警 - 13800002222 疑似冒充公检法
```

**系统命令**：

| 命令 | 作用 |
|------|------|
| `/help` | 显示帮助 |
| `/status` | 查看系统状态（LLM后端、检索可用性、机主状态） |
| `/presence busy` | 设为忙碌（来电转接） |
| `/presence free` | 设为空闲（正常接听） |
| `/presence dnd` | 设为勿扰（全部拒接） |
| `/quit` | 退出 |

## API 接口

```bash
# 启动 API 服务
python src/api.py

# 新来电处理
POST /api/call/new
{
  "caller_number": "13800001111",
  "call_text": "我是美团外卖的"
}

# 多轮对话
POST /api/call/turn
{
  "session_id": "xxx",
  "user_input": "好的谢谢"
}

# 系统状态
GET /api/status

# 机主状态设置
POST /api/presence
{
  "mode": "busy"
}
```

## 项目结构

```
YanXi_Integration/
├── config.yaml                 # 统一配置文件
├── pyproject.toml              # 依赖管理
├── README.md                   # 本文件
├── INTEGRATION.md              # 内部：模块整合来源说明
├── src/
│   ├── main.py                 # 主入口（文本/测试模式）
│   ├── api.py                  # FastAPI REST API
│   ├── models.py               # Pydantic 数据模型
│   ├── core/                   # 核心层：配置、LLM客户端、日志、枚举
│   ├── orchestration/          # LangGraph 编排层
│   ├── classification/         # 三级分类器 + 关键词表
│   ├── retrieval/              # 双路混合检索（BM25 + 向量）
│   ├── agents/                 # 业务处理器（诈骗/外卖/快递/紧急/普通）
│   ├── knowledge/              # 知识库（用户画像/习惯学习/知识扩展）
│   ├── notification/           # 通知卡片构建
│   ├── store/                  # 通话记录持久化
│   └── voice/                  # 语音识别 + 合成
├── data/                       # 运行时数据（向量库/日志/画像等）
└── tests/                      # 测试用例
```

## 设计原则

- **统一接口规范**：所有模块 `def xxx(input) -> output` 纯函数风格
- **三级降级分类**：关键词 → RAG 检索 → LLM，确保离线可用
- **LLM 自动降级**：DeepSeek → 智谱GLM → 通义千问，一个挂了自动切换
- **核心与交互解耦**：pipeline 内无 `input()`/`print()`，便于接入 API
- **枚举驱动**：所有分类/动作/状态使用枚举常量，消除字符串硬编码

## 文档

- [INTEGRATION.md](./INTEGRATION.md) — 内部文档：模块整合来源对照，详细记录每个文件从哪个子项目合并而来

## License

MIT License
