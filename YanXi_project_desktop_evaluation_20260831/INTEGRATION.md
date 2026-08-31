# 言犀 AI 智能通话助手 — 内部整合说明

> ⚠️ **内部文档** — 面向开发者，用于模块合并审查和代码溯源。
> 对外展示请查看 [README.md](./README.md)。

四个项目（CC、KCN、Terry、LZM）的统一整合版本。本文档详细记录每个文件的来源、引用的具体部分，以及整合时的设计决策。

## 整合来源对照总览

| 整合文件 | 来源 | 说明 |
|----------|------|------|
| `orchestration/orchestrator.py` | CC + Terry | LangGraph 骨架来自 CC，双路混合检索，多轮对话来自 Terry |
| `orchestration/nodes.py` | CC + Terry | 四个 Agent 节点拆分自 CC 的 orchestrator 内联逻辑 + Terry 的 agents.py |
| `classification/classifier.py` | CC | 三级分类器（关键词→RAG→LLM）整体来自 CC 的 agents/classifier.py |
| `classification/keywords.py` | CC + LZM | 16 类关键词表来自 CC，高危词权重表来自 LZM |
| `retrieval/bm25_retriever.py` | CC + LZM | BM25 稀疏检索，关键词加权策略来自 CC/LZM |
| `retrieval/vector_retriever.py` | CC + Terry | ChromaDB 向量检索来自 CC 的 knowledge/retriever.py + Terry 的 knowledge_base.py |
| `retrieval/hybrid_retriever.py` | CC | BM25(0.4) + 向量(0.6) 双路混合 |
| `retrieval/fusion.py` | CC | RRF 双路融合算法（去掉三路加权，简化为 fuse_two） |
| `retrieval/reranker.py` | CC | 语义重排序（embedder 优先 + 关键词降级容错） |
| `retrieval/embedder.py` | CC + Terry | BGE-large-zh 嵌入模型封装，替代 CC 的 small-zh 和 Terry 的 MiniLM |
| `core/llm_client.py` | CC + Terry | 统一 LLM 抽象层，综合 CC 的 core/llm_client.py 和 Terry 的 agents.py（双后端切换） |
| `core/config.py` | CC + KCN + Terry | 统一配置，合并 CC 的 config.yaml + KCN 的 utils/config.py + Terry 的 config.py |
| `core/enums.py` | CC | 枚举常量整体来自 CC 的字符串硬编码提取，统一规范化 |
| `api.py` | Terry | FastAPI REST API，整体来自 Terry 的 api.py |
| `models.py` | Terry | Pydantic 数据模型，整体来自 Terry 的 models.py |
| `main.py` | CC + Terry | 主入口文本模式来自 Terry 的 main.py，测试模式来自 CC |

## 项目结构

```
YanXi_Integration/
├── pyproject.toml              # Poetry 依赖管理
├── config.yaml                 # 统一配置文件
├── README.md                   # 本文件
├── src/
│   ├── __init__.py
│   ├── main.py                 # 主入口（文本/语音模式）
│   ├── api.py                  # FastAPI REST API 服务
│   ├── models.py               # Pydantic 数据模型
│   │
│   ├── core/                   # 核心层
│   │   ├── __init__.py
│   │   ├── config.py           # 配置加载与校验
│   │   ├── llm_client.py       # 统一 LLM 抽象层（DeepSeek/GLM/Qwen）
│   │   ├── logger.py           # 日志工具
│   │   └── enums.py            # 枚举常量（CallType, CallAction, PresenceMode等）
│   │
│   ├── orchestration/          # 编排层（LangGraph StateGraph）
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # 主调度器（基于CC的StateGraph）
│   │   ├── state.py            # OrchestratorState 定义
│   │   └── nodes.py            # 四个Agent节点函数
│   │
│   ├── classification/         # 分类模块
│   │   ├── __init__.py
│   │   ├── classifier.py       # 三级分类器（关键词→RAG→LLM）
│   │   └── keywords.py         # 关键词匹配表
│   │
│   ├── retrieval/              # 双路混合检索
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py   # 第一路：BM25 关键词检索（jieba 分词）
│   │   ├── vector_retriever.py # 第二路：向量语义检索（BGE-large-zh + ChromaDB）
│   │   ├── hybrid_retriever.py # 双路混合检索器（BM25 + 向量 → RRF 融合）
│   │   ├── fusion.py           # RRF 双路融合
│   │   ├── reranker.py         # 语义重排序
│   │   └── embedder.py         # 嵌入模型（BGE-large-zh）
│   │
│   ├── agents/                 # Agent 处理模块
│   │   ├── __init__.py
│   │   ├── scam_handler.py     # 诈骗拦截处理
│   │   ├── business_handler.py # 业务处理（外卖/快递/银行）
│   │   ├── urgent_handler.py   # 紧急来电处理
│   │   └── normal_handler.py   # 普通来电处理
│   │
│   ├── knowledge/              # 知识库
│   │   ├── __init__.py
│   │   ├── profile_store.py    # 来电者画像存储
│   │   ├── knowledge_expander.py # 知识库扩展
│   │   └── habit_learner.py    # 习惯学习
│   │
│   ├── notification/           # 通知模块
│   │   ├── __init__.py
│   │   └── card_builder.py     # 通知卡片构建
│   │
│   ├── store/                  # 存储模块
│   │   ├── __init__.py
│   │   └── call_logger.py      # 通话记录
│   │
│   └── voice/                  # 语音模块
│       ├── __init__.py
│       ├── stt.py              # 语音识别（Whisper）
│       └── tts.py              # 语音合成（Edge TTS）
│
├── data/                       # 数据目录
│   ├── chroma_db/              # ChromaDB 向量库
│   ├── graph/                  # 知识图谱数据
│   ├── habits/                 # 习惯数据
│   ├── profiles/               # 来电者画像
│   ├── call_logs/              # 通话记录
│   └── notifications/          # 通知卡片
│
└── tests/                      # 测试
    ├── __init__.py
    └── test_integration.py     # 集成测试
```

## 技术栈

- **Python**: 3.10+
- **依赖管理**: Poetry
- **LLM**: DeepSeek / 智谱GLM / 通义千问（统一抽象层，支持运行时切换）
- **嵌入模型**: BAAI/bge-large-zh-v1.5
- **向量数据库**: ChromaDB
- **关键词检索**: BM25 (rank-bm25 + jieba 分词)
- **编排框架**: LangGraph StateGraph
- **API 服务**: FastAPI + Pydantic v2
- **语音**: faster-whisper + Edge TTS

## 快速开始

```bash
# 安装依赖
poetry install

# 文本模式运行
poetry run python src/main.py --text

# API 服务
poetry run python src/api.py

# 运行测试
poetry run python tests/test_integration.py
```

## 架构设计

### 统一 API 规范
- 所有模块接口统一为 `def xxx(user_input: str) -> str` 格式
- 核心逻辑与交互解耦，pipeline 内不得有 `input()`/`print()`
- 消除模块级全局可变状态，状态通过函数参数传入
- 异常不得静默吞掉，至少 `logging.error`
- 分类逻辑使用枚举常量

### 双路混合检索
1. **BM25 关键词检索**: jieba 分词 + BM25Okapi，权重 0.4
2. **向量语义检索**: BGE-large-zh + ChromaDB，权重 0.6
3. **RRF 融合**: Reciprocal Rank Fusion 合并双路结果
4. **语义重排序**: embedder 语义相似度 + 关键词命中率综合排序

> 去掉了知识图谱检索（NetworkX + Louvain 社区检测），因为通话助手场景下知识图谱属于"大炮打蚊子"——需要构建图、做社区检测，但实际检索的只是诈骗话术模板/业务对话模板，用图谱没有明显收益。BM25 + 向量双路架构更轻量高效。

### LLM 抽象层
- 统一接口封装 DeepSeek、智谱GLM、通义千问
- 支持运行时切换和降级

---

## 整合来源详细说明

### 一、编排层 (orchestration/)

#### `orchestrator.py` — 主调度器
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/orchestrator.py` | LangGraph StateGraph 骨架（`_build_graph` 构建节点+条件边）、`CallOrchestrator` 类结构、`run()`/`resume_conversation()` 入口函数、`_node_classify`/`_node_lookup_profile`/`_node_infer_presence`/`_node_route`/`_node_notify` 共 5 个节点方法、`PresenceMode` 机主状态管理 |
| **KCN** | `src/retrieval/retrieval/` | 三路检索并行调用模式（`retrieve()` 方法中并行调用 vector→graph→ngram 三路）、RRF 加权融合调用、语义重排序调用 |
| **Terry** | `pipeline.py` | 多轮对话续接逻辑（`resume_conversation` 中重新分类→诈骗检测→业务回复映射）、机主状态设置接口 `set_presence()` |

#### `nodes.py` — Agent 节点函数
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/orchestrator.py` | `route_by_type()` 条件路由函数，根据 `type_id` 分派到 scam/business/urgent/normal 四个分支 |
| **Terry** | `agents.py` | 四个 Agent 节点（scam/business/urgent/normal）的拆分思路来自 Terry 的 5 类 Agent 体系 |

#### `state.py` — 状态定义
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/orchestrator.py` | `OrchestratorState` TypedDict 中所有字段（call_text, caller_number, type_id, call_type_name, confidence, classify_method, caller_profile, presence_mode, final_action, agent_reply, notification_card, timing） |

---

### 二、分类模块 (classification/)

#### `classifier.py` — 三级分类器
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/classifier.py` | 整体架构：三级降级策略（关键词→RAG→LLM）、`ClassifyResult` 数据类、`ThreeTierClassifier` 类、`classify()` 入口方法（空文本检测→无意义检测→关键词匹配→RAG检索→LLM增强→兜底）、`_rag_classify()` 通过检索结果统计类型分布、`_llm_classify()` 通过 LLM prompt 分类、`is_meaningless()` 无意义文本检测（连续数字/单字重复/词组重复/数字占比检测） |
| **LZM** | `call_agent.py` | `is_meaningless()` 中数字占比检测的阈值策略（>=60% 且不含正常关键词）参考了 LZM 的敏感词检测思路 |

#### `keywords.py` — 关键词匹配表
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/classifier.py` | `CATEGORY_KEYWORDS` 16 类关键词表完整提取：scam/scam_risk/food_delivery/express/taxi_arrived/telemarketing/game_promo/family/leader/friend/colleague/client/bank/interview |
| **LZM** | `call_agent.py` | 诈骗关键词权重表 `KEYWORD_WEIGHTS`（在 ngram_retriever.py 中使用），区分高危词(3.0)、中危词(2.0-2.5)、正常词(-1.0~-1.5) 三类权重 |

---

### 三、检索模块 (retrieval/)

#### `bm25_retriever.py` — BM25 关键词检索（新增）
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/knowledge/retriever.py` | `SCAM_KEYWORD_WEIGHTS` 关键词加权策略（高危词 3.0/中危词 2.0/正常词 -1.5 三级权重） |
| **LZM** | `call_agent.py` | `KEYWORD_WEIGHTS` 诈骗检测权重表、关键词命中加权逻辑 |

**设计理念**：用标准的 BM25 统计算法替代原来的 n-gram 倒排索引，更简洁高效。jieba 分词 + BM25Okapi，比纯 n-gram 字符级匹配更准确。保留 CC/LZM 的关键词加权策略作为辅助调整。

#### `vector_retriever.py` — 向量语义检索
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/knowledge/retriever.py` | ChromaDB PersistentClient 懒加载（`_init_chroma()`）、`collection.query()` 调用模式、`retrieve()` 方法、`add_documents()` 批量添加、相似度计算 `1/(1+distance)` |
| **Terry** | `knowledge_base.py` | `is_available()` 可用性检查接口 |

#### `hybrid_retriever.py` — 双路混合检索器（新增）
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/retrieval/fusion.py` | RRF 融合器 + 重排序管线 |

**设计理念**：简洁双路架构。不再使用三路（向量+图谱+n-gram），去掉知识图谱这一路——在通话助手场景中，知识图谱需要构建图+社区检测，但实际只检索"诈骗话术模板/业务对话模板"，用图谱没有明显收益，属于"大炮打蚊子"。BM25 替代 n-gram 倒排索引更标准高效。

#### `fusion.py` — RRF 融合器
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/retrieval/fusion.py` | `RRFFuser` 类结构、`doc_key()` 文档唯一标识生成、RRF 公式 `1/(k+rank+1)` |

**简化**：去掉 `fuse_weighted()` 三路加权融合，改为 `fuse_two()` 双路加权融合。保留 `fuse()` 多路等权接口作为扩展。

#### `reranker.py` — 语义重排序器
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/retrieval/reranker.py` | 综合得分公式（语义相似度 × weight + 关键词命中率 × weight + 原始相似度 × weight）、embedder 优先 + 关键词降级的容错策略 |

**简化**：去掉 BGE-reranker Cross-Encoder 依赖（太重），直接用 embedder 计算语义相似度做重排序。

#### `embedder.py` — 嵌入模型
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/knowledge/retriever.py` | SentenceTransformer 懒加载模式（`_load_model()`）、`encode()` 方法支持单文本/批量、`get_dimension()` 维度查询、HF_ENDPOINT 镜像设置 |
| **Terry** | `knowledge_base.py` | 嵌入模型选择的配置化方式（从 `config.yaml` 读取 `embedding_model` 和 `embedding_device`） |

**关键变更**：统一将 CC 的 `BAAI/bge-small-zh-v1.5` 和 Terry 的 `all-MiniLM-L6-v2` 替换为 `BAAI/bge-large-zh-v1.5`。

---

### 四、核心层 (core/)

#### `llm_client.py` — 统一 LLM 抽象层
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/core/llm_client.py` | OpenAI 兼容 API 调用（`client.chat.completions.create`）、自动重试（指数退避 `2^(attempt-1)`，默认 3 次）、超时控制（默认 30s）、`response_type="json"` 模式（JSON 解析 + ` ```json ``` ` 代码块提取 + `{...}` 正则提取）、`default_response` 兜底降级 |
| **Terry** | `agents.py` | 多后端支持的设计思路（智谱 GLM / DeepSeek / Ollama 多后端），整合版扩展为 DeepSeek + 智谱 GLM + 通义千问 三家 API、`switch_backend()` 运行时切换、`available_backends` 查询 |
| **Terry** | `config.py` | API Key 从环境变量读取的方式（`DEEPSEEK_API_KEY`/`ZHIPU_API_KEY`/`QWEN_API_KEY`） |

**关键整合**：CC 的 LLMClient 只封装了单一 OpenAI 后端，Terry 支持双后端切换。整合版统一为三后端 + 自动降级链路（`fallback_chain`），任一后端失败自动尝试下一个。

#### `config.py` — 统一配置
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `config.yaml` | RAG 配置段（embedding_model, chroma_persist_dir, collection_name, chunk_size, retrieval_top_k）、LLM 配置段（backend, temperature, max_tokens, retry） |
| **KCN** | `src/utils/config.py` | `DEFAULT_CONFIG` 默认值合并策略（`_deep_merge()` 深度合并字典）、知识图谱配置段（graph_path, communities_path, louvain_seed）、`PROJECT_ROOT` 路径解析 |
| **Terry** | `config.py` | 日志配置段（logging.level, logging.format, logging.file）、通话记录/用户画像/通知卡片的 persist_path 配置 |

#### `enums.py` — 枚举常量
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **CC** | `src/agents/classifier.py` + `src/agents/orchestrator.py` | 将 CC 中分散的字符串硬编码（`"scam"`, `"food_delivery"`, `"reject"`, `"forward"`, `"keyword"`, `"rag"`, `"llm"` 等）统一提取为 `CallType`（16 种来电类型）、`CallAction`（6 种动作）、`PresenceMode`（4 种状态）、`ClassifyMethod`（7 种分类方法）、`LLMBackend`（3 个 LLM 后端）五个枚举类 |

---

### 五、API 服务层

#### `api.py` — FastAPI REST API
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **Terry** | `api.py` | 整体 API 结构：`/api/call/new` 新来电处理、`/api/call/turn` 多轮对话、`/api/status` 系统状态、`/api/presence` 机主状态设置、`/health` 健康检查、FastAPI lifespan 生命周期管理、CORS 中间件配置、`sessions` 会话字典管理 |
| **CC** | `src/main.py` | API 中调用 `orchestrator.run()` 和 `orchestrator.resume_conversation()` 的入口方式 |

#### `models.py` — Pydantic 数据模型
| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **Terry** | `models.py` | `NewCallRequest`/`TurnRequest`/`TurnResponse`/`CallSummary` 通话处理模型、`UserProfileUpdate`/`UserProfileResponse` 用户画像模型、`SystemStatus` 系统状态模型（含 llm_backend, available_backends, vector/bm25/hybrid 可用性） |

---

### 六、主入口 (main.py)

| 来源 | 文件 | 引用的具体部分 |
|------|------|---------------|
| **Terry** | `main.py` | 文本交互模式（`run_text_mode`：`input()` 循环读取 → `orchestrator.run()` 处理 → 打印结果）、命令行参数解析（`--text`/`--test`）、横幅打印 `print_banner()` |
| **CC** | `src/main.py` | 测试模式（`run_test_mode`：预定义 7 个测试用例，覆盖 scam/food_delivery/express/family/leader 等场景）、测试用例的设计思路 |

---

### 七、未整合但保留接口的模块

以下模块在原项目中有实现，整合版已预留接口文件（`__init__.py`），但核心逻辑待后续阶段补充：

| 预留模块 | 来源 | 待整合内容 |
|----------|------|-----------|
| `agents/scam_handler.py` | Terry + LZM | Terry 的 agent_block 诈骗拦截 + LZM 的 RiskControlAgent 5 级风险分级 |
| `agents/business_handler.py` | Terry + LZM | Terry 的 agent_delivery 配送处理 + LZM 的 BusinessProcessAgent 异常场景处理 |
| `agents/urgent_handler.py` | Terry + CC | Terry 的 agent_important 重要来电转接 + CC 的 forward 逻辑 |
| `agents/normal_handler.py` | Terry | Terry 的 agent_general 通用兜底 + agent_escalation 人工升级 |
| `knowledge/profile_store.py` | Terry + CC | Terry 的 user_manager.py 用户画像 + CC 的 CallerProfileStore |
| `knowledge/habit_learner.py` | CC + Terry | CC 的 HabitLearner 习惯学习 + Terry 的课程时间判断 |
| `knowledge/knowledge_expander.py` | CC | CC 的 KnowledgeExpander 知识库扩展 |
| `notification/card_builder.py` | CC + KCN | CC 的 CardBuilder 通知卡片 + KCN 的 PipelineResult 格式化 |
| `store/call_logger.py` | CC | CC 的 CallLogger 通话记录持久化 |
| `voice/stt.py` | Terry + LZM | Terry 的 Vosk ASR + LZM 的 Whisper ASR |
| `voice/tts.py` | Terry + LZM | Terry 的 Edge TTS + LZM 的 tts_generator.py |

