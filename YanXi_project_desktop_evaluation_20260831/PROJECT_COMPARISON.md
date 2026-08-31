# 言犀 (YanXi) 版本对比与整合说明

> 本文档详细记录了两个版本的差异，以及整合过程中所做的变更。

---

## 一、版本来源

| 版本 | 仓库 | 说明 |
|------|------|------|
| **CC 版** | `CCai186/YanXi_project_CC` | CC 个人完整版，全部模块已实现运作 |
| **QM-newer 版** | `QM-newer/YanXi_project` | 整合版（CC + KCN + Terry + LZM），架构更清晰但大量模块为空壳 |
| **整合版（本目录）** | — | 以 QM-newer 为基础架构，填充 CC 的完整功能实现 |

---

## 二、QM-newer 比 CC 版少了什么

以下模块在 CC 版中有完整实现，QM-newer 版只有空目录或占位 __init__.py：

| 模块 | 作用 | CC 版代码行数 | QM-newer 版状态 |
|------|------|:---:|:---:|
| agents/call_types.py | 16 种来电类型定义 + 回复模板 | ~300 | 缺失（只有 enums 抽象） |
| agents/conversation_memory.py | 三层对话记忆（短期/工作/长期） | ~280 | 缺失 |
| agents/scam_detector.py | RAG + LLM 诈骗检测 | ~350 | 缺失 |
| agents/business_handler.py | 多轮业务对话（外卖/快递） | ~350 | 缺失 |
| agents/urgent_forwarder.py | 紧急程度评估与转接 | ~330 | 缺失 |
| knowledge/caller_profile.py (=profile_store) | 来电画像（信任评分/黑白名单） | ~230 | 缺失 |
| knowledge/habit_learner.py | LLM 习惯学习 + 持久化 | ~330 | 缺失 |
| knowledge/knowledge_expander.py | 内置知识库扩展（52 条） | ~230 | 缺失 |
| notification/card_builder.py | 4 种通知卡片 + 持久化 | ~200 | 缺失 |
| store/call_logger.py | JSONL 通话记录持久化 | ~140 | 缺失 |
| voice/stt.py | Faster-Whisper 语音识别 | ~450 | 缺失 |
| voice/tts.py | Edge-TTS 语音合成 | ~160 | 缺失 |
| voice/recorder.py | 通话录音管理 | ~85 | 缺失 |
| utils/presence.py | 机主 4 种状态管理 | ~200 | 缺失 |

## 三、QM-newer 比 CC 版多了什么

| 模块 | 作用 | 说明 |
|------|------|------|
| core/enums.py | 枚举常量（CallType / Action / Mode） | CC 版字符串散落各处，这里统一为枚举类 |
| core/llm_client.py | 多后端 LLM（DeepSeek + GLM + Qwen） | CC 版仅支持 DeepSeek，这里支持三个后端自动降级 |
| retrieval/bm25_retriever.py | BM25 关键词检索 | CC 版用简单字符串包含匹配，这里用 jieba + BM25Okapi |
| retrieval/vector_retriever.py | ChromaDB 向量检索 | 与 CC 版功能等价但更模块化 |
| retrieval/hybrid_retriever.py | 双路混合检索 | 封装 BM25 + 向量 双路 + RRF 融合 |
| retrieval/graph_retriever.py | 知识图谱检索（保留接口） | KCN 贡献 |
| api.py | FastAPI REST API | 完整的 HTTP 服务层 |
| models.py | Pydantic 数据模型 | 结构化请求/响应模型 |
| pyproject.toml | Poetry 依赖管理 | 比 requirements.txt 更规范 |
| tests/test_integration.py | 集成测试 | 测试用例雏形 |

## 四、整合策略

以 **QM-newer 版为架构基础**，将 **CC 版的功能实现**填充进去，适配使用 QM-newer 的枚举常量和多后端 LLM：

```
QM-newer 架构骨架
  ├── core/enums.py           增强：加入 CC 的 16 种来电类型定义
  ├── core/llm_client.py      保留：多后端 + 自动降级
  ├── core/config.py          增强：合并 CC 的默认配置值
  ├── classification/         保留：QM-newer 的三级分类器 + 关键词表
  ├── retrieval/              保留：BM25 + 向量 双路混合
  ├── api.py / models.py      保留：FastAPI 服务
  ├── tests/                  增强：补充更多测试用例
  │
  └── 以下全部从 CC 版填充
      ├── agents/call_types.py           CC 16 种来电类型 + 回复模板
      ├── agents/conversation_memory.py  CC 三层记忆
      ├── agents/scam_detector.py        CC 诈骗检测
      ├── agents/business_handler.py     CC 业务对话
      ├── agents/urgent_forwarder.py     CC 紧急转接
      ├── knowledge/profile_store.py     CC 来电画像
      ├── knowledge/habit_learner.py     CC 习惯学习
      ├── knowledge/knowledge_expander.py CC 知识库扩展
      ├── notification/card_builder.py   CC 通知卡片
      ├── store/call_logger.py           CC 通话记录
      ├── voice/stt.py / tts.py / recorder.py  CC 语音模块
      └── utils/presence.py              CC 状态管理
```

## 五、做了哪些改动

### 5.1 新增文件

从 CC 版移植了 14 个功能模块，替换 QM-newer 的空占位

### 5.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| core/enums.py | 增强 CallType 枚举，加入 CC 的 16 种来电类型及处理动作映射 |
| core/config.py | 合并 CC 的默认配置值（12 个配置段）|
| orchestration/state.py | 增加对话记忆/画像/录音等状态字段 |
| orchestration/nodes.py | 重写所有 4 个处理节点，接入 CC 的完整处理逻辑 |
| orchestration/orchestrator.py | 增加 resume_conversation/learn_habit 等方法，接入所有新模块 |
| main.py | 恢复 CC 的所有命令（/habit /profile /stats 等）+ 测试模式 |
| config.yaml | 加入 recorder/embedding 等配置段 |
| pyproject.toml | 加入 CC 所需的依赖 |
| tests/test_integration.py | 增加更多 test cases |

### 5.3 删除文件

- 旧的 `src/agents/__init__.py` 等空占位文件被替换为实际实现
