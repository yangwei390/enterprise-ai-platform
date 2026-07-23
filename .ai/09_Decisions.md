# Enterprise AI Platform 架构决策

## 为什么选择 FastAPI

原因：

- 性能高
- 异步支持优秀
- 自动生成 OpenAPI
- 企业应用广泛

---

## 为什么选择 PostgreSQL

原因：

- 企业级数据库
- 稳定
- SQL能力强
- ORM支持完善

---

## 为什么选择 Redis

原因：

- 缓存
- Session
- Queue
- Memory

---

## 为什么选择 Qdrant

原因：

- 开源
- RAG 支持完善
- 性能优秀
- 社区活跃

---

## 为什么选择 LangGraph

原因：

- Agent Workflow
- State 管理
- Memory
- Tool Calling

---

## 为什么 Docker

原因：

- 环境统一
- 易部署
- 易扩容
- 易迁移

---

## 为什么采用渐进式架构

原则：

先完成 MVP。

随着业务发展逐步扩展。

避免过度设计。

遵循：

YAGNI

KISS

SOLID

DRY

---

## 为什么采用全生命周期 RAG Provider

后续接入 RAGFlow 时，采用：

```text
KnowledgeProvider
├── LocalKnowledgeProvider
│   ├── lifecycle: LocalKnowledgeLifecycleProvider
│   └── retrieval: LocalRetrievalProvider
└── RagflowKnowledgeProvider
    ├── lifecycle: RagflowKnowledgeLifecycleProvider
    └── retrieval: RagflowRetrievalProvider
```

原因：

- Local RAG 与 RAGFlow 的文档解析、切片、Embedding、索引、召回、Rerank 和删除生命周期不同。
- 两套系统的数据源不自动同步，不能把 Provider 故障静默切换成另一套知识库。
- RAGFlow 是独立 RAG 后端，不应直接访问或修改其数据库、Elasticsearch、Infinity 或数据卷。
- 当前项目已有 Local RAG 链路，应保留默认 Local，不直接替换。
- 上层 API、Chat、Conversation、Memory、Prompt、LLM、Sources、Citations 和 Evaluation 需要统一，避免前端感知底层实现差异。
- `LocalKnowledgeProvider` 和 `RagflowKnowledgeProvider` 按后端聚合各自的 lifecycle 与 retrieval，避免错误组合不同后端的生命周期和检索能力。

边界：

- 现有知识库默认视为 `local`。
- Provider 绑定后不得直接原地修改。
- Provider 绑定发生在聚合 `KnowledgeProvider` 层，`KnowledgeProviderFactory` 根据绑定的 provider 返回完整聚合 Provider。
- Chat 检索通过已选择的聚合 Provider 访问其 `retrieval` 能力。
- Local lifecycle 只管理本地 PostgreSQL 业务记录、上传文件、Qdrant、BM25 和本地索引版本。
- Local retrieval 负责本地 Dense Retrieve、Sparse Retrieve、Fusion、Rerank、MMR 和 Neighbor Expansion。
- Ragflow lifecycle 只通过 RAGFlow HTTP API 管理 Dataset、Document、Chunk、索引、解析状态和重试。
- Ragflow retrieval 正式主链路使用 RAGFlow Retrieval API 返回 chunks。
- RAGFlow 返回 chunks 后，不再执行本地 Dense Retrieve、Sparse Retrieve、Fusion、Rerank、MMR 和 Neighbor Expansion。
- RAGFlow 具体 API 路径和字段必须在下一阶段根据实际部署版本确认，不在架构决策中猜测。
