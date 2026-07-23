# Enterprise AI Platform 系统架构

## 当前版本

V2

---

## 系统组成

客户端

↓

FastAPI

↓

Service

↓

RAG

↓

Memory

↓

Agent

↓

LangGraph V2 Runtime

↓

LLM

↓

PostgreSQL

Redis

Qdrant

---

## 后续模块规划

第一阶段

- 配置中心
- 日志系统
- 文件上传
- PDF解析

第二阶段

- Chunk
- Embedding
- Qdrant

第三阶段

- Retriever
- Reranker
- RAG问答

第四阶段

- Conversation
- 短期记忆
- 长期记忆

第五阶段

- Agent
- Workflow
- Tool Calling

第六阶段

- MCP
- Plugin

第七阶段

- 登录
- 权限
- 后台管理

---

## 系统原则

所有模块必须做到：

- 独立
- 可测试
- 可扩展
- 可维护

## Agent Runtime

当前项目仅保留 LangGraph V2 Runtime。

正式 Agent 入口统一进入：

- `/agent/chat`
- `/agent/chat/stream`
- `/agents/run`

运行时统一由 `AgentRuntimeFactory` 创建 `LangGraphAgentRuntime`。

项目不再保留可执行旧版 Agent Runtime、旧规则规划器、旧 Agent Executor 或旧 Runtime fallback。

Workflow Runtime 同步统一为 LangGraph V2 Runtime，`WorkflowRuntimeFactory` 仅返回 `LangGraphWorkflowRuntime`。

## RAG Provider Architecture

后续 RAG 能力采用全生命周期 Provider 边界：

```text
KnowledgeProvider
├── LocalKnowledgeProvider
│   ├── lifecycle: LocalKnowledgeLifecycleProvider
│   └── retrieval: LocalRetrievalProvider
└── RagflowKnowledgeProvider
    ├── lifecycle: RagflowKnowledgeLifecycleProvider
    └── retrieval: RagflowRetrievalProvider
```

当前本地 RAG 仍是默认实现。`LocalKnowledgeProvider` 和 `RagflowKnowledgeProvider` 是按知识库后端聚合的门面，各自组合自己的生命周期能力和检索能力；`KnowledgeProviderFactory` 根据知识库绑定的 provider 返回完整聚合 Provider，不允许 Local lifecycle 与 RAGFlow retrieval 被错误组合。

Local lifecycle 负责本地文档保存、解析、切片、Embedding、Qdrant、BM25、删除和重新索引；Local retrieval 负责本地召回、融合、Rerank、MMR 和邻居扩展。

RAGFlow 接入后作为独立知识库后端，通过 HTTP API 通信。RAGFlow lifecycle 负责自己的 Dataset、文档解析、切片、Embedding、索引、Chunk 管理、解析状态、失败重试和远端删除；RAGFlow retrieval 正式主链路使用 RAGFlow Retrieval API 返回 chunks。Chat 检索通过已选择的聚合 Provider 访问其 `retrieval` 能力。

Local 与 RAGFlow 不迁移、不自动同步、不双写、不共享底层索引。平台公共层继续负责 Conversation、Memory、Context、PromptBuilder、LLMFactory、Sources、Citations 和 ChatResponse。

详细规范见：

- `.ai/10_RAG_Provider_Architecture.md`
