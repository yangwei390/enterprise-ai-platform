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
