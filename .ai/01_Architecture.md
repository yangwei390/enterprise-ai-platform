# Enterprise AI Platform 系统架构

## 当前版本

V1

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