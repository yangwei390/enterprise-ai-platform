# Enterprise AI Platform 项目宪章

## 一、项目目标

打造一套企业级 AI Platform，具备以下核心能力：

- 企业知识库（RAG）
- AI Agent
- 长期记忆（Memory）
- MCP（Model Context Protocol）
- Workflow
- 多模型支持
- 企业级权限管理
- 企业级后台管理

最终目标：

打造一套可部署、可维护、可扩展的企业级 AI 平台。

---

## 二、技术栈

开发语言

- Python 3.12

Web 框架

- FastAPI

数据库

- PostgreSQL

缓存

- Redis

向量数据库

- Qdrant

Agent

- LangGraph

容器

- Docker

---

## 三、开发原则

1. 配置优先，不允许硬编码。
2. 一个模块只负责一个职责。
3. 先设计，再开发。
4. 每个功能必须能够独立测试。
5. 每完成一个功能立即提交 Git。
6. 文档与代码保持同步。

---

## 四、AI 协作规范

Codex、ChatGPT、Claude Code 修改代码前必须：

- 阅读本项目宪章
- 阅读系统架构文档
- 遵守编码规范
- 不修改无关模块
- 保持向后兼容

---

## 五、代码质量要求

- 高内聚、低耦合
- 优先组合，而不是继承
- 不重复代码（DRY）
- 保持简单（KISS）
- 每个接口必须处理异常
- 每个模块必须输出日志