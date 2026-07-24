# 智能客服 Agent 阶段一架构规范

## 1. 当前源码事实

当前项目已经具备 Agent、Tool、RAG 和 Evaluation 的基础能力：

- `backend/app/api/agent.py` 已提供统一入口：`POST /agent/chat`、`POST /agent/chat/stream`、`POST /agents/run`。
- `backend/app/agents/langgraph/factory.py` 的 `AgentRuntimeFactory.get_runtime()` 当前固定返回 `LangGraphAgentRuntime`。
- `backend/app/agents/langgraph/runtime.py` 负责加载 AgentDefinition、构造 LangGraph state、加载和保存 Agent Session Memory、执行 graph、返回 `AgentRuntimeResult`。
- `backend/app/agents/definition.py` 当前内置 `general_agent`、`knowledge_agent`、`knowledge_research_agent`，没有 `customer_service_agent`。
- `backend/app/agents/catalog.py` 当前推荐逻辑仍在 `general_agent` 和 `knowledge_research_agent` 之间选择，没有把客服 Agent 设为默认。
- `backend/app/tools/registry.py` 通过 `ToolRegistry` 管理工具发现、注册、启停、descriptor 和 provider。
- `backend/app/tools/executor.py` 通过 `ToolExecutor` 执行工具、校验参数、处理 timeout/retry/cache/错误。
- `backend/app/tools/providers/builtin.py` 当前只注册 `CalculatorTool`、`EchoTool`、`CurrentTimeTool`、`KnowledgeSearchTool`。
- `backend/app/tools/builtin/knowledge_tool.py` 的 `knowledge_search` 复用 `RagChatPipeline`，返回 answer、sources、citations、metadata。
- `backend/app/tools/schemas.py` 的 `KnowledgeSearchArgs` 当前只有 `query`、`knowledge_base_id`、`conversation_id`、`memory_context`，没有 `document_id`。
- `backend/app/models/knowledge_base.py` 当前没有商品字段，也没有 provider 字段。
- `backend/app/models/document.py` 当前没有商品说明书关联字段。
- `backend/app/retrievers/qdrant_retriever.py` 从 Qdrant payload 顶层读取 `document_id`。
- `backend/app/retrievers/sparse/bm25_index.py` 的 `SparseDocument` 和搜索结果携带顶层 `document_id`。
- `evaluation/v2/targets/agent.py` 当前可运行 Agent Runtime 并记录 tool_calls、observations、sources、citations、metadata。
- `evaluation/v2/metrics/agent.py` 已有工具选择、工具序列、工具成功率、循环次数等 Agent 指标。
- `evaluation/v2/metrics/tool.py` 已有工具执行成功、结果匹配、结果包含、timeout、retry、cache 等工具指标。

当前没有实现：

- `customer_service_agent`。
- 商品目录 ORM、Repository、Service。
- `products` 表。
- `product_document_links` 表。
- `search_products`、`recommend_products`、`compare_products`。
- `query_order`、`query_logistics`。
- `create_after_sales_ticket`、`create_human_handoff`。
- 订单、物流、售后、转人工真实系统接入。
- RAGFlow 接入。本阶段暂停 RAGFlow 开发，不影响 Local RAG。

## 2. 目标与非目标

### 2.1 阶段一目标

阶段一只建立一个智能客服 Agent：

```text
customer_service_agent
  -> PostgreSQL 商品目录
  -> Local RAG 产品说明书和规则文档
  -> Mock 订单 / 物流 / 售后工单 / 转人工服务
  -> LangGraph V2 Agent Runtime
```

阶段一能力：

- 商品查询。
- 多轮商品条件累计。
- 商品推荐。
- 商品对比。
- 产品说明书咨询。
- 订单状态查询。
- 物流查询。
- 退换货规则咨询。
- 创建模拟售后工单。
- 创建模拟转人工记录。

### 2.2 阶段一非目标

本阶段不做：

- 不使用多 Agent。
- 不接入 RAGFlow。
- 不连接真实商城系统。
- 不连接真实订单系统。
- 不连接真实物流系统。
- 不连接真实售后系统。
- 不连接真实人工客服平台。
- 当前架构文档步骤不创建数据库表，也不生成 Alembic migration；后续数据库 Checkpoint 将创建 `products`、`product_document_links` ORM 和 migration。
- 不把客服 Agent 设置为全局默认。
- 不通过 PDF 文件名自动创建商品。
- 不通过上传说明书自动猜测商品映射。

## 3. 总体架构

```text
POST /agent/chat 或 POST /agent/chat/stream
  -> AgentRuntimeFactory
  -> LangGraphAgentRuntime
  -> customer_service_agent AgentDefinition
  -> ToolScope 根据 tool_allowlist 限制可用工具
  -> ToolExecutor
      -> search_products
      -> recommend_products
      -> compare_products
      -> knowledge_search
      -> query_order
      -> query_logistics
      -> create_after_sales_ticket
      -> create_human_handoff
  -> Agent Session Memory 保存多轮上下文
  -> ChatResponse / SSE 返回
```

三类数据源：

| 数据源 | 负责内容 | 不负责内容 |
|---|---|---|
| PostgreSQL 商品目录 | 商品编码、品牌、型号、分类、模拟价格、模拟库存、上下架、功能标签、规格、热度、官网 URL | 说明书正文、订单、物流、工单 |
| Local RAG | 产品说明书、使用方法、维护、安全、故障、保修、退换货规则 | 商品结构化价格库存、订单物流 |
| Mock Service | 模拟订单、模拟物流、模拟售后工单、模拟转人工记录 | 真实商城、真实客服、真实支付或真实物流 |

## 4. Agent、Tool、Service、Repository 责任边界

### 4.1 Agent

`customer_service_agent` 负责：

- 理解用户意图。
- 选择合适工具。
- 在相同 `conversation_id` 下利用现有 Agent Session Memory 继承多轮上下文。
- 根据工具结果组织自然语言回答。
- 在证据不足、商品不存在、无说明书、订单无权限时明确说明。

Agent 不负责：

- 直接访问数据库。
- 直接拼 SQL。
- 自由创造商品、价格、库存、订单、物流、工单。
- 绕过 Tool 的权限、确认和幂等规则。

### 4.2 Tool

Tool 负责把 Agent 意图转换成确定性 Service 调用。

阶段一目标 Tool：

| Tool | 责任 |
|---|---|
| `search_products` | 结构化商品查询和分页 |
| `recommend_products` | 基于硬条件和评分规则推荐最多 3 款商品 |
| `compare_products` | 对确定商品集合做字段级对比 |
| `knowledge_search` | 查询 Local RAG 中的说明书和规则文档 |
| `query_order` | 查询 Mock 订单状态 |
| `query_logistics` | 查询 Mock 物流状态 |
| `create_after_sales_ticket` | 创建模拟售后工单 |
| `create_human_handoff` | 创建模拟转人工记录 |

Tool 不负责：

- 直接修改真实业务系统。
- 绕过参数校验。
- 返回未脱敏敏感信息。
- 在没有商品结果时让 LLM 自行补商品。

### 4.3 Service

Service 负责业务规则：

- 商品查询条件标准化。
- 推荐硬过滤和评分。
- 商品对比字段读取。
- 订单归属校验。
- 售后工单确认检查。
- 幂等键处理。
- Mock 数据读取和写入模拟记录。

Service 不负责：

- Agent 规划。
- Prompt 生成。
- LLM 总结。
- 直接处理 HTTP request/response。

### 4.4 Repository

Repository 负责数据访问：

- 商品表查询。
- 商品说明书关联查询。
- Mock Fixture 读取或模拟记录持久化。

Repository 不负责：

- 推荐评分。
- 订单权限判断。
- Tool 参数解释。
- LLM 输出。

## 5. 单 Agent 选择原因

阶段一使用单 `customer_service_agent`，原因：

- 阶段一目标是走通智能客服闭环，不是验证多 Agent 协作。
- 商品查询、推荐、说明书检索、订单、物流、售后都属于同一个客服会话域。
- 现有 LangGraph V2 Runtime 已支持工具选择、工具调用、反思、最终回答和 session memory。
- 单 Agent 更容易控制 Tool allowlist、安全边界、评估用例和回归范围。

升级多 Agent 的条件见第 11 节。

## 6. Tool 清单

### 6.1 已有 Tool

`knowledge_search` 已存在，当前参数：

```text
query
knowledge_base_id
conversation_id
memory_context
```

后续计划为 `KnowledgeSearchArgs` 增加可选：

```text
document_id
```

目标用途：

```text
根据型号查询商品
-> 查询 product_document_links
-> 获取主说明书 document_id
-> knowledge_search(document_id=...)
-> 只检索该型号说明书
```

### 6.2 后续新增 Tool

```text
search_products
recommend_products
compare_products
query_order
query_logistics
create_after_sales_ticket
create_human_handoff
```

这些 Tool 必须加入 `customer_service_agent.tool_allowlist`，不得加入所有 Agent 默认工具集。

## 7. 多轮会话边界

多轮商品条件累计使用现有能力：

- 相同 `conversation_id`。
- `LangGraphAgentRuntime._session_id()` 将 conversation 映射为 `conversation:{conversation_id}`。
- `LangGraphAgentRuntime._load_session()` 加载 Agent Session Memory。
- `LangGraphAgentRuntime._save_session()` 保存 messages、tool_results、plan、trace_id 等。

普通入口：

```text
POST /agent/chat
```

当前普通入口不会在缺少 `conversation_id` 时自动创建 Conversation。需要多轮继承时，客户端必须先获得有效 `conversation_id`，后续请求必须持续复用同一个 `conversation_id`。缺少 `conversation_id` 时，不承诺多轮商品条件继承。

流式入口：

```text
POST /agent/chat/stream
```

当前流式入口在缺少 `conversation_id` 时会创建 Conversation，并在事件中返回该 ID。客户端必须保存该 ID，并在后续轮次复用。

Evaluation 边界：

- `evaluation/v2/targets/agent.py` 当前构造 `AgentRuntimeRequest` 时没有传入 `agent_id`。
- Checkpoint 7 必须修改为等价逻辑：`agent_id=case.input.get("agent_id")`。
- 客服 Evaluation Fixture 必须显式包含 `input.agent_id=customer_service_agent`。
- 否则可能执行默认 `general_agent`，导致评估结果无效。

示例：

```text
用户第一轮：给我推荐几个豆浆机
-> search_products(category=豆浆机, sale_status=on_sale, in_stock_only=true, sort_by=popularity, sort_order=desc, page_size=3)

用户第二轮：挑几款200到300区间的
-> 继承 category=豆浆机
-> 增加 price_min=200, price_max=300
-> search_products(...)
```

阶段一不新增会话系统，不新增独立商品条件存储表。若后续发现现有 Agent Session Memory 对结构化槽位不稳定，再设计专门的 slot state。

## 8. Local RAG 接入边界

Local RAG 负责：

- 产品说明书。
- 使用方法。
- 清洁和维护。
- 安全要求。
- 故障处理。
- 保修说明。
- 退换货规则。

边界：

- 说明书上传不会自动创建商品。
- 商品可以暂时没有说明书。
- 没有关联说明书时，Agent 不得编造产品详细功能。
- 当前 Qdrant payload 顶层有 `document_id`。
- 当前 BM25 文档也有顶层 `document_id`。
- 后续需要让 Dense 和 Sparse 检索使用一致的 `document_id` 过滤语义。
- 不修改 `RetrieverPipeline` 的步骤顺序。
- 不修改 Fusion、Rerank、MMR、Neighbor Expansion。
- 不修改 `DocumentPipeline`。

## 9. Mock 系统边界

Mock Service 负责：

- 模拟订单。
- 模拟物流。
- 模拟售后工单。
- 模拟转人工记录。

必须明确：

- 价格、库存、热度、订单、物流和工单都是模拟数据。
- 本系统不是九阳官方客服。
- 不连接真实商城、订单、物流、售后或人工客服系统。
- 不伪造客服姓名、等待时间或处理结果。
- 创建售后工单和转人工记录只代表本地模拟记录。

## 10. 安全和故障处理

- 查询订单必须校验订单号和模拟客户身份或手机号后四位。
- 不允许查询其他客户订单。
- 订单、手机号、地址等敏感字段必须脱敏。
- 创建售后工单必须先展示摘要并获得用户明确确认。
- 创建售后工单必须使用幂等键，重复请求返回同一张模拟工单。
- 创建转人工记录必须返回 `handoff_id`，并标记 `mock=true`。
- Tool 执行失败时返回明确错误，Agent 不得把失败说成成功。
- 商品无结果时明确说明无匹配商品，不自动放宽硬条件。
- 说明书无关联或无检索结果时明确说明当前资料不足。

阶段一的对话确认只是 Mock 安全保护。未来接入真实写操作时，必须使用正式 Workflow Approval 或其他审批机制。

## 11. 未来升级多 Agent 的条件

满足以下条件后再考虑多 Agent：

- 商品、订单、售后、知识检索各自 Service 已稳定。
- Tool selection 评估能稳定区分商品查询、推荐、说明书、订单、物流、售后。
- 单 Agent 的上下文过长或工具选择明显混乱。
- 需要独立权限、独立模型策略或独立审计策略。
- 真实商城、真实售后或人工客服系统接入后，需要更强审批和隔离。

未满足这些条件前，不引入多 Agent。

## 12. 后续开发 Checkpoint

Checkpoint 1：

- 商品数据层。
- `products`、`product_document_links` ORM 和 migration。
- 数据库约束。
- schema 变更前按项目红线获得确认。

Checkpoint 2：

- 商品应用层。
- Product Repository / Service / API。
- 幂等 Seed。
- `link_product_manual.py`。

Checkpoint 3：

- 说明书限定检索。
- `knowledge_search(document_id=...)`。
- Qdrant/BM25 `document_id` 一致过滤。
- 跨型号污染验证。

Checkpoint 4：

- Mock 业务层。
- 订单、物流、售后、转人工 Service。
- 归属校验、脱敏、确认、幂等。

Checkpoint 5：

- 客服领域 Tool。
- 商品查询、推荐、对比 Tool。
- Mock 业务 Tool。
- ToolRegistry 注册。

Checkpoint 6：

- 客服 Agent。
- `customer_service_agent`。
- Tool schema 和 tool_allowlist。
- 不改变默认 Agent。
- 不得在依赖 Tool 尚未实现时提前注册不可用 Agent。

Checkpoint 7：

- Evaluation V2 和回归。
- 补齐 Evaluation V2 用例和指标。
- 验证普通、流式 Agent 入口。
- 修改 `evaluation/v2/targets/agent.py`，让 `AgentRuntimeRequest` 透传 `agent_id=case.input.get("agent_id")`。
- 客服 Evaluation Fixture 显式包含 `input.agent_id=customer_service_agent`。
- 普通和流式入口分别验证 `conversation_id` 行为。
