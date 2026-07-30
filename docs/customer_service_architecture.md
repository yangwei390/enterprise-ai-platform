# 智能客服正式架构

## 唯一主链

```text
Safety
  → Understanding（每个普通回合最多一次）
  → Reducer Preview
  → Entity Resolver
  → Strongly Typed Command
  → Dedicated Command Adapter
  → Tool args_schema.model_validate()
  → Tool
  → Result Contract
  → CommitCoordinator
  → Deterministic Presenter / RAG Evidence Gate
```

LLM 只输出 `SemanticFrame`，表达“用户说了什么”。真实实体、Tool、数据库事实、
最终参数和可信业务状态都不能由 LLM 直接决定。

## 状态边界

- 跨轮可信业务状态只有 `metadata["customer_service"]["state"]`。
- 单次 Graph 执行状态只有顶层 `AgentState.customer_service_execution`。
- `PendingTransaction` 是单轮 Tool 事务，不得持久化到 metadata。
- `PendingAfterSales` 是 Tool 创建草稿后产生的跨轮业务状态，与
  `PendingTransaction` 不同。
- 规则、LLM、历史消息和 Planner 只能产生状态预览；只有
  `CommitCoordinator` 可以在 Tool 成功且结果契约通过后提交业务状态。

## 目录职责

- `backend/app/agents/customer_service_core/understanding.py`：规则优先、LLM
  仅补充缺口的 `SemanticFrame` 理解入口。
- `backend/app/agents/customer_service_core/command_builder.py`：Reducer 和 Resolver
  之后的强类型 Command 构建入口。
- `backend/app/agents/customer_service_core/contracts.py`：正式状态、阶段、事务和 Command。
- `backend/app/agents/customer_service_core/reducer.py`：状态预览。
- `backend/app/agents/customer_service_core/entity_resolver.py`：引用解析和池外校验决策。
- `backend/app/agents/customer_service_core/adapters.py`：Command 到 Tool Schema 的唯一入口。
- `backend/app/agents/customer_service_core/commit.py`：结果校验和原子提交。
- `backend/app/agents/customer_service_core/hooks.py`：LangGraph 通用客服 Tool hook。
- `backend/app/agents/customer_service_core/strategy.py`：唯一正式 Planner Strategy。
- `backend/app/agents/customer_service_core/presenter.py`：确定性输出和证据门禁。

禁止新增客服版本目录、版本配置、版本路由、平行状态或通用字典参数拼装器。

## 执行阶段

- `NEW`：执行 Safety 和 Understanding。
- `WAITING_TOOL`：等待当前事务 Tool Result。
- `CONTINUE`：复用第一轮 `GoalSnapshot` 继续确定性规划，不再次理解。
- `READY_FOR_FINAL`：进入确定性最终输出。
- `READY_FOR_CLARIFICATION`：输出收敛追问。
- `FAILED`：失败终止，业务状态不变。

每轮最多执行两个业务 Tool。商品身份验证成功后可以继续一次限定
`knowledge_base_id + document_id` 的 `knowledge_search`；其他普通 Tool 结束后直接
进入最终输出。

## 实体和候选

- 商品候选按有界批次保存 `product_code/name/category/batch_id/position/`
  `primary_manual_document_id`。
- 序数只在最近相关品类批次内解析，不能映射为数据库默认第一条。
- 候选池外显式商品编码或名称返回 `VERIFICATION_REQUIRED`，通过
  `search_products` 校验。
- 显式订单号必须通过 `query_order` 校验归属。
- 历史候选只提供身份可信度，不提供价格、库存、物流或权限的新鲜度。

## 失败和证据

语义失败、实体解析失败、Tool Schema 失败、权限失败、Tool 失败和 Result 契约失败
都不得更新业务状态。商品目录事实来自商品 Tool；说明书事实必须先验证商品，再对
绑定文档执行 RAG。RAG 没有 answer、source 和 citation 时，不输出说明书业务断言。
