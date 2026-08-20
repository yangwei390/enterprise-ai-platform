# 智能客服正式架构

本文是正式架构入口。精确状态、流程和验收定义分别见：

- `customer_service_state_contract.md`
- `customer_service_formal_flow_design.md`
- `customer_service_acceptance_scenarios.md`

## 唯一主链

```text
Safety
  → Query 补全
  → 共享重量 LLM（意图、槽位、引用表达、能力候选）
  → Python Resolver
  → Strongly Typed Command
  → Dedicated Adapter
  → Tool args_schema.model_validate()
  → Tool
  → Result Contract
  → CommitCoordinator
  → Deterministic Presenter / RAG Evidence Gate
```

当前五个理解阶段均为 `llm_only`，但不是五次模型调用：普通业务轮最多调用一次轻量
Query 补全模型和一次共享重量模型。LLM 只能产生候选，不能决定可信实体、保护参数、
数据库事实或最终回答。

## 状态边界

- 跨轮可信业务状态只有 `metadata["customer_service"]["state"]`。
- 原始消息是有界语义记忆，不是可信业务事实。
- 单轮执行状态是 `AgentState.customer_service_execution`，不得持久化。
- Tool 成功且 Result 契约通过后，只有 `CommitCoordinator` 可以提交业务状态。
- 商品或订单序数必须由 Python Resolver 在当前可见候选批次内验证。
- 原候选回复从消息窗口裁剪后，对应序数立即失效。

## 知识与业务分流

- 商品目录、推荐、订单、物流、售后和转人工进入业务 Tool 链。
- 知识问答当前只覆盖商品说明书；必须先验证商品，再限定可信
  `knowledge_base_id + document_id` 调用 RAG。
- RAG 缺少 answer、source 或 citation 时不得输出说明书事实。
- 写操作保持 `draft → operation_id/draft_id → 跨轮明确确认 → confirm`，并通过幂等门禁。
- 最终业务输出使用确定性 Presenter，Final LLM 为 0。

## 正式实现

正式代码只位于 `backend/app/agents/customer_service_core/`。禁止新增版本目录、版本类名、
平行路由、旧 Planner 兼容入口或绕过 Command/Adapter 的 Tool 参数拼装路径。
