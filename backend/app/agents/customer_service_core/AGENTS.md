# Customer Service Core

本目录只承载智能客服的通用对话编排，不实现数据库访问或具体 Tool。

## 文件职责

- `schemas.py`：意图、请求、槽位、DST 与 FSM 事件的数据契约。
- `dst.py`：DST 的加载、迁移、校验与原子状态更新。
- `router.py`：安全前置路由与确定性意图规则。
- `contextualizer.py`：显式字段提取、请求重写与 LLM 输出规范化。
- `semantic_schemas.py`：语义提取、字段来源与业务域候选载荷的数据契约。
- `semantic_rewrite.py`：基于用户原话、受控记忆和 DST 生成候选语义，不解析真实目标。
- `resolver.py`：基于可信 DST 的通用目标解析。
- `fsm.py`：状态转移、槽位检查、打断和恢复。
- `dispatcher.py`：把 FSM 指令转换为业务执行计划。
- `actions.py`：FSM 业务动作到既有 Tool 的固定映射。
- `contracts.py`：正式业务状态、执行阶段、事务、强类型 Command 和结果契约。
- `adapters.py`：Command 到真实 Tool Schema 的唯一参数转换入口。
- `commit.py`：Tool Result 契约校验和成功后原子提交。
- `hooks.py`：LangGraph ToolNode 使用的统一客服前置/后置 hook。
- `presenter.py`：结构化 Tool 结果的确定性输出和 RAG 引用门禁。
- `strategy.py`：唯一正式 Planner Strategy 与显式执行阶段控制。
- `reducer.py`：业务状态预览；只生成 proposed patch，不直接提交可信状态。
- `entity_resolver.py`：候选批次内引用解析与池外显式实体校验决策。
- `understanding.py`：规则优先、LLM 补充的正式 SemanticFrame 理解入口。
- `read_only_fallback.py`：重写后规则仍无法承接时，只推荐白名单只读能力和候选槽位。
- `command_builder.py`：从 SemanticFrame、状态预览和解析结果生成强类型 Command。

## 约束

- 不写具体商品类别、品牌、型号或订单号。
- LLM 输出只能作为候选语义，真实目标必须由 DST 校验。
- 只读能力 Fallback 不得推荐或生成写操作；建议必须重新经过强类型 Command、
  Resolver、权限和 Adapter Schema 校验。
- Tool 参数只能从已校验请求和 DST 生成。
- 新逻辑只读取 `ConversationDST`；旧字段仅允许作为会话迁移输入和兼容输出，
  不得参与新的路由、目标解析或状态决策。
- 不在本目录直接访问 Repository、数据库、Redis、Qdrant 或外部 API。
- 写操作必须保留草稿、显式确认和幂等门禁。
- `metadata["customer_service"]["state"]` 是唯一跨轮业务状态；禁止新增平行状态。
- 单轮执行态只放 `AgentState.customer_service_execution`，不得持久化到 metadata。
- Tool 参数必须由专用 Command Adapter 生成并通过真实 `args_schema.model_validate()`。
- 只有 Tool 成功且结果契约校验通过后，`CommitCoordinator` 才能提交状态。
- 本目录和正式客服链路不得使用版本号命名或版本分支。
