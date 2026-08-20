# 智能客服正式链路清理结果

## 1. 正式入口

正式生产入口统一为：

```text
planner_strategy="customer_service"
  -> CustomerServiceStrategy
  -> SemanticFrame
  -> Resolver
  -> 强类型 Command
  -> Adapter
  -> Tool
  -> CommitCoordinator
  -> Presenter
```

项目不再保留并行 Planner、旧 DST/FSM 编排或旧语义重写实现。

## 2. 已完成清理

- 删除旧单文件 Planner 和只服务旧 Planner 的十二个 core 模块；
- 删除纯旧架构测试，并把仍有价值的安全、并发和 session CAS 测试迁入正式链路；
- hooks 缺少正式 Execution 或 PendingTransaction 时直接阻断，不再动态回退旧实现；
- 能力选择器统一使用 `capability_selector.py`、`CapabilitySuggestion`、
  `CapabilitySelectionResult`、`select_capability` 和 `capability_selection` 调试键；
- 删除已无正式入口的用途目录匹配分支；用途能力统一进入说明书 RAG；
- 删除旧的单一意图模式配置，保留五个独立阶段 Mode；
- 包入口不再导出旧状态类型。

## 3. 已验证迁移

- 同一会话、同一售后 operation 的并发确认只能执行一次；
- session CAS 不会用过期状态复活已确认的售后草稿；
- 售后明确确认文本、同轮确认、参数篡改、事务匹配继续由正式门禁验证；
- 正式 Tool 继续通过真实 Schema、权限、Result 契约和 Commit 门禁。

## 4. 保留的兼容边界

售后 `pending_after_sales` 兼容镜像继续保留。运行时 session CAS 使用该镜像防止并发确认
复活旧草稿；它不是第二套 Planner。未来如果移除，必须先把 CAS 比较切换到正式
`CustomerServiceState.pending_after_sales`，并重新验证并发保存顺序。

## 5. 明确保留

- `backend/app/agents/customer_service_contract.py`；
- `backend/app/agents/customer_service_core/` 正式链路；
- `backend/app/services/customer_service.py`；
- `backend/app/schemas/customer_service.py`；
- `backend/app/tools/builtin/customer_service.py`；
- 客服业务服务、真实 Tool、正式架构、API、运行时和记忆测试。
