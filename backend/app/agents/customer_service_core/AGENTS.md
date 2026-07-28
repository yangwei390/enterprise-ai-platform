# Customer Service Core

本目录只承载智能客服的通用对话编排，不实现数据库访问或具体 Tool。

## 文件职责

- `schemas.py`：意图、请求、槽位、DST 与 FSM 事件的数据契约。
- `dst.py`：DST 的加载、迁移、校验与原子状态更新。
- `router.py`：安全前置路由与确定性意图规则。
- `contextualizer.py`：显式字段提取、请求重写与 LLM 输出规范化。
- `resolver.py`：基于可信 DST 的通用目标解析。
- `fsm.py`：状态转移、槽位检查、打断和恢复。
- `dispatcher.py`：把 FSM 指令转换为业务执行计划。
- `actions.py`：FSM 业务动作到既有 Tool 的固定映射。

## 约束

- 不写具体商品类别、品牌、型号或订单号。
- LLM 输出只能作为候选语义，真实目标必须由 DST 校验。
- Tool 参数只能从已校验请求和 DST 生成。
- 新逻辑只读取 `ConversationDST`；旧字段仅允许作为会话迁移输入和兼容输出，
  不得参与新的路由、目标解析或状态决策。
- 不在本目录直接访问 Repository、数据库、Redis、Qdrant 或外部 API。
- 写操作必须保留草稿、显式确认和幂等门禁。
