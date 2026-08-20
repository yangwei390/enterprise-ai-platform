# 客服状态重构方案（已收口）

早期通用 DST 设计已被正式强类型状态替代。当前唯一契约见
`customer_service_state_contract.md`，唯一流程见 `customer_service_formal_flow_design.md`。

## 当前原则

- 根可信状态为 `CustomerServiceState`。
- 商品、订单、对话焦点、售后草稿和待澄清动作使用独立强类型子状态。
- LLM 产生的是语义、槽位、引用表达和能力候选，不能写可信状态。
- Python Reducer 只生成预览；Resolver 验证实体；Command 和 Adapter 校验参数；
  `CommitCoordinator` 在 Tool 成功且 Result 契约通过后提交。
- 品类切换清除旧商品候选及相关焦点；原候选回复被裁剪后，序数立即失效。
- 不再保留通用 DST 合并器、兼容性 Provider 或旧 Planner 状态镜像。
