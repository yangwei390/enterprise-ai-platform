# 智能客服语义处理方案（已收口）

精确流程与开关语义见 `customer_service_formal_flow_design.md`。

## 当前处理链

```text
用户原话
  → Query 补全
  → 意图路由
  → 槽位提取
  → 引用表达理解
  → 能力候选选择
  → Python Resolver / Command / Adapter
```

五个阶段各有独立配置。当前均使用 `llm_only`，但共享模型调用：

- 轻量模型只输出 `rewritten_query: str`；
- 重量模型一次性输出意图、槽位、引用表达和能力候选；
- 重量模型接收真实 Tool JSON Schema，包括参数类型、说明和 required 约束；
- LLM 输出必须重新经过 Pydantic、Resolver、Command、Adapter 和真实 Tool Schema；
- LLM 不得直接确定可信商品、订单、知识库范围、Tool 保护参数或最终答案。

Query 补全使用最近的完整用户轮次和受控业务摘要。上下文不得以孤立客服回复开头，
`recent_dialogue` 不重复携带当前 `raw_query`。

说明书问题进入受限 RAG 证据链；其他业务进入 Tool 链。最终输出均由确定性 Presenter
生成，不调用 Final LLM。
