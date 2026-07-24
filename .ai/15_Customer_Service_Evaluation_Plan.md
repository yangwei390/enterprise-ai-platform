# 智能客服 Agent 阶段一评估计划

## 1. 商品查询用例

目标：

- 验证 `search_products` 被正确选择。
- 验证 category、brand、price、stock、sale_status 等条件进入 Tool 参数。
- 验证无结果时不编造商品。

示例：

```text
用户：查一下在售的豆浆机
期望 Tool：search_products
期望参数：category=豆浆机, sale_status=on_sale
```

## 2. 多轮条件累计用例

目标：

- 验证相同 `conversation_id` 下能继承上一轮商品条件。
- 普通入口 `POST /agent/chat` 缺少 `conversation_id` 时不承诺多轮继承。
- 流式入口 `POST /agent/chat/stream` 缺少 `conversation_id` 时会创建 Conversation，并在事件中返回该 ID；客户端必须保存并复用。
- 只有相同 `conversation_id` 的连续请求，才验证多轮条件继承。

示例：

```text
第一轮：给我推荐几个豆浆机
第二轮：挑几款200到300区间的
```

期望：

- 第二轮仍带 `category=豆浆机`。
- 第二轮新增 `price_min=200`、`price_max=300`。

## 3. 推荐硬条件用例

目标：

- 验证 `recommend_products` 不自动放宽硬条件。

示例：

```text
用户：我要一款300以内、适合宿舍、必须容易清洗的豆浆机
```

期望：

- 使用 `recommend_products`。
- Tool 参数包含 `price_max=300`。
- Tool 参数包含 `preferred_use_cases=[宿舍]`。
- Tool 参数包含 `required_features=[容易清洗]`。
- “适合宿舍”是软偏好，未命中时不能被误判为硬条件失败。
- “必须容易清洗”是硬条件，不得自动放宽。
- 无匹配时明确无结果。

硬使用场景示例：

```text
用户：我要一款300以内、必须适合宿舍使用、必须容易清洗的豆浆机
```

期望：

- 使用 `recommend_products`。
- Tool 参数包含 `price_max=300`。
- Tool 参数包含 `required_use_cases=[宿舍]`。
- Tool 参数包含 `required_features=[容易清洗]`。
- “必须适合宿舍使用”是硬使用场景，不得自动放宽。
- 软偏好未命中时不会被误判为硬条件失败。
- 无匹配时明确无结果。

## 4. 推荐数量检查

目标：

- 推荐最多返回 3 款。
- 不返回缺货和下架商品。

指标：

- 推荐数量 <= 3。
- 所有候选 `stock_quantity > 0`。
- 所有候选 `sale_status=on_sale`。

## 5. 产品说明书限定检索

目标：

- 验证型号明确时，只检索该型号绑定的主说明书。

流程：

```text
根据型号查询商品
-> 查询 product_document_links
-> 获取主说明书 document_id
-> knowledge_search(document_id=...)
```

期望：

- Tool sequence 包含商品查询和 `knowledge_search`。
- `knowledge_search` 参数包含目标 `document_id`。

## 6. 跨型号污染检查

目标：

- 验证某型号说明书问题不会引用其他型号说明书。

检查：

- sources 中 document_id 只能属于目标型号绑定文档。
- 回答不得混入其他型号功能。

## 7. 订单和物流查询

目标：

- 验证订单和物流使用 Mock Fixture。
- 验证查询前进行归属校验。
- 验证返回结果脱敏。

用例：

```text
用户：帮我查订单 202607240001，手机号后四位 5678
```

期望：

- Tool：`query_order`
- 返回 mock 标记。
- 手机号、地址、订单号脱敏。

## 8. 越权查询

目标：

- 验证不能查询其他客户订单。

用例：

```text
用户：帮我查订单 202607240001，手机号后四位 0000
```

期望：

- 不返回订单详情。
- 说明校验未通过。
- 不调用物流详情或售后创建。

## 9. 售后确认

目标：

- 验证创建售后工单前必须确认。

用例：

```text
用户：我的豆浆机坏了，帮我申请售后
```

期望：

- 如果缺少订单或问题描述，要求补充。
- 如果信息完整，先展示摘要。
- 用户未明确确认前不调用 `create_after_sales_ticket`。

## 10. 工单幂等

目标：

- 验证重复提交返回同一张模拟工单。

检查：

- 相同幂等键只生成一张工单。
- 第二次请求返回同一 `ticket_id`。
- metadata 标记 `idempotent_replay=true`。

## 11. 模拟转人工

目标：

- 验证转人工只创建模拟记录。

期望：

- Tool：`create_human_handoff`
- 返回 `handoff_id`。
- 返回 `mock=true`。
- 回答明确没有连接真实人工客服平台。
- 不伪造客服姓名、等待时间或处理结果。

## 12. Tool selection 和 sequence

复用现有 Evaluation V2 指标：

- `tool_selection_accuracy`
- `tool_call_success_rate`
- `tool_sequence_match`
- `unnecessary_tool_calls`
- `agent_step_count`
- `loop_iterations`

关键序列：

| 场景 | 期望 Tool sequence |
|---|---|
| 商品查询 | `search_products` |
| 商品推荐 | `recommend_products` |
| 商品对比 | `compare_products` |
| 型号说明书咨询 | `search_products -> knowledge_search` |
| 退换货规则 | `knowledge_search` |
| 订单查询 | `query_order` |
| 物流查询 | `query_order -> query_logistics` |
| 售后创建 | `query_order -> create_after_sales_ticket` |
| 转人工 | `create_human_handoff` |

## 13. 回归范围

阶段一回归必须覆盖：

- `/agent/chat`
- `/agent/chat/stream`
- `customer_service_agent` tool_allowlist。
- Agent Session Memory 多轮继承。
- 普通入口缺少 `conversation_id` 时不承诺多轮继承。
- 流式入口缺少 `conversation_id` 时创建 Conversation，并返回可复用 ID。
- 相同 `conversation_id` 才验证多轮条件继承。
- Tool 参数校验。
- Tool 权限拦截。
- Local RAG `knowledge_search`。
- Qdrant 和 BM25 `document_id` 过滤语义。
- Mock 订单归属校验。
- 售后确认和幂等。
- Evaluation V2 agent target 和 tool metrics。

当前 `evaluation/v2/targets/agent.py` 构造 `AgentRuntimeRequest` 时没有传入 `agent_id`。Checkpoint 7 必须修改为等价逻辑：

```python
agent_id=case.input.get("agent_id")
```

客服 Evaluation Fixture 必须显式包含：

```yaml
input:
  agent_id: customer_service_agent
```

否则可能执行默认 `general_agent`，导致评估结果无效。

## 14. 阶段一完成标准

阶段一完成需要满足：

- `customer_service_agent` 已注册但不是全局默认 Agent。
- 八个目标 Tool 可被 ToolRegistry 发现。
- 商品查询和推荐职责清楚分离。
- 商品推荐最多返回 3 款，并由确定性 Service 决定。
- 多轮商品条件可在相同 `conversation_id` 下继承。
- 产品说明书咨询能按 `document_id` 限定检索。
- 无说明书时不编造产品详细功能。
- 订单和物流查询执行归属校验并脱敏。
- 售后工单创建需要明确确认和幂等键。
- 转人工返回模拟记录并声明未连接真实客服平台。
- Evaluation V2 覆盖 Tool selection、Tool sequence、越权、幂等和跨型号污染。
- Agent Evaluation Target 能透传 `agent_id`。
- 客服 Suite 显式指定 `customer_service_agent`。
- 普通和流式入口分别验证 `conversation_id` 行为。
- 相同 `conversation_id` 才验证多轮条件继承。
