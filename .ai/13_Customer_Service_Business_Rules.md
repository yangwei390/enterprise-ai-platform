# 智能客服 Agent 阶段一业务规则

## 1. 模拟数据声明

阶段一是学习和架构验证环境：

- 商品价格是模拟价格。
- 商品库存是模拟库存。
- 商品热度是模拟热度。
- 订单是 Mock Fixture。
- 物流是 Mock Fixture。
- 售后工单是模拟工单。
- 转人工是模拟记录。
- 本系统不是九阳官方客服。
- 不连接真实商城、订单、物流、售后或人工客服平台。

Agent 每次涉及订单、物流、售后或转人工时，都必须避免暗示已经连接真实外部系统。

## 2. 商品查询规则

商品查询使用：

```text
search_products
```

支持参数：

```text
keyword
brand
category
price_min
price_max
required_features
excluded_features
preferred_features
required_use_cases
preferred_use_cases
in_stock_only
sale_status
sort_by
sort_order
page
page_size
```

规则：

- 默认只查 `is_active=true` 的商品。
- 用户没有明确要求时，默认 `sale_status=on_sale`。
- 用户没有明确要求时，默认 `in_stock_only=true`。
- 硬条件包括 `category`、`brand`、`price_min/price_max`、`required_features`、`excluded_features`、`required_use_cases`、库存、上下架状态。
- 软偏好包括 `preferred_features`、`preferred_use_cases`、容量、清洁、噪音、便携、家庭人数等。
- 只有用户明确说“必须”“只要”“不能接受”时，对应偏好才升级为硬条件。
- 软偏好参与推荐评分，未完全命中不能导致自动放宽硬条件。
- 如保留旧 `features` 参数，只能作为 `required_features` 的兼容别名，并由 Service 统一。
- 如保留旧 `use_cases` 参数，只能作为 `preferred_use_cases` 的兼容别名，并由 Service 统一转换。
- `required_use_cases` 是硬过滤，不得被自动放宽。
- `preferred_use_cases` 只参与评分。
- 用户明确说“必须适合宿舍”“只能用于宿舍”等，进入 `required_use_cases`。
- 用户只说“适合宿舍”“主要在宿舍使用”，默认进入 `preferred_use_cases`。
- 不允许自动放宽硬条件。
- 无结果时明确说明没有匹配商品，并提示用户可以调整条件。
- LLM 只能解释工具返回的商品，不能自由创造商品。

## 3. 商品推荐规则

商品推荐使用：

```text
recommend_products
```

只有用户提供以下信息之一时，才进入推荐：

- 使用场景。
- 预算偏好。
- 必须功能。
- 排除条件。
- 对人群、容量、清洁、噪音、便携等偏好。

推荐规则：

- 由确定性 Service 执行硬过滤和评分。
- 最多返回 3 款。
- 默认排除缺货商品。
- 默认排除下架商品。
- 不自动放宽用户硬条件。
- 没有结果时明确说明。
- 推荐理由必须来自商品结构化字段或说明书检索证据。
- LLM 只解释结果，不能创造候选商品。

## 4. 商品对比规则

商品对比使用：

```text
compare_products
```

规则：

- 只能对已查询到或用户明确指定的商品进行对比。
- 对比字段来自商品数据库和已关联说明书。
- 商品数据库字段优先用于价格、库存、型号、分类、规格、标签。
- 说明书只用于功能说明、操作方法、安全要求、维护和故障处理。
- 缺失字段必须标记为“当前资料未提供”，不能猜测。

## 5. 订单状态

订单查询使用：

```text
query_order
```

阶段一订单来自只读 Mock Fixture。

建议状态：

```text
created
paid
packed
shipped
delivered
cancelled
refunding
refunded
after_sales
```

返回内容：

- 订单号脱敏。
- 下单时间。
- 模拟订单状态。
- 商品摘要。
- 金额可返回模拟金额。
- 不返回完整手机号、完整地址或真实身份信息。

## 6. 订单归属校验

查询订单至少校验：

- 订单号。
- 模拟客户身份或手机号后四位。

规则：

- 参数不足时要求用户补充。
- 校验失败时不得返回订单详情。
- 不允许查询其他客户订单。
- 返回内容必须脱敏。
- 不在日志、metadata 或错误响应中保存完整手机号、地址或其他敏感信息。

## 7. 物流状态

物流查询使用：

```text
query_logistics
```

阶段一物流来自只读 Mock Fixture。

建议状态：

```text
pending
picked_up
in_transit
out_for_delivery
delivered
exception
returned
```

规则：

- 查询物流前必须校验订单归属。
- 不允许只凭快递单号查询其他客户物流。
- 物流节点中的地址、手机号、收件人必须脱敏。
- 物流异常只能描述 Mock Fixture 中存在的信息。

## 8. 退换货规则边界

退换货规则咨询使用：

```text
knowledge_search
```

规则：

- 基于 Local RAG 中的退换货规则文档回答。
- 没有检索到规则时明确说明当前资料不足。
- 不得编造官方政策。
- 不得承诺真实退款、真实换货或真实客服处理结果。
- 如果用户要发起售后，进入 `create_after_sales_ticket` 的确认流程。

## 9. 售后工单确认和幂等

创建售后工单使用：

```text
create_after_sales_ticket
```

必须满足：

- 校验订单存在。
- 校验订单归属。
- 校验订单状态是否允许模拟售后。
- 收集问题描述。
- 服务端生成稳定 `operation_id` / `draft_id`。
- 展示操作摘要。
- 获得用户明确确认。
- 使用幂等键。
- 重复请求返回同一张模拟工单。
- 不修改真实订单。

售后确认与幂等流程：

```text
收集并校验售后信息
-> 服务端生成稳定 operation_id / draft_id
-> 展示操作摘要
-> 用户明确确认
-> 使用该 ID 作为 idempotency_key 创建工单
-> 网络重试和重复提交继续复用同一个 key
```

幂等规则：

- `idempotency_key` 在模拟工单存储中唯一。
- 重复请求返回相同 `ticket_id`。
- metadata 标记 `idempotent_replay=true`。
- 不使用时间或时间桶生成幂等键。
- 用户实质修改售后内容时，旧摘要失效，生成新 draft 并重新确认。

阶段一确认语义：

- 用户明确表达“确认创建”“提交售后”“确认提交”等，才允许创建。
- 用户只是咨询退换货、问是否能退、描述问题，不等于确认创建。
- Mock 确认不是正式企业审批。未来真实写操作必须接入 Workflow Approval 或其他审批机制。

## 10. 投诉转人工

转人工使用：

```text
create_human_handoff
```

规则：

- 创建模拟转人工记录。
- 返回 `handoff_id`。
- 返回 `mock=true`。
- 明确没有连接真实人工客服平台。
- 不伪造客服姓名。
- 不伪造等待时间。
- 不伪造处理结果。

触发场景：

- 用户明确要求人工客服。
- 用户投诉且当前工具无法解决。
- 售后或订单校验失败后用户要求人工处理。

## 11. 敏感数据脱敏

必须脱敏：

- 手机号。
- 收件人姓名。
- 收货地址。
- 订单号。
- 物流单号。
- 工单描述中可能包含的身份证、银行卡、详细地址。

脱敏规则示例：

```text
手机号：138****5678
订单号：2026****7788
地址：广东省深圳市***街道
姓名：张*
```

## 12. 失败和无结果处理

商品无结果：

- 说明没有找到符合条件的商品。
- 给出可调整的条件方向。
- 不自动放宽价格、库存、上下架或必须功能。

说明书无结果：

- 说明当前知识库没有找到该型号的说明书资料。
- 不编造功能、操作步骤或安全要求。

订单无权限：

- 说明订单信息校验未通过。
- 不返回订单详情。

售后未确认：

- 展示摘要并等待用户明确确认。
- 不创建工单。

Tool 失败：

- 明确告诉用户当前能力暂时不可用。
- 不把失败描述成成功。
