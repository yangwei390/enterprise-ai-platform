# 智能客服可信状态契约（Review 稿）

## 1. 状态分层

```text
有界原始对话记忆
  用途：Query 补全和语义理解
  可信度：非业务事实

metadata["customer_service"]["state"]
  用途：跨轮可信业务状态
  可信度：只有 Tool 成功并通过 Result 契约后提交的字段可信

AgentState.customer_service_execution
  用途：当前轮事务、阶段和两 Tool 连续执行
  生命周期：单轮，不持久化
```

禁止使用历史回复文本恢复商品编码、订单号、说明书绑定或 Tool 参数。

## 2. 根状态建议

```python
class CustomerServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductContext = Field(default_factory=ProductContext)
    order: OrderContext = Field(default_factory=OrderContext)
    dialog_focus: DialogFocus = Field(default_factory=DialogFocus)
    pending_after_sales: PendingAfterSales | None = None
    pending_clarification: PendingClarification | None = None
```

现有 `order_candidates`、`active_order_ref`、`clarification_target` 和
`clarification_rounds` 通过一次兼容读取迁移到强类型子状态；正式新逻辑只读新结构。

## 3. ProductContext

```python
class ProductContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_category: str | None = None
    filters: ProductFilters = Field(default_factory=ProductFilters)
    active_batch: ProductCandidateBatch | None = None
    active_product_code: str | None = None
    last_question: ProductQuestionFocus | None = None
    pending_query: PendingProductQuery | None = None
```

### 3.1 字段契约

| 字段 | 来源 | 写入条件 | 失效条件 | 容量/生命周期 |
|---|---|---|---|---|
| `active_category` | 用户显式表达经标准化，或商品 Tool 返回类别 | 商品 Tool 成功并提交；创建可恢复查询时可写入待确认区，不直接成为商品事实 | 用户明确切换品类、业务结束、超时 | 只保留最新品类 |
| `filters` | 用户显式约束或经受控语义提取的候选，最终由 Command 校验 | 对应商品 Tool 成功后提交有效过滤条件 | 品类切换时整体清空；用户明确取消字段时删除 | 有界强类型字段，不使用任意业务字典作为长期事实 |
| `active_batch` | 商品 Tool Result | Result 契约通过后提交 | 新批次替代、品类切换、TTL 到期、原可见轮次被裁剪 | 只保留一个当前批次，最多20项 |
| `active_product_code` | Resolver 在有效批次中选中，或显式编码经 Tool 验证 | 验证成功后提交 | 品类切换、批次失效、商品不再属于当前批次 | 0或1个 |
| `last_question` | 已成功承接的用户问题谓词 | 商品已验证且问题进入有效执行链 | 品类切换、问题不可继承、TTL 到期 | 只保留最近一个可继承问题 |
| `pending_query` | 系统追问时生成的可恢复动作 | 已明确需要重新查询某品类但等待用户确认 | 用户确认后消费、拒绝、切换意图、两轮超限 | 0或1个 |

### 3.2 商品批次

```python
class ProductCandidateBatch(BaseModel):
    batch_id: str
    query: str
    category: str | None
    source_turn_id: str | None
    items: list[CandidateProduct] = Field(max_length=20)

class CandidateProduct(BaseModel):
    product_code: str
    name: str
    category: str | None
    batch_id: str
    position: int
    primary_manual_document_id: int | None
```

候选批次必须来自商品 Tool。`primary_manual_document_id` 只是商品 Tool 返回的绑定关系，
真正调用 RAG 时仍需校验允许的知识库范围。

### 3.3 当前可见批次定义

一个商品批次同时满足以下条件才可解析序数：

1. 它等于 `product.active_batch`；
2. 用户当前表达的品类与 `active_category` 一致，或当前表达没有切换品类；
3. 产生该批次的客服回复仍在原始消息窗口中；
4. 没有被新批次替代；
5. 没有因业务结束或明确切换领域被清除。

已确认：不设置固定轮次 TTL。只要没有查询新批次且原候选回复仍在记忆窗口中，批次
继续有效；原候选回复一旦被裁剪，对应序数立即失效。

### 3.4 品类切换原则

当用户明确从鼠标切换到键盘：

- 清空旧鼠标 `active_batch`、`active_product_code`、`filters`、`last_question`；
- 查询并提交最新键盘批次；
- 不在正式商品状态中保留旧鼠标候选。

用户之后问“第一个鼠标多少钱”时，不允许解析旧鼠标。系统应直接重新查询一个鼠标，
并使用类似文案：

> 抱歉，我目前不确定您询问的是哪款鼠标，现在为您推荐以下鼠标。

如果需要用户确认后再查，则把动作写入 `pending_query`；当前已确认的产品行为是可以直接
安全查询时优先查询，不增加无意义追问。

## 4. OrderContext

```python
class OrderCandidate(BaseModel):
    order_ref: str
    display_label: str
    batch_id: str
    position: int

class OrderCandidateBatch(BaseModel):
    batch_id: str
    source_turn_id: str | None
    items: list[OrderCandidate] = Field(max_length=20)

class OrderContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_batch: OrderCandidateBatch | None = None
    active_order_ref: str | None = None
```

| 字段 | 来源 | 写入条件 | 失效条件 | 容量/生命周期 |
|---|---|---|---|---|
| `active_batch` | `query_order` 的成功结果 | Result 契约和订单归属验证通过 | 新订单批次、原回复不可见、业务结束 | 一个批次，最多20项 |
| `active_order_ref` | 显式订单号验证或当前批次 Resolver | 订单归属验证成功 | 批次失效、用户切换订单、权限变化 | 0或1个 |

LLM 可以提取“第二个订单”的 `ordinal=1`，但只有 Resolver 能从有效订单批次取得真实
`order_ref`。显式订单号也必须先通过 `query_order` 验证归属。

## 5. DialogFocus

```python
DialogDomain = Literal[
    "general",
    "product",
    "manual",
    "order",
    "logistics",
    "after_sales",
    "handoff",
]

class DialogFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_domain: DialogDomain | None = None
    active_action: str | None = None
    active_batch_id: str | None = None
    last_successful_predicate: ProductQuestionPredicate | None = None
    source_turn_id: str | None = None
```

| 字段 | 来源 | 写入条件 | 失效条件 |
|---|---|---|---|
| `active_domain` | 已通过路由并成功承接的业务域 | Tool 成功，或进入明确澄清 | 用户明确切换域、业务结束、对应原始记忆丢失 |
| `active_action` | 已成功承接的动作 | Command 成功生成；写操作在草稿生成后提交 | 动作完成、取消、切换意图 |
| `active_batch_id` | 商品或订单 Tool Result | Result 提交时写入 | 对应批次失效 |
| `last_successful_predicate` | 用户已成功提出且系统进入正确证据链的问题 | 目标商品已验证 | 品类/领域切换、不可继承谓词、对应原始记忆丢失 |
| `source_turn_id` | 当前 Graph 业务轮次标识 | Focus 写入时 | Focus 清除 |

DialogFocus 记录“对话进行到哪里”，不记录上一实体的答案。

## 6. Predicate 继承

### 6.1 允许继承

- 蓝牙/连接能力；
- 充电能力；
- 兼容性；
- 价格；
- 商品特征；
- 按键；
- 说明书中的使用方法；
- 订单详情；
- 物流状态。

示例：

```text
第二个能连接蓝牙吗？
第一个呢？
```

第二轮只继承问题谓词“是否支持蓝牙”，不能继承第二个商品的答案。所有“能否打游戏”
等用途能力问题，继承后仍统一进入说明书证据链。

### 6.2 禁止继承

- 售后原因、手机号后四位；
- 写操作确认或取消；
- 转人工原因；
- 用户评价、情绪或偏好结论；
- 已经失效批次对应的问题；
- 跨领域问题，例如从物流切到商品后继承“到哪了”。

## 7. PendingClarification

```python
ClarificationKind = Literal[
    "missing_slot",
    "ambiguous_reference",
    "confirm_safe_query",
    "missing_evidence",
    "confirm_write",
]

class ResumeAction(BaseModel):
    action: str
    domain: DialogDomain
    safe_slots: dict[str, JsonValue] = Field(default_factory=dict)

class PendingClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarification_id: str
    kind: ClarificationKind
    domain: DialogDomain
    target_description: str
    missing_fields: list[str]
    resume_action: ResumeAction | None = None
    attempts: int = Field(default=1, ge=1, le=2)
    created_turn_id: str | None
    last_asked_turn_id: str | None
```

约束：

- `safe_slots` 只能保存用户明确表达且通过字段 Schema 的值，不保存 LLM 猜测的实体事实；
- 肯定回复只恢复当前有效 `resume_action`；
- 写操作确认继续使用 `PendingAfterSales` 的 `draft_id + operation_id`，不能只靠
  `PendingClarification`；
- 同一目标第二次仍失败后清除 Pending 并升级；
- 新意图与 Pending 的域或动作冲突时，清除或暂停 Pending，不隐式确认。

## 8. PendingAfterSales

继续使用现有字段：

```python
class PendingAfterSales(BaseModel):
    draft_id: str
    operation_id: str
    order_no: str
    customer_phone_last4: str
    status: str
```

写入条件：`after_sales_draft` Tool 成功且 Result 契约通过。

清除条件：

- `confirm` 成功；
- 用户明确取消；
- 草稿失效；
- 幂等校验拒绝后按错误策略终止。

任何 LLM 输出的“用户已确认”都不能代替当前轮明确确认文本和 Python 门禁。

售后草稿被商品、订单等其他意图打断时保留；恢复售后时必须重新向用户展示当前草稿
对象，并要求明确确认，不能把打断后的普通肯定词直接解释为提交。

## 9. 确定性状态转移表

| 前置状态 | 用户输入 | 语义候选 | Python 决策 | Tool | 提交结果 |
|---|---|---|---|---|---|
| 当前鼠标批次2项 | “第二个能连接蓝牙吗” | 商品序数1、蓝牙谓词 | Resolver 取当前批次 position=1 | 商品验证→限定说明书RAG | 提交选中商品与 Focus；不保存答案 |
| 上轮蓝牙问题成功 | “第一个呢” | 商品序数0、省略谓词 | 继承蓝牙谓词；Resolver 取 position=0 | 限定说明书RAG | 更新 active product，不继承旧答案 |
| 当前订单批次2项 | “第二个查看详情” | 订单序数1、详情动作 | Resolver 取 order position=1 | `query_order` | 提交 active_order_ref |
| 当前鼠标状态 | “推荐个键盘” | 新品类键盘 | 清空鼠标结构化状态 | 商品 Tool | 只提交键盘批次 |
| 当前键盘状态 | “第一个鼠标多少钱” | 新品类鼠标、序数、价格 | 旧鼠标已失效，序数不可用；安全重查一个鼠标 | 商品 Tool | 新鼠标成为唯一当前批次 |
| Pending=重新查询鼠标 | “需要” | 肯定回复 | 恢复 `resume_action=search mouse` | 商品 Tool | 消费 Pending，提交鼠标批次 |
| 原商品回复仍在且无新批次 | 多轮后问“第一个多少钱” | 序数、价格 | Resolver继续解析当前批次 | 商品事实Tool | 不因固定轮次失效 |
| 原商品回复已被裁剪 | “第一个多少钱” | 序数、价格 | Resolver拒绝不可见引用 | 无或安全重查 | 不使用丢失记忆中的商品 |
| RAG首次无证据 | “它保修多久” | 说明书问题 | 保存 missing_evidence 澄清 | 无 | 商品状态不变 |
| 同目标第二次无证据 | 补充后仍无法回答 | 同目标 | 清除 Pending 并升级 | handoff | 不提交虚构答案；不得用售后Tool冒充通用工单 |
| 售后草稿存在 | “确认” | 确认表达 | 校验 draft_id/operation_id/幂等 | confirm Tool | 成功后清除草稿 |
| 售后草稿存在 | “换个鼠标” | 新商品意图 | 不把它当售后确认 | 商品 Tool | 售后草稿保留并暂停确认资格 |

## 10. Commit 规则

| 阶段失败 | 是否更新可信业务状态 |
|---|---|
| Query 补全失败 | 否 |
| 意图/槽位/引用候选失败 | 否 |
| Resolver 失败 | 否；只允许写 PendingClarification |
| Command Schema 失败 | 否 |
| 权限失败 | 否 |
| Tool Schema 失败 | 否 |
| Tool 执行失败 | 否 |
| Result 契约失败 | 否 |
| RAG 证据门禁失败 | 不提交答案；允许更新澄清计数 |
| Tool成功且Result通过 | 由 CommitCoordinator 原子提交 |

## 11. 已确认决策

1. 商品和订单候选不采用固定轮次 TTL；
2. 批次对应的原始客服消息被裁剪后，序数立即失效；
3. 售后草稿被其他意图打断时保留，恢复时重新明确确认；
4. “能否打游戏”等用途能力问题统一进入说明书；
5. 五个独立语义阶段全部使用 `llm_only`；它表示各逻辑阶段由模型提供候选，不表示
   调用五次模型。普通回合实际为一次轻量 Query 补全和一次共享重量 Function 调用。

## 12. 后续仍需单独确认的实施项

1. `ResumeAction.safe_slots` 是否改成各业务域的判别联合类型，避免任何字典；
2. 重新进入旧品类时采用“直接安全查询一个”还是“先询问是否重查”；本文按照此前讨论，
   采用可安全查询时直接查询。
