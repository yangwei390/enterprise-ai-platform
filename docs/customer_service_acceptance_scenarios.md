# 智能客服场景与验收测试表（Review 稿）

## 1. 验收口径

每个场景必须同时验证：

- Query 补全是否调用、输入和输出；
- 意图、槽位、Tool 和引用阶段分别使用规则还是 LLM；
- Resolver 使用的候选批次；
- 最终 Tool 名称和强类型参数；
- Result 契约与状态 Commit；
- 最终输出是否由 Presenter 产生；
- 真实 LLM 调用次数；
- 失败时可信业务状态是否保持不变。

以下表格中的“重量 LLM”表示产生候选，不表示它拥有最终 Tool 执行权。

## 2. 商品目录与推荐

| ID | 前置状态 | 用户输入 | 预期理解与分支 | Tool及关键参数 | 状态/输出 |
|---|---|---|---|---|---|
| P01 | 空 | 推荐个鼠标 | 商品推荐；count默认1 | `recommend_products(keyword="鼠标", page_size=1)` | 提交一个鼠标批次；模板输出1项 |
| P02 | 空 | 给我推荐2个鼠标 | 商品推荐；count=2 | `recommend_products(..., page_size=2)` | 最多输出2项 |
| P03 | 空 | 有鼠标么 | 商品可用性查询 | `search_products(keyword="鼠标")` | 有则列目录；无则输出缺货模板 |
| P04 | 空 | 推荐个键帽 | 商品推荐，关键字必须存在 | `recommend_products(keyword="键帽", page_size=1)` | 0结果输出指定缺货模板，不泛查全库 |
| P05 | 当前鼠标G304 | 换一个 | 继承鼠标品类，排除已展示商品 | 推荐Tool，排除编码由Python注入 | 返回未展示鼠标；新批次替代旧批次 |
| P06 | 鼠标已无其他结果 | 别的呢 | Query补全为其他鼠标；continuation | 推荐Tool | 0结果输出指定缺货模板，不重复旧商品 |
| P07 | 当前鼠标 | 推荐个键盘 | 明确切换品类 | 推荐键盘 | 清空鼠标上下文，只保存键盘 |
| P08 | 当前键盘 | 第一个鼠标多少钱 | 旧鼠标不可引用；安全重查鼠标 | 商品Tool，count=1 | 文案先说明不确定旧款，再展示新查鼠标 |
| P09 | 空 | 你家卖鼠标么 | 简单完整句 | 商品搜索Tool | 不得落入通用澄清 |
| P10 | 空 | 推荐一个不存在品类 | 商品推荐 | 推荐Tool带原始关键字 | 0结果输出指定缺货模板 |

缺货统一文案：

> 抱歉呢亲亲，小店目前还没有上架 【{商品名称/类型}】 这类宝贝呢 。
>
> 非常感谢您的咨询，我已经把您的需求记在小本本上反馈给采购部门啦！要不您看看咱们店的主打宝贝？或者告诉我您的具体用途，我帮您在现货里挑挑看有没有能替代的合适好物~

## 3. 商品引用与说明书

| ID | 前置状态 | 用户输入 | LLM候选 | Resolver/RAG | 预期结果 |
|---|---|---|---|---|---|
| M01 | 当前鼠标批次2项 | 第二个能连接蓝牙吗 | ordinal=1；manual question | position=1；商品验证→绑定文档RAG | 回答第二款，带来源和引用 |
| M02 | M01成功 | 第一个呢 | ordinal=0；继承蓝牙谓词 | position=0；独立RAG | 不继承第二款答案 |
| M03 | 当前唯一商品 | 它能连接蓝牙么 | pronoun=active product | Resolver确认唯一当前商品 | 必须进入说明书链，不重复目录 |
| M04 | 当前唯一商品 | 可以打游戏吗 | manual question | 统一进入绑定说明书RAG | 不使用目录用途字段直接回答；无证据则不做断言 |
| M05 | 当前唯一商品 | 保修多长时间 | manual question | 限定绑定说明书 | 有证据才回答 |
| M06 | 无当前商品 | 它保修多久 | 代词无法落地 | Resolver失败 | 追问具体商品，不调用RAG |
| M07 | 原候选回复仍在且无新批次 | 多轮后问第一个能充电吗 | ordinal=0 | Resolver继续使用当前批次 | 不采用固定轮次TTL |
| M08 | 原候选回复已被裁剪 | 第一个能充电吗 | ordinal=0 | Resolver拒绝不可见引用 | 追问或安全重查 |
| M09 | 商品无绑定说明书 | 它支持蓝牙吗 | manual question | 商品验证成功，绑定缺失 | 不回答断言；进入澄清/升级 |
| M10 | 有绑定文档，RAG无citation | 它支持蓝牙吗 | manual question | Evidence Gate失败 | 不输出“支持/不支持” |
| M11 | RAG证据完整 | 它支持蓝牙吗 | manual question | Gate通过 | 确定性Presenter；Final LLM=0 |

## 4. 订单与物流

| ID | 前置状态 | 用户输入 | 预期理解 | Tool | 状态/输出 |
|---|---|---|---|---|---|
| O01 | 空 | 我的订单到哪了 | 物流意图但无唯一订单 | `query_order(None)`先列订单 | 提交订单批次 |
| O02 | 订单批次2项 | 第二个查看详情 | order ordinal=1，detail | `query_order(resolved_ref)` | 提交active_order_ref |
| O03 | 订单批次2项 | 第二个到哪了 | logistics ordinal=1 | `query_logistics(resolved_ref)` | 模板输出物流 |
| O04 | 空 | 查订单A123 | 显式订单号 | 先`query_order(A123)`验证归属 | 验证成功才提交 |
| O05 | 当前订单 | 它到哪了 | active order引用 | 物流Tool | 不重新列所有订单 |
| O06 | 订单批次失效 | 第二个呢 | 过期序数 | 无 | 追问/重新列订单，不猜测 |

## 5. 售后、高风险与转人工

| ID | 前置状态 | 用户输入 | Python门禁 | Tool | 预期状态 |
|---|---|---|---|---|---|
| A01 | 已验证订单 | 我要退货，原因… | 槽位完整 | `after_sales_draft` | 提交draft_id/operation_id |
| A02 | 有售后草稿 | 确认 | 明确确认+幂等校验 | `after_sales_confirm` | 成功后清除草稿 |
| A03 | 有售后草稿 | 取消 | 明确取消 | 不调用confirm | 清除或标记取消 |
| A04 | 有售后草稿 | 换个鼠标 | 新意图，不是确认 | 商品Tool | 不误提交售后；草稿保留但暂停确认资格 |
| A05 | 售后草稿被打断 | 确认 | 模糊确认不可提交 | 无 | 展示当前售后草稿并要求明确确认提交 |
| A06 | 无草稿 | 确认 | 无有效待确认操作 | 无 | 提示没有待确认操作 |
| A07 | 空 | 转人工 | 转人工意图 | `create_human_handoff` | 参数Schema和权限通过才执行 |
| A08 | 同一澄清失败2次 | 仍然无法说明 | 达升级上限 | handoff | 确定性告知升级；当前不创建通用工单 |

## 6. Query补全与独立开关

| ID | 开关组合 | 输入 | 预期调用 |
|---|---|---|---|
| Q01 | 五项全部=`llm_only` | 有鼠标么 | 轻量LLM补全（可原样返回）；重量LLM完成后续候选判断 |
| Q02 | 五项全部=`llm_only` | 别的呢 | 轻量LLM补成“还有其他鼠标吗”；重量LLM完成后续候选判断 |
| Q03 | 五项全部=`llm_only` | 第二个呢 | 重量LLM给出引用表达候选；Resolver验证真实实体 |
| Q04 | 任意 | Prompt注入 | Safety先阻断，Query LLM不调用 |
| Q05 | Query LLM失败 | 别的呢 | 记录失败；不得伪造补全，进入澄清 |
| Q06 | 五项全部=`llm_only` | 你家卖键盘么 | 重量LLM输出商品业务意图、槽位和能力候选 |
| Q07 | 五项全部=`llm_only` | 推荐个游戏用的 | 重量LLM输出用途候选；不输出真实商品；最终走说明书规则 |
| Q08 | 五项全部=`llm_only` | 查询完整商品请求 | 重量LLM选择原生Function候选 |
| Q09 | 五项全部=`llm_only` | 第二个呢 | 重量LLM输出ordinal=1；Resolver再验证 |
| Q10 | 五项全部=`llm_only` | 同一完整请求 | 每项执行详情准确显示LLM来源和真实调用次数 |

说明：Q01-Q10 中五项 `llm_only` 是五个逻辑阶段。普通回合只发生两次真实模型调用：
Query 补全使用轻量模型一次；意图、槽位、引用和能力选择共享重量模型的一次 Function
调用，四项执行详情使用同一个 `shared_call_id`。

## 7. Tool Schema和保护参数

| ID | LLM候选 | 预期校验 |
|---|---|---|
| T01 | `recommend_products`缺keyword | Pydantic拒绝，不调用真实Tool |
| T02 | LLM填写`knowledge_base_id` | 删除/拒绝保护参数，由Python可信范围注入 |
| T03 | LLM填写`excluded_product_codes` | 不信任该值，由当前批次Python计算 |
| T04 | LLM输出不存在Tool | 白名单拒绝，转澄清 |
| T05 | LLM为只读请求选择写Tool | 风险策略拒绝 |
| T06 | 显式商品编码不在当前池 | 先调用商品Tool验证，不直接信任 |
| T07 | 显式订单号未验证 | 先`query_order`，不得直接查物流/售后 |
| T08 | Tool参数通过候选Schema但不通过真实args_schema | 不调用Tool，不提交状态 |

## 8. 失败不污染与Presenter

| ID | 故障点 | 预期结果 |
|---|---|---|
| F01 | Query补全超时 | 业务状态不变，记录真实LLM失败 |
| F02 | 重量LLM无Tool call | 转澄清，业务状态不变 |
| F03 | Resolver多候选 | 追问，不能默认第一条 |
| F04 | 权限失败 | Tool不执行，状态不变 |
| F05 | Tool抛异常 | PendingTransaction rejected，状态不变 |
| F06 | Tool返回success但Result字段缺失 | Result契约拒绝，状态不变 |
| F07 | 商品Tool成功 | Commit后Presenter读取结构化结果 |
| F08 | 流式客服Final | 发送Presenter文本delta，Final LLM=0 |
| F09 | 普通商品轮遥测 | 只统计实际发生的LLM调用 |
| F10 | CONTINUE第二Tool | 不再次调用Understanding，复用GoalSnapshot |

## 9. 上下文裁剪

| ID | 原始对话 | 预期上下文 |
|---|---|---|
| C01 | 多轮对话超窗口 | 从完整user轮次开始，不出现孤立assistant |
| C02 | 当前raw_query已单独传入 | `recent_dialogue`不重复当前用户消息 |
| C03 | 商品消息被裁剪 | 对应批次立即失去序数可见性 |
| C04 | 历史摘要提到旧商品 | 只能辅助Query补全，不能恢复可信product_code |
| C05 | Tool历史结果被压缩 | 当前可信业务状态不受文本裁剪影响，但过期序数不可用 |

## 10. 自动化测试分层

### 10.1 单元测试

- 五个 Mode 策略；
- Query补全输出契约；
- 意图、槽位和引用候选Schema；
- 商品/订单批次有效性；
- Predicate继承白名单；
- PendingClarification状态转移；
- Tool候选参数和保护参数；
- Evidence Gate与Presenter。

### 10.2 组件测试

- Understanding → Resolver → Command；
- Command → Adapter → 真实args_schema；
- Tool Result → Commit；
- 商品验证 → 绑定说明书RAG；
- 售后draft → 跨轮confirm。

### 10.3 端到端回归

至少固定以下会话：

```text
推荐鼠标 → 换一个 → 别的呢 → 推荐键盘 → 第一个鼠标多少钱
```

```text
推荐两个鼠标 → 第二个能连接蓝牙吗 → 第一个呢
```

```text
我的订单到哪了 → 第二个查看详情 → 它到哪了
```

```text
申请退货 → 补齐槽位 → 生成草稿 → 确认
```

## 11. 完成标准

实施完成必须同时满足：

1. 所有确认场景通过；
2. Pyright backend 0 errors；
3. Ruff通过；
4. 客服、API、异步运行时、Tool、记忆和LangGraph关键回归通过；
5. 流式Final真实LLM调用为0；
6. 不修改`.env`、数据库Schema、CI/CD和生产配置；
7. 不提交、不推送；
8. 输出可删除代码清单，但删除前另行取得老大确认。

## 12. 已确认决策

1. 五个独立开关全部使用 `llm_only`，接受完整句也调用对应 LLM；
2. “能否打游戏”等用途能力问题一律走说明书 RAG；
3. 售后草稿被其他意图打断后保留，但必须重新明确确认；
4. 候选批次不采用固定轮次 TTL；
5. 原候选消息被裁剪后，对应序数立即失效。

## 13. 后续仍需单独确认的实施项

1. 连续澄清失败当前采用仅转人工；如需通用咨询工单，是否另立 Tool 设计任务。
