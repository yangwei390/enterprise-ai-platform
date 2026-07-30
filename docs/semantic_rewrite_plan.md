# 通用记忆 + 语义改写正式方案

## 现状：已有什么

| 层 | 文件 | 状态 |
|---|---|---|
| 通用记忆 | context_builder.py + runtime.py | ✅ 已实现（3轮+摘要） |
| 结构化业务记忆 | dst.py + schemas.py | ✅ 已实现（candidate_history） |
| 本地规则-意图 | router.py → deterministic_route | ✅ 关键词匹配 |
| 本地规则-实体 | contextualizer.py → extract_* 系列 | ✅ 序数/价格/型号/属性 |
| 本地规则-约束 | customer_service.py → _trusted_constraint_operations | ✅ SET/REMOVE 检测 |
| 本地规则-引用 | customer_service.py → _validated_target_codes | ✅ 跨批次+品类过滤 |
| LLM 分类 | customer_service.py → _classify_customer_service_intent | ✅ tool_calling 结构化输出 |
| 三模式切换 | customer_service.py L2399-2535 | ✅ rule_only/hybrid/llm_only |
| 统一输出 | contextualizer.py → build_contextualized_request | ✅ ContextualizedRequest |
| 目标解析 | resolver.py → resolve_targets | ✅ 纯函数 |

## 方案总纲

核心原则：
- semantic_rewrite 只负责"用户说了什么"（语义提取）
- resolver 负责"指的是谁"（目标解析）
- 校验层负责"合不合法"（确定性验证）
- 确定性提取器（regex/枚举）是 explicit 的唯一来源
- LLM 输出永远是 inferred，不存在升级路径

## 核心链路

```
安全前置路由（注入拦截、售后待确认、确认/取消/修改、observation 后续）
             ↓
受控通用记忆（runtime 已恢复的有界 messages）+ ConversationDST
             ↓
规则解析：生成字段级 proposals
             ↓
按模式决定是否让 LLM 补充（llm_only 必调，hybrid 按条件调）
             ↓
合并 proposals（explicit 不可覆盖，inferred 允许 LLM 补充，冲突进追问）
             ↓
resolver.py 按业务域分发，使用可信 DST 候选历史解析真实目标
             ↓
统一确定性校验（目标存在、类别匹配、绑定关系）
             ↓
生成 ContextualizedRequest + 校验后生成 rewritten_query
             ↓
DST / FSM / DispatchPlan
             ↓
FSM guard（需要 ContextualizedRequest 和 DispatchPlan 作为输入）
             ↓
Tool 参数只读已验证请求和 DST
```

## 改动 1：新建 semantic_schemas.py（数据契约）

位置：backend/app/agents/customer_service_core/semantic_schemas.py

```python
class FieldSource(StrEnum):
    EXPLICIT = "explicit"    # 确定性提取器认定（regex/枚举匹配）
    INFERRED = "inferred"    # 关键词推断或 LLM 输出

class FieldProposal(BaseModel):
    """仅用于 intent/domain/target 这三个规则与 LLM 可能冲突的字段"""
    model_config = ConfigDict(extra="forbid")
    value: Any
    source: FieldSource
    # explicit 来源：regex 提取、枚举精确匹配
    # inferred 来源：关键词推断、LLM 输出（LLM 永远 inferred）

class TargetSemantics(BaseModel):
    """语义片段，不是解析结果。只提取用户说了什么，不解析指向谁。"""
    model_config = ConfigDict(extra="forbid")
    ordinal: int | None = None          # "第一款" → 0
    category_hint: str | None = None    # "鼠标" → "鼠标和指针设备"
    explicit_name: str | None = None    # "G304"（regex/名称匹配）
    explicit_code: str | None = None    # 商品编码（regex 提取）
    reference_text: str | None = None   # 原始引用文本

class ProductSemanticPayload(BaseModel):
    constraint_ops: ProductConstraintOperations = Field(default_factory=ProductConstraintOperations)
    attributes: list[str] = Field(default_factory=list)
    recommendation_count: int | None = None

class OrderSemanticPayload(BaseModel):
    action: OrderAction | None = None
    explicit_order_ref: str | None = None     # regex 提取 → 天然 explicit
    explicit_phone_last4: str | None = None   # regex 提取 → 天然 explicit

class AfterSalesSemanticPayload(BaseModel):
    issue_type: str | None = None
    order_ref: str | None = None
    confirmed: bool = False

class KnowledgeSemanticPayload(BaseModel):
    question: str | None = None

class HandoffSemanticPayload(BaseModel):
    order_ref: str | None = None
    reason: str | None = None

class SemanticParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 以下三个字段使用 FieldProposal（规则与 LLM 可能冲突）
    intent_proposal: FieldProposal | None = None
    domain_proposal: FieldProposal | None = None
    target_semantics: TargetSemantics | None = None
    # 以下字段来源单一（确定性提取器），不需要 FieldProposal 包装
    payload_proposal: (
        ProductSemanticPayload
        | OrderSemanticPayload
        | AfterSalesSemanticPayload
        | KnowledgeSemanticPayload
        | HandoffSemanticPayload
        | None
    ) = None
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    needs_llm: bool = False
```

explicit/inferred 判定规则：
- regex 提取（订单号、商品编码、价格、手机号）→ 天然 explicit，不需要 FieldProposal
- 枚举精确匹配（用户说"键盘"→ category="键盘"）→ 天然 explicit
- 关键词推断（"客服" → intent=HANDOFF）→ inferred
- LLM 任何输出 → 永远 inferred，无升级路径

FieldProposal 只用于 intent/domain/target 这三个"规则和 LLM 可能冲突"的字段。
其余字段来源单一，不存在冲突，不需要包装。

## 改动 2：新建 semantic_rewrite.py（语义提取入口）

位置：backend/app/agents/customer_service_core/semantic_rewrite.py

职责：接收 (raw_query, dst, messages) → 输出 SemanticParseResult
只做"用户说了什么"的提取，不做目标解析。

内部流程：
1. 调现有 deterministic_route → intent_proposal（关键词命中 → inferred）
2. 调现有 extract_* 系列 → payload 字段（regex → 天然 explicit）
3. 提取 TargetSemantics（序数、品类词、显式名称），不解析具体 product_code
4. 计算 needs_llm（见改动 3）

## 改动 3：三模式协作 + needs_llm 判定

```python
async def _run_semantic_pipeline(state, query, metadata, runtime_turn_id):
    dst = load_dst(metadata)
    messages = state["messages"]  # runtime 已恢复为有界上下文

    # 1. 规则语义提取
    parse = semantic_rewrite.parse(query, dst, messages)

    # 2. 按模式决定是否调 LLM
    mode = CustomerServiceIntentMode(settings.CUSTOMER_SERVICE_INTENT_MODE)
    if mode == CustomerServiceIntentMode.LLM_ONLY:
        # llm_only：必调 LLM，无论规则是否完整
        llm_result = await _classify_with_context(state, query, parse)
        parse = semantic_rewrite.merge(parse, llm_result)
    elif mode == CustomerServiceIntentMode.HYBRID and parse.needs_llm:
        # hybrid：满足条件时调 LLM 补充
        llm_result = await _classify_with_context(state, query, parse)
        parse = semantic_rewrite.merge(parse, llm_result)
    # rule_only：不调 LLM

    # 3. 按业务域分发目标解析（resolver.py）
    resolution = resolve_target(
        domain=parse.domain_proposal.value if parse.domain_proposal else None,
        semantics=parse.target_semantics,
        dst=dst,
    )

    # 4. 校验（追问文本由 Pipeline 根据失败原因生成）
    if resolution.clarification_required:
        question = _generate_clarification(resolution, parse, dst)
        return clarification(question)

    # 5. 生成 ContextualizedRequest + 校验后生成 rewritten_query
    request = build_contextualized_request(parse, resolution)
    request.rewritten_query = generate_rewritten_query(request, resolution)
    _commit_route(metadata, request, runtime_turn_id)
```

needs_llm 判定：

```python
def _compute_needs_llm(parse: SemanticParseResult) -> bool:
    if parse.gaps:
        return True
    if parse.conflicts:
        return True
    # 关键字段仅为 inferred → LLM 有挑战机会
    if parse.intent_proposal and parse.intent_proposal.source == FieldSource.INFERRED:
        return True
    if parse.domain_proposal and parse.domain_proposal.source == FieldSource.INFERRED:
        return True
    return False
```

## 改动 4：merge 策略

```python
def merge(rule: SemanticParseResult, llm: SemanticParseResult) -> SemanticParseResult:
    result = rule.model_copy(deep=True)
    for field_name in ("intent_proposal", "domain_proposal"):
        rule_proposal = getattr(rule, field_name)
        llm_proposal = getattr(llm, field_name)
        if rule_proposal is None:
            # 规则没提取到，用 LLM（仍标记 inferred）
            setattr(result, field_name, llm_proposal)
        elif rule_proposal.source == FieldSource.EXPLICIT:
            pass  # 确定性提取器认定，不可覆盖
        elif llm_proposal and rule_proposal.value != llm_proposal.value:
            # 规则 inferred + LLM 不同意 → 冲突，进追问
            result.conflicts.append(field_name)
        elif llm_proposal and rule_proposal.value == llm_proposal.value:
            pass  # 一致，保留
    # LLM 补充 gaps 中的字段
    for gap in rule.gaps:
        llm_value = getattr(llm, gap, None)
        if llm_value is not None:
            setattr(result, gap, llm_value)
    result.gaps = [g for g in result.gaps if getattr(result, g, None) is not None]
    return result
```

核心原则：
- 确定性提取器认定的 explicit → 不可覆盖
- 规则 inferred + LLM 一致 → 保留
- 规则 inferred + LLM 不同意 → 冲突，进追问（不盲目选任何一方）
- LLM 输出永远 inferred

## 改动 5：目标解析统一归 resolver.py，按域分发

```python
# resolver.py 扩展

def resolve_target(
    *,
    domain: CustomerServiceDomain | None,
    semantics: TargetSemantics | None,
    dst: ConversationDST,
) -> TargetResolution:
    if domain == CustomerServiceDomain.PRODUCT:
        return _resolve_product_target(semantics, dst)
    if domain in {
        CustomerServiceDomain.ORDER,
        CustomerServiceDomain.LOGISTICS,
        CustomerServiceDomain.AFTER_SALES,
        CustomerServiceDomain.HUMAN_HANDOFF,
    }:
        return _resolve_order_target(semantics, dst)
    return TargetResolution()

def _resolve_product_target(
    semantics: TargetSemantics | None,
    dst: ConversationDST,
) -> TargetResolution:
    product_domain = dst.domains.get(CustomerServiceDomain.PRODUCT)
    if product_domain is None or semantics is None:
        return TargetResolution(clarification_required=True)

    # 1. 确定候选池
    if semantics.category_hint:
        pool = _filter_by_category_recent_first(
            product_domain.candidate_history, semantics.category_hint,
        )
    else:
        pool = [
            CandidateHistoryEntry(
                ref=item.ref, display_name=item.display_name,
                category=None, batch_id="current", position=item.position,
            )
            for item in product_domain.candidates
        ]

    # 2. 解析
    if semantics.explicit_code:
        return _resolve_explicit(semantics.explicit_code, product_domain.candidate_history)
    if semantics.explicit_name:
        return _resolve_by_name(semantics.explicit_name, pool)
    if semantics.ordinal is not None:
        if semantics.ordinal >= len(pool):
            return TargetResolution(clarification_required=True, out_of_range=True)
        return TargetResolution(
            resolved_ids=[pool[semantics.ordinal].ref],
            source=TargetResolutionSource.ORDINAL,
        )
    if product_domain.active_ref and not semantics.category_hint:
        return TargetResolution(
            resolved_ids=[product_domain.active_ref],
            source=TargetResolutionSource.ACTIVE,
        )
    return TargetResolution(clarification_required=True)

def _resolve_order_target(
    semantics: TargetSemantics | None,
    dst: ConversationDST,
) -> TargetResolution:
    order_domain = dst.domains.get(CustomerServiceDomain.ORDER)
    if order_domain is None:
        return TargetResolution(clarification_required=True)
    if semantics and semantics.explicit_code:
        # 显式订单号
        return TargetResolution(
            resolved_ids=[semantics.explicit_code],
            source=TargetResolutionSource.EXPLICIT,
        )
    if semantics and semantics.ordinal is not None:
        if semantics.ordinal >= len(order_domain.candidates):
            return TargetResolution(clarification_required=True, out_of_range=True)
        return TargetResolution(
            resolved_ids=[order_domain.candidates[semantics.ordinal].ref],
            source=TargetResolutionSource.ORDINAL,
        )
    if order_domain.active_ref:
        return TargetResolution(
            resolved_ids=[order_domain.active_ref],
            source=TargetResolutionSource.ACTIVE,
        )
    return TargetResolution(clarification_required=True)

def _filter_by_category_recent_first(
    history: list[CandidateHistoryEntry],
    category: str,
) -> list[CandidateHistoryEntry]:
    """按 candidate_history 追加顺序倒序，返回最近相关批次的候选"""
    normalized = normalize_product_category(category)
    matched = [
        item for item in history
        if item.category is None or normalize_product_category(item.category) == normalized
    ]
    if not matched:
        return []
    # 按追加顺序最后一个匹配项的 batch_id = 最近批次
    recent_batch = matched[-1].batch_id
    recent_items = [item for item in matched if item.batch_id == recent_batch]
    return sorted(recent_items, key=lambda item: item.position)
```

序号语义："第一款鼠标" = 最近一个包含鼠标的批次中的第一款。
按 candidate_history 追加顺序倒序查找，不依赖 batch_id 格式排序。

现有 _validated_target_codes 迁移到 resolver.py 后删除，不保留两套。

## 改动 6：追问文本由 Pipeline 生成

TargetResolution 不包含 question 字段（保持现有 Schema 不变）。
追问文本由 Pipeline 根据失败原因统一生成：

```python
def _generate_clarification(
    resolution: TargetResolution,
    parse: SemanticParseResult,
    dst: ConversationDST,
) -> str:
    if resolution.out_of_range:
        domain = parse.domain_proposal.value if parse.domain_proposal else None
        count = _candidate_count(dst, domain)
        return f"当前只有 {count} 个候选，请选择有效序号。"
    if parse.conflicts:
        return "我暂时无法确定您的具体需求，请说明商品名称、订单号或具体问题。"
    return "请说明您要查询的具体商品或订单。"
```

## 改动 7：安全前置路由（语义改写之前）

以下逻辑必须在语义改写之前执行，重构时保留原有位置：

```python
# customer_service.py adecide() 执行顺序：

# ===== 安全前置路由 =====
# 1. Prompt 注入拦截
if _is_prompt_injection(query):
    return _final("我不能忽略系统规则或绕过工具确认流程。")

# 2. 售后待确认状态（确认/取消/修改）
pending = _pending_after_sales(metadata)
if pending is not None:
    ...  # CONFIRM/CANCEL/MODIFY/AMBIGUOUS 处理

# 3. 已有 Tool observation 的后续处理
if _last_tool_name(observations) == "search_products":
    return _manual_followup_decision(state, observations[-1])
# ... 其他 observation 处理

# ===== 语义改写 Pipeline =====
await _run_semantic_pipeline(state, query, metadata, runtime_turn_id)

# ===== 语义改写之后 =====
# 4. 生成 ContextualizedRequest + DispatchPlan
contextualized_request = _current_contextualized_request(metadata, runtime_turn_id)
dispatch_plan = _current_dispatch_plan(metadata, contextualized_request)

# 5. FSM guard（需要 ContextualizedRequest 和 DispatchPlan 作为输入）
fsm_guard = _fsm_guard_decision(contextualized_request, dispatch_plan)
if fsm_guard is not None:
    return fsm_guard

# 6. Tool 执行
...
```

## 改动 8：rewritten_query 在校验后生成

```python
def generate_rewritten_query(request: ContextualizedRequest, resolution: TargetResolution) -> str:
    # 基于校验后的数据生成，不信任 LLM 的 rewritten_query
    if isinstance(request.payload, ProductPayload) and resolution.resolved_ids:
        target_name = _display_name_for_code(resolution.resolved_ids[0]) or "该商品"
        if request.payload.attributes:
            return f"查询 {target_name} 的 {'、'.join(request.payload.attributes)}"
        return f"查询 {target_name} 的相关信息"
    return request.raw_query
```

LLM 的 rewritten_query 只作为调试数据，不进入下游。

## 改动 9：编排逻辑从 customer_service.py 抽离

把 L2399-2535（模式切换）+ L3025-3189（_store_customer_service_route）收敛到 _run_semantic_pipeline。
customer_service.py 的 adecide() 只保留：安全前置路由 → 语义 Pipeline → FSM guard → 执行分发。

## 前置工程步骤

新建 semantic_schemas.py 和 semantic_rewrite.py 之前：
- 更新 customer_service_core 目录的职责说明（如有 AGENTS.md 则补充，无则在 .ai/ 规范中记录）
- 明确两个新文件的职责边界

## 不动的部分

- context_builder.py：不改
- dst.py：不改
- fsm.py / compatibility.py / normalization.py：不改
- schemas.py 中现有 TargetResolution：不加 question 字段
- 数据库 / .env / Runtime 框架：不动

## 执行顺序

| 步骤 | 内容 | 验证 |
|------|------|------|
| 0 | 更新目录职责说明 | — |
| 1 | 新建 semantic_schemas.py | ruff + pyright |
| 2 | 新建 semantic_rewrite.py，收拢 extract_* 为 parse() | 专项测试 |
| 3 | 扩展 resolver.py，按域分发 + 品类过滤 + 最近批次优先 | 专项测试 |
| 4 | 迁移 _validated_target_codes → resolver.py，新旧结果对比 | 现有跨批次测试不回归 |
| 5 | 删除旧 _validated_target_codes | 全量测试 |
| 6 | 实现 merge（LLM 永远 inferred，冲突进追问） | 专项测试 |
| 7 | 改三模式协作逻辑（llm_only 必调、hybrid 按 needs_llm） | 三模式参数化测试 |
| 8 | LLM prompt 增加 rule_result + gaps，输出永远标记 inferred | 专项测试 |
| 9 | 补全订单/售后/转人工/知识库 payload 提取 | 专项测试 |
| 10 | customer_service.py 编排瘦身（安全前置保留，FSM guard 后置） | 全量测试 |
| 11 | 集成测试 + 回归 | 全量测试 + ruff + pyright |

每步跑专项测试，不全留到最后。删除旧代码前先确认新旧结果一致。

## 测试范围

| 场景 | 覆盖点 |
|------|------|
| 三种模式行为差异 | rule_only 不调 LLM；llm_only 必调；hybrid 按条件调 |
| LLM 异常 | 超时、非法 JSON → 降级为规则结果 |
| LLM 伪造 explicit | LLM 声称 explicit → 系统仍视为 inferred |
| 商品跨批次引用 | "第一款鼠标"→最近鼠标批次第一个 |
| 订单跨批次引用 | "第二个订单"→正确解析 |
| 跨域订单引用 | "第二个订单的物流"、"给第一个订单申请售后" |
| 品类切换 | 键盘→鼠标，旧 slot 兼容推导 |
| 跨业务打断与恢复 | 商品咨询中途插入售后，再回到商品 |
| 售后确认/取消/修改 | 安全前置路由正确处理 |
| Prompt 注入 | 拦截在语义改写之前 |
| 规则与 LLM 冲突 | 冲突后必须追问 |
| explicit 不可覆盖 | 确定性提取器认定的值，LLM 不能改 |
| inferred 可补充 | 规则推断错误时 LLM 可以纠正（一致则保留，不一致则追问） |
| FSM guard 顺序 | 在 ContextualizedRequest 生成之后执行 |

## 风险点

1. 步骤 10 是大改——建议步骤 1-9 先做稳，每步验证，步骤 10 最后做。
2. 步骤 4-5 迁移 _validated_target_codes 是高风险操作，必须先跑新旧对比。
3. _filter_by_category_recent_first 按追加顺序倒序，不依赖 batch_id 格式。
