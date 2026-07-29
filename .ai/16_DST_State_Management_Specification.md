# 16 - DST 状态管理规范

> 本文件是智能客服 Agent 对话状态追踪（DST）的开发规范。
> 所有涉及 slot 合并、条件继承、Tool 参数组装的改动，必须遵守本规范。

---

## 1. 核心原则

- **slot 级裁决**：合并以单个 slot 为粒度，禁止按 intent 整体继承。
- **职责分离**：LLM 只做语义识别，DST 规则引擎独占状态写权限。
- **单一数据源**：Tool 参数只从最终 DST 生成，禁止多路径拼装。

---

## 2. 分层架构

```
用户原话 → LLM分类器 → 值标准化 → DST规则引擎 → Tool参数生成
                                         ↑
                                  CompatibilityProvider
```

| 层 | 职责 | 禁止 |
|---|---|---|
| LLM 分类器 | 识别用户说了什么，输出 SET/KEEP/REMOVE | 禁止输出 CLEAR，禁止决定状态如何变化 |
| 值标准化 | 将 LLM 输出映射为系统标准值 | 禁止跳过直接写入 DST |
| DST 规则引擎 | 校验、执行操作、兼容推导、留痕 | 禁止访问数据库 |
| CompatibilityProvider | 提供 slot 兼容性判断结果 | 禁止修改 DST 状态 |
| Tool 参数生成 | 从最终 DST 读取并组装参数 | 禁止从 LLM 原始输出/metadata/历史 messages 拼装 |

---

## 3. LLM 输出 Schema

LLM 对每个 slot 输出且仅输出以下三种操作：

| 操作 | 语义 | 触发条件 | 是否带 value |
|------|------|----------|-------------|
| `SET` | 用户给了新值 | "来个键盘" → category | 必须 |
| `KEEP` | 用户没提 | 未提及品牌 | 禁止 |
| `REMOVE` | 用户明确取消 | "不限品牌""不要游戏款" | 禁止 |

示例输出：

```json
{
  "category": {"op": "SET", "value": "键盘"},
  "required_use_cases": {"op": "KEEP"},
  "brand": {"op": "REMOVE"}
}
```

### 禁止事项

- LLM 不得输出 `CLEAR`（清空是 DST 规则引擎的权限）
- LLM 不得推断用户未说出的取消意图
- LLM 不得输出不在枚举白名单内的 category 值

---

## 4. 值标准化

- LLM 输出写入 DST 前，必须经过标准化映射（如 `keyboard` → `键盘`）
- 映射表配置化维护，禁止硬编码在逻辑代码中
- 标准化失败（未知值）→ 拒绝写入，标记为校验失败

---

## 5. DST 规则引擎处理顺序

严格按以下顺序执行，不可跳步：

### Step 1：校验

- `SET` 必须携带合法 value
- `REMOVE` / `KEEP` 不得携带 value
- value 必须通过枚举白名单校验
- 校验失败 → 该 slot 不执行，记录错误

### Step 2：执行显式操作

- `SET` → 写入新值
- `REMOVE` → 校验合法后转为内部 `CLEAR`，清空该 slot
- `KEEP` → 不动

### Step 3：兼容规则推导

当 `category` 发生 SET 变化时，对其余 slot 逐一检查兼容性：

| CompatibilityProvider 返回 | DST 动作 |
|---|---|
| 兼容 | 保留旧值 |
| 不兼容 | CLEAR（记录原因） |
| 无法判断 | 放宽条件（不传该参数），记录原因 |

### Step 4：CLEAR 留痕

所有 CLEAR（无论来源是 REMOVE 转换还是兼容推导）必须记录：

```json
{
  "cleared_slot": "required_use_cases",
  "reason": "user_remove | category_incompatible | compatibility_unknown",
  "previous_value": ["游戏"],
  "trigger": "category SET 键盘"
}
```

---

## 6. CompatibilityProvider 规范

- 独立模块，通过依赖注入提供给 DST 引擎
- DST 引擎不持有数据库连接，不直接查询商品目录
- Provider 接口：

```python
class CompatibilityProvider(Protocol):
    def check_slot_compatibility(
        self,
        new_category: str,
        slot_name: str,
        slot_value: Any,
    ) -> Literal["compatible", "incompatible", "unknown"]:
        ...
```

- 当前实现：查商品目录判断 use_case 是否适用于目标 category
- 测试时必须可 mock

---

## 7. Tool 参数生成规则

- **唯一数据源**：合并完成后的 DST slots
- 禁止从以下来源拼装参数：
  - LLM 原始分类输出
  - metadata.customer_service 中的历史字段
  - session messages 中的历史 tool 结果
  - 用户原话直接解析
- 参数组装函数签名中只接受 DST state，不接受其他上下文

---

## 8. 开发检查清单

新增或修改 slot 相关逻辑时，逐项确认：

- [ ] LLM 输出是否只含 SET/KEEP/REMOVE？
- [ ] 值是否经过标准化再写入？
- [ ] DST 是否按 slot 逐个裁决（而非整体继承）？
- [ ] CLEAR 是否由规则引擎执行（而非 LLM）？
- [ ] 兼容规则是否通过 Provider 注入（而非 DST 直接查库）？
- [ ] CLEAR 是否留有原因记录？
- [ ] Tool 参数是否只从 DST 读取？
- [ ] 是否覆盖了回归场景：品类切换 / 明确取消 / 未提及继承 / 兼容保留？

---

## 9. 回归测试必覆盖场景

| 场景 | 用户输入 | 预期 |
|------|----------|------|
| 品类切换 | "来个键盘"（上轮是鼠标） | category=键盘，use_cases 按兼容规则处理 |
| 明确取消 | "不限品牌" | brand 被 CLEAR，留痕 reason=user_remove |
| 未提及继承 | "便宜点的"（上轮是游戏鼠标） | category=鼠标 KEEP，use_cases=游戏 KEEP，新增 price_max |
| 兼容保留 | "来个游戏键盘"（上轮是游戏鼠标） | category=键盘 SET，use_cases=游戏 兼容保留 |
| 值标准化 | LLM 输出 "keyboard" | DST 写入 "键盘" |
