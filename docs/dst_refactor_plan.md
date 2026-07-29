# DST 状态合并根治方案（终版）

## 问题本质

DST 合并逻辑按 intent 整体继承旧 slots，导致当前轮 LLM 正确识别的新实体被旧值覆盖。反复打补丁无法根治，需从架构层面解决。

## 架构分层

用户原话 → LLM分类器 → 值标准化 → DST规则引擎 → Tool参数生成
                                         ↑
                                  CompatibilityProvider

## 各层职责

### ① LLM 分类器（只识别用户说了什么）

输出三种操作，禁止输出 CLEAR：

| 操作 | 语义 | 示例 |
|------|------|------|
| SET | 用户给了新值 | "来个键盘" → category: SET "键盘" |
| KEEP | 用户没提 | 未提及品牌 → brand: KEEP |
| REMOVE | 用户明确取消 | "不限品牌" → brand: REMOVE |

输出 schema：

    {
      "category": {"op": "SET", "value": "键盘"},
      "required_use_cases": {"op": "KEEP"},
      "brand": {"op": "REMOVE"}
    }

### ② 值标准化层

LLM 输出写入 DST 前，统一映射为标准值（如 keyboard → 键盘）。映射表配置化，不硬编码。

### ③ DST 规则引擎（唯一状态写权限）

处理顺序：

1. 校验：SET 必带合法值；REMOVE/KEEP 不带值；值过枚举白名单
2. 执行显式操作：SET → 写入；REMOVE → 校验合法后转 CLEAR；KEEP → 不动
3. 兼容规则推导：category 变化时，由 CompatibilityProvider 返回兼容结果：
   - 兼容 → 保留旧 slot
   - 不兼容 → CLEAR（记录原因）
   - 无法判断 → 放宽条件（不传该参数），记录原因
4. CLEAR 必须留痕：

    {"cleared_slot": "required_use_cases", "reason": "compatibility_unknown"}

### ④ CompatibilityProvider（独立组件）

- DST 不访问数据库，兼容性结果由此 Provider 注入
- 可 mock、可替换、可独立单测
- 当前实现：查商品目录判断 use_case 是否适用于新 category

### ⑤ Tool 参数生成

唯一数据源 = 最终 DST 状态。禁止从 LLM 原始输出、metadata、历史 messages 拼装参数。

## 不动的部分

- LangGraph runtime 框架层
- Session 存取机制
- Agent 定义与工具注册

## 改动范围

- classifier prompt + 输出解析（改 schema 为 SET/KEEP/REMOVE）
- 值标准化映射表（新增配置）
- DST 合并函数（重写）
- CompatibilityProvider（新增独立模块）
- Tool 参数组装入口收拢（清理旧路径）
- CLEAR 留痕日志
- 回归测试：品类切换、明确取消、未提及继承、兼容保留、值标准化
