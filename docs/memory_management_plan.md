# 智能客服 Agent 记忆管理优化方案

## 现状诊断

核心问题：客服 Agent 完全绕过了 MemoryService，走的是全量存取。

| 组件 | 设计意图 | 实际使用 |
|------|----------|----------|
| MemoryService（Window+Summary+TokenBudget） | 给 Chat 用 | 客服 Agent 不用 |
| runtime _inject_session_state | 恢复会话 | 客服分支：全量恢复，无截断 |
| runtime _build_session_state | 保存会话 | 客服分支：全量保存，无压缩 |
| SummaryMemory | 摘要 | 只是拼接消息截断 1000 字符，不是语义摘要 |
| TokenBudgetManager | 控制 token | 客服路径完全不调用 |

结果：每轮对话的 messages + tool_results（每条最大 6000 字符）无限累积。5 轮后 LLM 上下文里全是历史 tool 结果，注意力从当前意图漂移。

## 方案总纲

让客服 Agent 接入已有的 MemoryService 架构，补上 Turn 级压缩和结构化摘要。

## 改动 1：恢复时按 Turn 截断

位置：runtime.py _inject_session_state 客服分支

逻辑：
- 把 session_state.messages 按 turn 分组（一条 user 开始 = 一个 turn）
- 保留最近 3 个完整 turn 的原始消息
- 更早的 turn 不恢复原文，改为恢复一条结构化摘要消息

## 改动 2：保存时生成 Turn 摘要

位置：runtime.py _build_session_state 客服分支

逻辑：
- 不再全量保存 messages
- 保存时：最近 3 turn 原文 + 更早 turn 压缩为摘要
- 摘要格式（确定性模板，不调 LLM）：

    [Turn N] 用户意图: product_recommendation | 条件: category=键盘 | 工具: recommend_products | 结果: 推荐了G512 X 75

## 改动 3：Tool 结果渐进截断

位置：_build_session_state 或新增 _compress_tool_results

逻辑：
- 当前 turn 的 tool_results：完整保留
- 上一 turn 的 tool_results：截断到 500 字符
- 更早的：只保留 {tool_name, success, total} 元信息

## 改动 4：SummaryMemory 改为真正的语义摘要

位置：summary_memory.py

逻辑：
- 废弃当前的拼接截断
- 改为确定性模板摘要（从 DST 提取）：

    会话摘要：用户先后咨询了办公鼠标（推荐MX Master 4）、游戏鼠标（推荐G304）、键盘（推荐G512 X 75）。当前关注：第一款鼠标的蓝牙连接能力。

- 数据源 = DST 的 slot_change_log + seen_refs + 结构化候选历史，不需要调 LLM

## 改动 5：配置项

新增到 settings.py：

    CUSTOMER_SERVICE_MAX_HISTORY_TURNS: int = 3
    CUSTOMER_SERVICE_TOOL_RESULT_MAX_CHARS_HISTORY: int = 500

## 不动的部分

- MemoryProvider / Redis 存取机制
- CheckpointManager
- Tool Cache
- 非客服 Agent 的记忆路径

## 改动量

约 120 行。核心在 runtime.py 的两个函数 + summary_memory.py 重写。不动框架，不动存储层。
