import { FormEvent, useEffect, useMemo, useState } from "react";
import { agentChat } from "../api/agent";
import {
  getCacheDebug,
  getCheckpointsDebug,
  getMcpDebug,
  getMemoryDebug
} from "../api/debug";
import type {
  AgentChatResponseData,
  AgentTraceResult,
  AgentTraceStep,
  CacheDebugSnapshot,
  CheckpointsDebugSnapshot,
  McpDebugSnapshot,
  MemoryDebugSnapshot
} from "../types/common";

type ViewerState<T> = {
  loading: boolean;
  error: string;
  data: T | null;
};

type ToolTimelineItem = {
  id: string;
  tool_name: string;
  provider: string;
  arguments: unknown;
  success: boolean | null;
  cache_hit: boolean | null;
  retry_count: number | null;
  duration_ms: number | null;
  error: string | null;
  raw: unknown;
};

const SENSITIVE_KEY_PATTERN = /(api[_-]?key|authorization|cookie|password|passwd|secret|token|access[_-]?token|refresh[_-]?token)/i;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function redactSensitive(value: unknown, parentKey = "", depth = 0): unknown {
  if (SENSITIVE_KEY_PATTERN.test(parentKey)) {
    return "[REDACTED]";
  }
  if (depth > 8) {
    return "[MAX_DEPTH]";
  }
  if (Array.isArray(value)) {
    return value.map((item) => redactSensitive(item, parentKey, depth + 1));
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        redactSensitive(item, key, depth + 1)
      ])
    );
  }
  return value;
}

function readPath(value: unknown, path: string[]): unknown {
  let current = value;
  for (const key of path) {
    if (!isRecord(current)) {
      return undefined;
    }
    current = current[key];
  }
  return current;
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value;
    }
    if (typeof value === "number") {
      return String(value);
    }
  }
  return "-";
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function firstBoolean(...values: unknown[]): boolean | null {
  for (const value of values) {
    if (typeof value === "boolean") {
      return value;
    }
  }
  return null;
}

function traceStatus(step: AgentTraceStep): string {
  return firstString(step.status, readPath(step.output, ["status"]));
}

function traceDuration(step: AgentTraceStep): number | null {
  return firstNumber(step.duration_ms, step.duration, readPath(step.output, ["duration_ms"]));
}

function traceToolName(step: AgentTraceStep): string {
  return firstString(
    step.tool_name,
    readPath(step.input, ["tool_name"]),
    readPath(step.input, ["tool"]),
    readPath(step.output, ["tool_name"]),
    readPath(step.output, ["name"])
  );
}

function traceErrorText(step: AgentTraceStep): string | null {
  if (typeof step.error === "string" && step.error.trim()) {
    return step.error;
  }
  const outputError = readPath(step.output, ["error"]);
  if (typeof outputError === "string" && outputError.trim()) {
    return outputError;
  }
  return null;
}

function buildAgentState(result: AgentChatResponseData | null): Record<string, unknown> {
  if (!result) {
    return {};
  }
  const metadataState = isRecord(result.metadata.state) ? result.metadata.state : {};
  const agentLoop = isRecord(result.metadata.agent_loop) ? result.metadata.agent_loop : {};
  return {
    messages: metadataState.messages ?? result.metadata.messages ?? [],
    step_count: metadataState.step_count ?? agentLoop.steps ?? result.trace.length,
    llm_call_count: metadataState.llm_call_count ?? agentLoop.llm_calls ?? null,
    tool_call_count: metadataState.tool_call_count ?? agentLoop.tool_calls ?? result.tool_calls.length,
    reflection_count: metadataState.reflection_count ?? agentLoop.reflections ?? null,
    current_action: metadataState.current_action ?? result.action,
    observations: metadataState.observations ?? result.observations,
    final_answer: metadataState.final_answer ?? result.answer,
    metadata: result.metadata
  };
}

function normalizeTrace(result: AgentChatResponseData | null): AgentTraceStep[] {
  return result?.trace ?? [];
}

function unifiedTrace(result: AgentChatResponseData | null): AgentTraceResult | null {
  const value = result?.metadata.agent_trace;
  return isRecord(value) ? (value as AgentTraceResult) : null;
}

function toolItemFromTrace(step: AgentTraceStep, index: number): ToolTimelineItem | null {
  const toolName = traceToolName(step);
  const action = firstString(step.action, step.step, step.name);
  if (toolName === "-" && !action.toLowerCase().includes("tool")) {
    return null;
  }
  const status = traceStatus(step);
  const outputMetadata = readPath(step.output, ["metadata"]);
  const metadata = isRecord(outputMetadata) ? outputMetadata : step.metadata ?? {};
  return {
    id: `trace-${index}`,
    tool_name: toolName,
    provider: firstString(metadata.provider, readPath(step.output, ["provider"])),
    arguments: redactSensitive(
      readPath(step.input, ["arguments"]) ?? readPath(step.input, ["args"]) ?? step.input
    ),
    success: firstBoolean(
      readPath(step.output, ["success"]),
      status === "-" ? null : !["failed", "error", "timeout"].includes(status.toLowerCase())
    ),
    cache_hit: firstBoolean(metadata.cache_hit, readPath(step.output, ["cache_hit"])),
    retry_count: firstNumber(metadata.retry_count, metadata.attempt_count),
    duration_ms: traceDuration(step),
    error: traceErrorText(step),
    raw: redactSensitive(step)
  };
}

function toolItemFromCall(call: Record<string, unknown>, index: number): ToolTimelineItem {
  const metadata = isRecord(call.metadata) ? call.metadata : {};
  return {
    id: `call-${index}`,
    tool_name: firstString(call.tool_name, call.tool, call.name),
    provider: firstString(metadata.provider, call.provider),
    arguments: redactSensitive(call.arguments ?? call.args ?? call.input ?? {}),
    success: firstBoolean(call.success),
    cache_hit: firstBoolean(metadata.cache_hit, call.cache_hit),
    retry_count: firstNumber(metadata.retry_count, metadata.attempt_count, call.retry_count),
    duration_ms: firstNumber(metadata.duration_ms, call.duration_ms),
    error: typeof call.error === "string" ? call.error : null,
    raw: redactSensitive(call)
  };
}

function buildToolTimeline(result: AgentChatResponseData | null): ToolTimelineItem[] {
  if (!result) {
    return [];
  }
  const fromTrace = result.trace
    .map((step, index) => toolItemFromTrace(step, index))
    .filter((item): item is ToolTimelineItem => item !== null);
  const fromCalls = result.tool_calls.map((call, index) => toolItemFromCall(call, index));
  const seen = new Set<string>();
  return [...fromTrace, ...fromCalls].filter((item) => {
    const key = `${item.tool_name}:${JSON.stringify(item.arguments)}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(redactSensitive(value), null, 2)}</pre>;
}

function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const ok = ["success", "ok", "completed", "passed"].includes(normalized);
  const failed = ["failed", "error", "timeout", "cancelled"].includes(normalized);
  return (
    <span className={ok ? "status-ok" : failed ? "status-failed" : "status-neutral"}>
      {status || "-"}
    </span>
  );
}

function EmptyState({ text }: { text: string }) {
  return <p className="muted empty-state">{text}</p>;
}

export default function AgentPage() {
  const [query, setQuery] = useState("劳动法第二章说什么");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("4");
  const [result, setResult] = useState<AgentChatResponseData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [memoryState, setMemoryState] = useState<ViewerState<MemoryDebugSnapshot>>({
    loading: false,
    error: "",
    data: null
  });
  const [cacheState, setCacheState] = useState<ViewerState<CacheDebugSnapshot>>({
    loading: false,
    error: "",
    data: null
  });
  const [checkpointState, setCheckpointState] = useState<ViewerState<CheckpointsDebugSnapshot>>({
    loading: false,
    error: "",
    data: null
  });
  const [mcpState, setMcpState] = useState<ViewerState<McpDebugSnapshot>>({
    loading: false,
    error: "",
    data: null
  });

  const trace = useMemo(() => normalizeTrace(result), [result]);
  const agentTrace = useMemo(() => unifiedTrace(result), [result]);
  const agentState = useMemo(() => buildAgentState(result), [result]);
  const toolTimeline = useMemo(() => buildToolTimeline(result), [result]);

  useEffect(() => {
    void refreshDebugViewers();
  }, []);

  async function loadViewer<T>(
    loader: () => Promise<T>,
    setter: (state: ViewerState<T>) => void
  ) {
    setter({ loading: true, error: "", data: null });
    try {
      setter({ loading: false, error: "", data: await loader() });
    } catch (requestError) {
      setter({
        loading: false,
        error: requestError instanceof Error ? requestError.message : String(requestError),
        data: null
      });
    }
  }

  async function refreshDebugViewers() {
    await Promise.all([
      loadViewer(getMemoryDebug, setMemoryState),
      loadViewer(getCacheDebug, setCacheState),
      loadViewer(getCheckpointsDebug, setCheckpointState),
      loadViewer(getMcpDebug, setMcpState)
    ]);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError("请输入内容");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await agentChat({
        query: trimmedQuery,
        knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
        conversation_id: null,
        memory_context: null,
        metadata: {}
      });
      setResult(response);
      void refreshDebugViewers();
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>Agent Studio</h2>
        <p>Run AgentRuntime through /agent/chat and inspect trace, state, tools, memory, Redis, and MCP.</p>
      </div>

      <div className="studio-grid">
        <form className="card form studio-run-card" onSubmit={handleSubmit}>
          <h3>Agent Run</h3>
          <label>
            Query
            <textarea
              placeholder="请输入 Agent 任务，例如：劳动法第二章说什么"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <label>
            Knowledge Base ID
            <input
              value={knowledgeBaseId}
              onChange={(event) => setKnowledgeBaseId(event.target.value)}
              placeholder="可为空"
            />
          </label>
          {error && <p className="error-text">{error}</p>}
          <button type="submit" disabled={loading}>
            {loading ? "Running..." : "Run Agent"}
          </button>
        </form>

        <div className="card studio-answer-card">
          <h3>Answer</h3>
          {result ? (
            <>
              <p className="answer">{result.answer}</p>
              <p className="muted">
                action: {result.action} · trace steps: {trace.length} · tools: {toolTimeline.length}
                {agentTrace?.trace_id ? ` · trace_id: ${agentTrace.trace_id}` : ""}
              </p>
            </>
          ) : (
            <EmptyState text="Run an Agent task to inspect the response." />
          )}
        </div>
      </div>

      <div className="studio-panels">
        <section className="card">
          <div className="section-header">
            <h3>Unified Trace Contract</h3>
            <span className="muted">{agentTrace?.runtime ?? "-"}</span>
          </div>
          {agentTrace ? (
            <>
              <div className="state-grid compact-grid">
                <div className="metric-card"><span>trace_id</span><strong>{agentTrace.trace_id ?? "-"}</strong></div>
                <div className="metric-card"><span>agent</span><strong>{agentTrace.agent_id ?? "-"}</strong></div>
                <div className="metric-card"><span>grounded</span><strong>{String(agentTrace.evidence?.grounded_answer ?? "-")}</strong></div>
                <div className="metric-card"><span>errors</span><strong>{String(agentTrace.errors?.length ?? 0)}</strong></div>
              </div>
              <details open>
                <summary>AgentDefinition / Planner</summary>
                <JsonBlock value={{ agent_definition: agentTrace.agent_definition, planner: agentTrace.planner, plan_steps: agentTrace.plan_steps }} />
              </details>
              <details>
                <summary>Tool Scope / Tool Calls</summary>
                <JsonBlock value={{ tool_scope: agentTrace.tool_scope, tool_calls: agentTrace.tool_calls }} />
              </details>
              <details>
                <summary>RAG / Evidence</summary>
                <JsonBlock value={{ retrieval: agentTrace.retrieval, evidence: agentTrace.evidence }} />
              </details>
              <details>
                <summary>Memory / Checkpoint</summary>
                <JsonBlock value={{ memory: agentTrace.memory, checkpoint: agentTrace.checkpoint }} />
              </details>
              <details>
                <summary>Final / Errors / Timing / Tokens</summary>
                <JsonBlock value={{ final_answer: agentTrace.final_answer, errors: agentTrace.errors, timing: agentTrace.timing, token_usage: agentTrace.token_usage }} />
              </details>
            </>
          ) : (
            <EmptyState text="Unified trace contract is empty until an Agent run completes." />
          )}
        </section>

        <section className="card">
          <div className="section-header">
            <h3>Agent Trace</h3>
            <span className="muted">{trace.length} steps</span>
          </div>
          {trace.length === 0 ? (
            <EmptyState text="Trace is empty." />
          ) : (
            <div className="agent-trace-list timeline-list">
              {trace.map((step, index) => {
                const status = traceStatus(step);
                const duration = traceDuration(step);
                const stepError = traceErrorText(step);
                return (
                  <div className="agent-trace-step timeline-step" key={`${step.step ?? index}-${index}`}>
                    <div className="timeline-index">{index + 1}</div>
                    <div className="timeline-body">
                      <div className="section-header">
                        <h4>
                          {firstString(step.node, step.name, step.step)} · {firstString(step.action, "-")}
                        </h4>
                        <StatusPill status={status} />
                      </div>
                      <div className="metric-row">
                        <span>tool: {traceToolName(step)}</span>
                        <span>duration: {duration === null ? "-" : `${duration}ms`}</span>
                        <span>step: {String(step.step ?? index + 1)}</span>
                      </div>
                      {stepError && <p className="error-text">{stepError}</p>}
                      <details>
                        <summary>Input / Output</summary>
                        <JsonBlock value={{ input: step.input, output: step.output, metadata: step.metadata }} />
                      </details>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="card">
          <h3>LangGraph State</h3>
          {result ? (
            <div className="state-grid">
              <div className="metric-card">
                <span>step_count</span>
                <strong>{String(agentState.step_count ?? "-")}</strong>
              </div>
              <div className="metric-card">
                <span>llm_call_count</span>
                <strong>{String(agentState.llm_call_count ?? "-")}</strong>
              </div>
              <div className="metric-card">
                <span>tool_call_count</span>
                <strong>{String(agentState.tool_call_count ?? "-")}</strong>
              </div>
              <div className="metric-card">
                <span>reflection_count</span>
                <strong>{String(agentState.reflection_count ?? "-")}</strong>
              </div>
              <div className="state-wide">
                <JsonBlock value={agentState} />
              </div>
            </div>
          ) : (
            <EmptyState text="State is derived from Agent metadata and trace after a run." />
          )}
        </section>

        <section className="card">
          <div className="section-header">
            <h3>Tool Timeline</h3>
            <span className="muted">Builtin and MCP tools share this timeline.</span>
          </div>
          {toolTimeline.length === 0 ? (
            <EmptyState text="No tool calls observed." />
          ) : (
            <div className="tool-timeline">
              {toolTimeline.map((item) => (
                <div className="tool-timeline-item" key={item.id}>
                  <div className="section-header">
                    <h4>{item.tool_name}</h4>
                    <StatusPill status={item.success === false ? "failed" : item.success === true ? "success" : "unknown"} />
                  </div>
                  <div className="metric-row">
                    <span>provider: {item.provider}</span>
                    <span>cache_hit: {String(item.cache_hit ?? "-")}</span>
                    <span>retry_count: {String(item.retry_count ?? "-")}</span>
                    <span>duration: {item.duration_ms === null ? "-" : `${item.duration_ms}ms`}</span>
                  </div>
                  {item.error && <p className="error-text">{item.error}</p>}
                  <details>
                    <summary>Arguments / Raw</summary>
                    <JsonBlock value={{ arguments: item.arguments, raw: item.raw }} />
                  </details>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <div className="section-header">
            <h3>Memory Viewer</h3>
            <button type="button" className="secondary compact-button" onClick={() => void refreshDebugViewers()}>
              Refresh
            </button>
          </div>
          {memoryState.loading ? (
            <EmptyState text="Loading memory snapshot..." />
          ) : memoryState.error ? (
            <p className="error-text">Unavailable: {memoryState.error}</p>
          ) : memoryState.data ? (
            <>
              <div className="state-grid compact-grid">
                <div className="metric-card"><span>provider</span><strong>{memoryState.data.provider ?? "-"}</strong></div>
                <div className="metric-card"><span>sessions</span><strong>{String(memoryState.data.session_count ?? "-")}</strong></div>
                <div className="metric-card"><span>cache</span><strong>{String(memoryState.data.cache_count ?? "-")}</strong></div>
                <div className="metric-card"><span>checkpoints</span><strong>{String(memoryState.data.checkpoint_count ?? "-")}</strong></div>
              </div>
              <details>
                <summary>Memory State</summary>
                <JsonBlock value={memoryState.data} />
              </details>
            </>
          ) : (
            <EmptyState text="No memory snapshot loaded." />
          )}
        </section>

        <section className="card">
          <h3>Redis Viewer</h3>
          {memoryState.loading || cacheState.loading || checkpointState.loading ? (
            <EmptyState text="Loading Redis-related debug data..." />
          ) : memoryState.error ? (
            <p className="error-text">Unavailable: {memoryState.error}</p>
          ) : memoryState.data?.provider !== "redis" ? (
            <EmptyState text={`Disabled or not using Redis provider. Current provider: ${memoryState.data?.provider ?? "-"}`} />
          ) : (
            <>
              <div className="state-grid compact-grid">
                <div className="metric-card"><span>Redis enabled</span><strong>true</strong></div>
                <div className="metric-card"><span>session count</span><strong>{String(memoryState.data.session_count ?? "-")}</strong></div>
                <div className="metric-card"><span>cache count</span><strong>{String(cacheState.data?.cache_count ?? memoryState.data.cache_count ?? "-")}</strong></div>
                <div className="metric-card"><span>checkpoint count</span><strong>{String(memoryState.data.checkpoint_count ?? checkpointState.data?.checkpoints.length ?? "-")}</strong></div>
              </div>
              {cacheState.error && <p className="error-text">Cache unavailable: {cacheState.error}</p>}
              {checkpointState.error && <p className="error-text">Checkpoints unavailable: {checkpointState.error}</p>}
              <details>
                <summary>TTL / Cache / Checkpoints</summary>
                <JsonBlock value={{ memory: memoryState.data, cache: cacheState.data, checkpoints: checkpointState.data }} />
              </details>
            </>
          )}
        </section>

        <section className="card">
          <h3>MCP Viewer</h3>
          {mcpState.loading ? (
            <EmptyState text="Loading MCP debug state..." />
          ) : mcpState.error ? (
            <p className="error-text">Unavailable: {mcpState.error}</p>
          ) : mcpState.data?.enabled === false ? (
            <EmptyState text="MCP is disabled." />
          ) : mcpState.data ? (
            <>
              <div className="state-grid compact-grid">
                <div className="metric-card"><span>enabled</span><strong>{String(mcpState.data.enabled ?? "-")}</strong></div>
                <div className="metric-card"><span>servers</span><strong>{String(mcpState.data.configured_servers?.length ?? 0)}</strong></div>
                <div className="metric-card"><span>tools</span><strong>{String(mcpState.data.discovered_tool_count ?? mcpState.data.tools?.length ?? 0)}</strong></div>
                <div className="metric-card"><span>registry</span><strong>{String(mcpState.data.registry_version ?? "-")}</strong></div>
              </div>
              <div className="tool-timeline">
                {(mcpState.data.configured_servers ?? []).map((server, index) => (
                  <div className="tool-timeline-item" key={String(server.name ?? index)}>
                    <div className="section-header">
                      <h4>{String(server.name ?? `server-${index + 1}`)}</h4>
                      <StatusPill status={String(server.connected ?? server.status ?? "unknown")} />
                    </div>
                    <div className="metric-row">
                      <span>transport: {String(server.transport ?? "-")}</span>
                      <span>enabled: {String(server.enabled ?? "-")}</span>
                      <span>health: {String(server.health ?? "-")}</span>
                      <span>error: {String(server.error ?? server.last_error ?? "-")}</span>
                    </div>
                  </div>
                ))}
              </div>
              <details>
                <summary>MCP Tools / Health</summary>
                <JsonBlock value={mcpState.data} />
              </details>
            </>
          ) : (
            <EmptyState text="No MCP debug data loaded." />
          )}
        </section>

        <section className="card result-card">
          <h3>Raw Response</h3>
          <JsonBlock value={result} />
        </section>
      </div>
    </div>
  );
}
