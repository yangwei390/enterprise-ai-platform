import { FormEvent, useState } from "react";
import { agentChat } from "../api/agent";
import type { AgentChatResponseData } from "../types/common";

export default function AgentPage() {
  const [query, setQuery] = useState("劳动法第二章说什么");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("4");
  const [result, setResult] = useState<AgentChatResponseData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
        <p>Run AgentRuntime through /agent/chat and inspect tool calls, observations, sources, and trace.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
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

        <div className="stack">
          <div className="card">
            <h3>Answer</h3>
            <p className="answer">{result?.answer ?? ""}</p>
            {result && (
              <p className="muted">
                action: {result.action} · trace steps: {result.trace.length}
              </p>
            )}
          </div>

          <div className="card">
            <h3>Execution Trace</h3>
            <div className="agent-trace-list">
              {result?.trace.map((step, index) => (
                <div className="agent-trace-step" key={`${step.step}-${index}`}>
                  <div className="section-header">
                    <h4>Step {index + 1} · {step.step}</h4>
                    <span className={step.status === "success" ? "status-ok" : "status-failed"}>
                      {step.status}
                    </span>
                  </div>
                  <p className="muted">
                    {step.name} · {step.duration_ms}ms
                  </p>
                  {step.error && <p className="error-text">{step.error}</p>}
                  <details>
                    <summary>Input / Output</summary>
                    <pre>{JSON.stringify({ input: step.input, output: step.output }, null, 2)}</pre>
                  </details>
                </div>
              ))}
            </div>
          </div>

          <div className="card result-card">
            <h3>Tool Calls</h3>
            <pre>{JSON.stringify(result?.tool_calls ?? [], null, 2)}</pre>
          </div>

          <div className="card result-card">
            <h3>Observations</h3>
            <pre>{JSON.stringify(result?.observations ?? [], null, 2)}</pre>
          </div>

          <div className="card">
            <h3>Sources</h3>
            <div className="trace-chunk-list">
              {result?.sources.map((source, index) => (
                <div className="trace-chunk-card" key={index}>
                  <div className="trace-chunk-meta">
                    <span>source: {String(source.source ?? "-")}</span>
                    <span>document_id: {String(source.document_id ?? "-")}</span>
                    <span>chunk_index: {String(source.chunk_index ?? "-")}</span>
                    <span>
                      score: {String(source.rerank_score ?? source.score ?? "-")}
                    </span>
                  </div>
                  <pre className="trace-preview">
                    {JSON.stringify(source, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>

          <div className="card result-card">
            <h3>Raw Metadata / JSON</h3>
            <details open>
              <summary>Response</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}
