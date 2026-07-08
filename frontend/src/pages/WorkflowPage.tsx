import { FormEvent, useState } from "react";
import { runWorkflow } from "../api/workflow";
import type { WorkflowRunResponseData } from "../types/common";

export default function WorkflowPage() {
  const [query, setQuery] = useState("劳动法第二章说什么");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("4");
  const [workflowId, setWorkflowId] = useState("default_knowledge_workflow");
  const [result, setResult] = useState<WorkflowRunResponseData | null>(null);
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
      const response = await runWorkflow({
        query: trimmedQuery,
        knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
        workflow_id: workflowId || "default_knowledge_workflow",
        inputs: {}
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
        <h2>Workflow Studio</h2>
        <p>Run default_knowledge_workflow through /workflow/run and inspect node outputs and trace.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
          <label>
            Query
            <textarea
              placeholder="请输入 Workflow 任务，例如：劳动法第二章说什么"
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
          <label>
            Workflow ID
            <input
              value={workflowId}
              onChange={(event) => setWorkflowId(event.target.value)}
              placeholder="default_knowledge_workflow"
            />
          </label>
          {error && <p className="error-text">{error}</p>}
          <button type="submit" disabled={loading}>
            {loading ? "Running..." : "Run Workflow"}
          </button>
        </form>

        <div className="stack">
          <div className="card">
            <h3>Answer</h3>
            <p className="answer">{result?.answer ?? ""}</p>
            {result && (
              <p className="muted">
                workflow: {String(result.metadata.workflow_id ?? "-")} · trace steps: {result.trace.length}
              </p>
            )}
          </div>

          <div className="card">
            <h3>Workflow Trace</h3>
            <div className="agent-trace-list">
              {result?.trace.map((step) => (
                <div className="agent-trace-step" key={`${step.step}-${step.node_id}`}>
                  <div className="section-header">
                    <h4>
                      Step {step.step} · {step.node_id}
                    </h4>
                    <span className={step.status === "success" ? "status-ok" : "status-failed"}>
                      {step.status}
                    </span>
                  </div>
                  <p className="muted">
                    type: {step.node_type} · duration: {step.duration_ms}ms
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
            <h3>Node Outputs</h3>
            <pre>{JSON.stringify(result?.node_outputs ?? {}, null, 2)}</pre>
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
