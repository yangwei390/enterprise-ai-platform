import { FormEvent, useState } from "react";
import { runAgent } from "../api/agent";
import type { AgentResult } from "../types/common";

export default function AgentPage() {
  const [task, setTask] = useState("计算 1+2*3");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("2");
  const [enableTools, setEnableTools] = useState(true);
  const [enableMemory, setEnableMemory] = useState(true);
  const [result, setResult] = useState<AgentResult | { error: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const agentResult = result && "answer" in result ? result : null;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    try {
      setResult(
        await runAgent({
          task,
          knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
          enable_tools: enableTools,
          enable_memory: enableMemory
        })
      );
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>Agent</h2>
        <p>Run Agent Planner, Workflow Executor, Reflection, and final answer.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
          <label>
            Task
            <textarea value={task} onChange={(event) => setTask(event.target.value)} />
          </label>
          <label>
            Knowledge Base ID
            <input value={knowledgeBaseId} onChange={(event) => setKnowledgeBaseId(event.target.value)} />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={enableTools}
              onChange={(event) => setEnableTools(event.target.checked)}
            />
            Enable Tools
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={enableMemory}
              onChange={(event) => setEnableMemory(event.target.checked)}
            />
            Enable Memory
          </label>
          <button type="submit" disabled={loading}>POST /agents/run</button>
        </form>
        <div className="stack">
          <div className="card">
            <h3>Answer</h3>
            <p className="answer">{agentResult?.answer ?? ""}</p>
          </div>
          <div className="card result-card">
            <h3>Steps / Artifacts / Reflection</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
