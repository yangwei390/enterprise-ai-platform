import { FormEvent, useState } from "react";
import { planAndRunWorkflow } from "../api/workflow";
import type { WorkflowResult } from "../types/common";

export default function WorkflowPage() {
  const [task, setTask] = useState("计算 1+2*3");
  const [result, setResult] = useState<WorkflowResult | { error: string } | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    try {
      setResult(await planAndRunWorkflow(task));
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>Workflow</h2>
        <p>Plan and run workflow definitions with logs and artifacts.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
          <label>
            Task
            <textarea value={task} onChange={(event) => setTask(event.target.value)} />
          </label>
          <button type="submit" disabled={loading}>POST /workflows/plan-and-run</button>
        </form>
        <div className="card result-card">
          <h3>Workflow Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
