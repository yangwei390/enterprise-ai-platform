import { FormEvent, useEffect, useState } from "react";
import { executeTool, listTools } from "../api/tools";
import type { ToolDefinition, ToolResult } from "../types/common";

export default function ToolsPage() {
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [expression, setExpression] = useState("1 + 2 * 3");
  const [toolName, setToolName] = useState("calculator");
  const [argumentsText, setArgumentsText] = useState('{"expression":"1 + 2 * 3"}');
  const [result, setResult] = useState<ToolResult | { error: string } | null>(null);

  useEffect(() => {
    void refreshTools();
  }, []);

  async function refreshTools() {
    try {
      setTools(await listTools());
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
  }

  async function runCalculator() {
    try {
      setResult(await executeTool("calculator", { expression }));
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
  }

  async function handleExecute(event: FormEvent) {
    event.preventDefault();
    try {
      setResult(await executeTool(toolName, JSON.parse(argumentsText) as Record<string, unknown>));
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>Tools</h2>
        <p>Inspect ToolRegistry and execute built-in or remote tools.</p>
      </div>
      <div className="two-column">
        <div className="stack">
          <div className="card form">
            <h3>内置 Calculator 测试</h3>
            <label>
              Expression
              <input value={expression} onChange={(event) => setExpression(event.target.value)} />
            </label>
            <button type="button" onClick={() => void runCalculator()}>
              Execute calculator
            </button>
          </div>
          <form className="card form" onSubmit={handleExecute}>
            <h3>执行工具</h3>
            <label>
              Tool Name
              <input value={toolName} onChange={(event) => setToolName(event.target.value)} />
            </label>
            <label>
              Arguments JSON
              <textarea value={argumentsText} onChange={(event) => setArgumentsText(event.target.value)} />
            </label>
            <button type="submit">POST /tools/execute</button>
          </form>
          <div className="card">
            <div className="section-header">
              <h3>Registered Tools</h3>
              <button type="button" className="secondary" onClick={() => void refreshTools()}>
                Refresh
              </button>
            </div>
            <pre>{JSON.stringify(tools, null, 2)}</pre>
          </div>
        </div>
        <div className="card result-card">
          <h3>ToolResult</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
