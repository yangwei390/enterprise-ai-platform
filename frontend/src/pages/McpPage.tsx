import { FormEvent, useEffect, useState } from "react";
import { listMcpTools, registerMcpTool } from "../api/mcp";
import type { ToolDefinition } from "../types/common";

export default function McpPage() {
  const [name, setName] = useState("remote_echo");
  const [description, setDescription] = useState("远程 echo 工具");
  const [endpoint, setEndpoint] = useState("http://127.0.0.1:9000/echo");
  const [method, setMethod] = useState("POST");
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [result, setResult] = useState<unknown>(null);

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    try {
      setTools(await listMcpTools());
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    try {
      const registered = await registerMcpTool({
        name,
        description,
        endpoint,
        method,
        headers: {},
        parameters: {
          type: "object",
          properties: {
            text: { type: "string" }
          },
          required: ["text"]
        },
        timeout: 30
      });
      setResult(registered);
      await refresh();
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>MCP</h2>
        <p>Register remote HTTP tools as MCP-style platform tools.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
          <label>
            Name
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            Description
            <input value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <label>
            Endpoint
            <input value={endpoint} onChange={(event) => setEndpoint(event.target.value)} />
          </label>
          <label>
            Method
            <select value={method} onChange={(event) => setMethod(event.target.value)}>
              <option>POST</option>
              <option>GET</option>
            </select>
          </label>
          <button type="submit">POST /mcp/tools/register</button>
        </form>
        <div className="stack">
          <div className="card result-card">
            <h3>Register Result</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
          <div className="card result-card">
            <h3>MCP Tools</h3>
            <button type="button" className="secondary" onClick={() => void refresh()}>
              GET /mcp/tools
            </button>
            <pre>{JSON.stringify(tools, null, 2)}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
