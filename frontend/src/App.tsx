import { useState } from "react";
import type { ReactNode } from "react";
import { API_BASE_URL } from "./api/client";
import DashboardPage from "./pages/DashboardPage";
import KnowledgePage from "./pages/KnowledgePage";
import ChatPage from "./pages/ChatPage";
import ToolsPage from "./pages/ToolsPage";
import WorkflowPage from "./pages/WorkflowPage";
import AgentPage from "./pages/AgentPage";
import McpPage from "./pages/McpPage";

type PageKey =
  | "dashboard"
  | "knowledge"
  | "chat"
  | "tools"
  | "workflow"
  | "agent"
  | "mcp";

const navItems: Array<{ key: PageKey; label: string }> = [
  { key: "dashboard", label: "Dashboard" },
  { key: "knowledge", label: "Knowledge" },
  { key: "chat", label: "Chat" },
  { key: "tools", label: "Tools" },
  { key: "workflow", label: "Workflow" },
  { key: "agent", label: "Agent" },
  { key: "mcp", label: "MCP" }
];

const pages: Record<PageKey, ReactNode> = {
  dashboard: <DashboardPage />,
  knowledge: <KnowledgePage />,
  chat: <ChatPage />,
  tools: <ToolsPage />,
  workflow: <WorkflowPage />,
  agent: <AgentPage />,
  mcp: <McpPage />
};

export default function App() {
  const [activePage, setActivePage] = useState<PageKey>("dashboard");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <div>
            <h1>Enterprise AI Studio</h1>
            <p>RAG · Agent · Tools</p>
          </div>
        </div>
        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              key={item.key}
              className={activePage === item.key ? "nav-item active" : "nav-item"}
              type="button"
              onClick={() => setActivePage(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <strong>API Base URL</strong>
            <span>{API_BASE_URL}</span>
          </div>
        </header>
        <section className="page">{pages[activePage]}</section>
      </main>
    </div>
  );
}
