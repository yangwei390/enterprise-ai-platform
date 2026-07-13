import { useEffect, useState } from "react";
import { listAgents } from "../api/agents";
import AgentCard from "../components/AgentCard";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import Skeleton from "../components/Skeleton";
import type { AgentAssistant } from "../types/agent";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentAssistant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadAgents() {
      setLoading(true);
      setError("");
      try {
        const response = await listAgents();
        setAgents(response.items);
      } catch (requestError) {
        console.error(requestError);
        setError("加载智能助手失败，请稍后重试。");
      } finally {
        setLoading(false);
      }
    }

    void loadAgents();
  }, []);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Agents"
        title="AI Assistants"
        description="Choose an assistant for knowledge-grounded tasks and guided work."
      />
      {loading ? (
        <Skeleton rows={3} variant="card" />
      ) : error ? (
        <ErrorState title="Agents error" message={error} />
      ) : agents.length === 0 ? (
        <EmptyState title="No assistants" description="当前系统没有可用智能助手。" />
      ) : (
        <section className="card-grid">
          {agents.map((agent) => (
            <AgentCard agent={agent} key={agent.id} />
          ))}
        </section>
      )}
    </div>
  );
}
