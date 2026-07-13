import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAgent } from "../api/agents";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import Loading from "../components/Loading";
import PageHeader from "../components/PageHeader";
import type { AgentAssistant } from "../types/agent";

export default function AgentDetailPage() {
  const { agentId } = useParams();
  const [agent, setAgent] = useState<AgentAssistant | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadAgent() {
      if (!agentId) {
        setLoading(false);
        return;
      }
      setLoading(true);
      setError("");
      try {
        setAgent(await getAgent(agentId));
      } catch (requestError) {
        console.error(requestError);
        setError("加载智能助手详情失败，请稍后重试。");
      } finally {
        setLoading(false);
      }
    }

    void loadAgent();
  }, [agentId]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Agent Detail"
        title={agent?.name ?? "AI Assistant"}
        description={agent?.description ?? "查看智能助手能力并开始对话。"}
      />
      {loading ? (
        <Loading label="Loading assistant..." />
      ) : error ? (
        <ErrorState title="Agent error" message={error} />
      ) : !agent ? (
        <EmptyState title="Assistant not found" description="当前智能助手不存在或不可用。" />
      ) : (
        <section className="workspace-panel agent-detail-panel">
          <div>
            <h2>Capabilities</h2>
            <ul className="capability-list large">
              {agent.capabilities.map((capability) => (
                <li key={capability}>{capability}</li>
              ))}
            </ul>
          </div>
          <div className="agent-detail-actions">
            <Link className="text-link" to="/agents">
              Back to Assistants
            </Link>
            <Link className="button-link" to={`/agents/${agent.id}/chat`}>
              Start Conversation
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}
