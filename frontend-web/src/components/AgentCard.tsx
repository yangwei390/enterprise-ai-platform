import { Link } from "react-router-dom";
import type { AgentAssistant } from "../types/agent";

type AgentCardProps = {
  agent: AgentAssistant;
};

export default function AgentCard({ agent }: AgentCardProps) {
  return (
    <article className="card agent-card">
      <div className="agent-card-header">
        <h3>{agent.name}</h3>
        {agent.recommended && <span>Recommended</span>}
      </div>
      <p>{agent.description}</p>
      <ul className="capability-list">
        {agent.capabilities.slice(0, 4).map((capability) => (
          <li key={capability}>{capability}</li>
        ))}
      </ul>
      <div className="agent-card-actions">
        <Link className="text-link" to={`/agents/${agent.id}`}>
          Details
        </Link>
        <Link className="button-link" to={`/agents/${agent.id}/chat`}>
          Start Chat
        </Link>
      </div>
    </article>
  );
}
