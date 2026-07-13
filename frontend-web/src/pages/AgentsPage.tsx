import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";

export default function AgentsPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Agents"
        title="Agent Workspace"
        description="Browse and run user-facing agents after the runtime integration sprint."
      />
      <section className="card-grid">
        <article className="card">
          <h3>Research Agent</h3>
          <p>Placeholder card for knowledge-grounded agent tasks.</p>
        </article>
        <article className="card">
          <h3>Tool Agent</h3>
          <p>Placeholder card for tool-assisted user workflows.</p>
        </article>
      </section>
      <EmptyState
        title="No live agents connected"
        description="Agent discovery and execution are intentionally out of scope for Sprint W1."
      />
    </div>
  );
}
