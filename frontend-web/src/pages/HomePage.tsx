import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";

const featureCards = [
  {
    title: "Ask knowledge",
    description: "Start from enterprise knowledge and get grounded answers."
  },
  {
    title: "Run agents",
    description: "Use guided AI workers for task-oriented workflows."
  },
  {
    title: "Review history",
    description: "Continue prior work without losing context."
  }
];

export default function HomePage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Workspace"
        title="Enterprise AI Workspace"
        description="A focused user portal for chat, agents, knowledge access, and work history."
      />

      <section className="hero-panel">
        <div>
          <h2>Work with your enterprise AI system from one place.</h2>
          <p>
            This Sprint establishes the user-facing shell. Business workflows will be connected in later sprints.
          </p>
        </div>
        <EmptyState
          title="Ready for integration"
          description="Chat, Agent, Knowledge, and History routes are available as stable workspace surfaces."
        />
      </section>

      <section className="card-grid">
        {featureCards.map((card) => (
          <article className="card" key={card.title}>
            <h3>{card.title}</h3>
            <p>{card.description}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
