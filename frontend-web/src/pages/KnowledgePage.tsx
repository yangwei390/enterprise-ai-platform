import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";

export default function KnowledgePage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Knowledge"
        title="Knowledge Access"
        description="A future user-facing entry point for browsing available knowledge bases."
      />
      <section className="workspace-panel">
        <EmptyState
          title="Knowledge list is pending"
          description="Sprint W1 keeps this page as a stable route without calling backend knowledge APIs."
        />
      </section>
    </div>
  );
}
