import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";

export default function HistoryPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="History"
        title="Workspace History"
        description="A timeline for conversations and agent runs once user history is connected."
      />
      <section className="workspace-panel">
        <EmptyState
          title="No history loaded"
          description="Conversation history integration is reserved for a later sprint."
        />
      </section>
    </div>
  );
}
