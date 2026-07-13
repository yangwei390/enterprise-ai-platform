import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";

export default function ChatPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Chat"
        title="Workspace Chat"
        description="A user-facing chat surface for future RAG conversation integration."
      />
      <section className="workspace-panel chat-shell">
        <div className="message-list">
          <EmptyState
            title="Chat is not connected yet"
            description="Sprint W1 only builds the route, layout, and placeholder interaction surface."
          />
        </div>
        <div className="composer">
          <input disabled placeholder="Chat business logic will be connected in a later sprint." />
          <button type="button" disabled>Send</button>
        </div>
      </section>
    </div>
  );
}
