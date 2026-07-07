import { FormEvent, useState } from "react";
import { sendChat } from "../api/chat";
import type { ChatResponse } from "../types/common";

export default function ChatPage() {
  const [query, setQuery] = useState("ViewModel 是什么？");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("2");
  const [conversationId, setConversationId] = useState("");
  const [scoreThreshold, setScoreThreshold] = useState("0.5");
  const [enableMemory, setEnableMemory] = useState(true);
  const [enableTools, setEnableTools] = useState(false);
  const [result, setResult] = useState<ChatResponse | { error: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const chatResult = result && "answer" in result ? result : null;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    try {
      const response = await sendChat({
        query,
        knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
        conversation_id: conversationId ? Number(conversationId) : null,
        enable_memory: enableMemory,
        enable_tools: enableTools,
        score_threshold: scoreThreshold ? Number(scoreThreshold) : null
      });
      setResult(response);
      if (!conversationId && response.conversation_id) {
        setConversationId(String(response.conversation_id));
      }
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  function handleNewConversation() {
    setConversationId("");
    setResult(null);
  }

  return (
    <div>
      <div className="page-title">
        <h2>Chat</h2>
        <p>RAG chat with memory, tools, citations, and metadata.</p>
      </div>
      <div className="two-column">
        <form className="card form" onSubmit={handleSubmit}>
          <label>
            Query
            <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <label>
            Knowledge Base ID
            <input
              value={knowledgeBaseId}
              onChange={(event) => setKnowledgeBaseId(event.target.value)}
            />
          </label>
          <label>
            Conversation ID
            <input value={conversationId} onChange={(event) => setConversationId(event.target.value)} />
          </label>
          <label>
            Score Threshold
            <input value={scoreThreshold} onChange={(event) => setScoreThreshold(event.target.value)} />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={enableMemory}
              onChange={(event) => setEnableMemory(event.target.checked)}
            />
            Enable Memory
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={enableTools}
              onChange={(event) => setEnableTools(event.target.checked)}
            />
            Enable Tools
          </label>
          <button type="submit" disabled={loading}>POST /chat</button>
          <button type="button" className="secondary" onClick={handleNewConversation}>
            New Conversation
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => setResult({ error: "/chat/stream UI streaming 后续接入" })}
          >
            /chat/stream
          </button>
        </form>
        <div className="stack">
          <div className="card">
            <h3>Answer</h3>
            <p className="answer">{chatResult?.answer ?? ""}</p>
            {chatResult && (
              <p className="muted">
                conversation_id: {String(chatResult.conversation_id)} · message_id: {String(chatResult.message_id)}
              </p>
            )}
          </div>
          <div className="card result-card">
            <h3>Citations / Metadata</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
