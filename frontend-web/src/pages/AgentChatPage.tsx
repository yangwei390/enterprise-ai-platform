import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAgent, streamAgentChat } from "../api/agents";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import Loading from "../components/Loading";
import MarkdownMessage from "../components/MarkdownMessage";
import MessageComposer from "../components/MessageComposer";
import PageHeader from "../components/PageHeader";
import { useToast } from "../components/Toast";
import { DEFAULT_KNOWLEDGE_BASE_ID } from "../config/defaults";
import { useAutoScroll } from "../hooks/useAutoScroll";
import type { AgentAssistant } from "../types/agent";
import type { CitationView } from "../types/citation";
import type { UiChatMessage } from "../types/chat";
import { buildCitationView } from "../utils/citations";

export default function AgentChatPage() {
  const { showToast } = useToast();
  const { agentId } = useParams();
  const [agent, setAgent] = useState<AgentAssistant | null>(null);
  const [agentLoading, setAgentLoading] = useState(true);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<UiChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState(DEFAULT_KNOWLEDGE_BASE_ID);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [agentStatus, setAgentStatus] = useState("");
  const [selectedCitation, setSelectedCitation] = useState<CitationView | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const { enableFollow, handleScroll, scrollToBottom } = useAutoScroll(messageListRef);

  useEffect(() => {
    async function loadAgent() {
      if (!agentId) {
        setAgentLoading(false);
        return;
      }
      setAgentLoading(true);
      setError("");
      try {
        setAgent(await getAgent(agentId));
      } catch (requestError) {
        console.error(requestError);
        setError("加载智能助手失败，请稍后重试。");
      } finally {
        setAgentLoading(false);
      }
    }

    void loadAgent();
  }, [agentId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, agentStatus, scrollToBottom]);

  useEffect(() => {
    if (!selectedCitation) {
      return undefined;
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedCitation(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedCitation]);

  async function handleSubmit() {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || streaming || !agentId) {
      return;
    }

    const userMessage: UiChatMessage = {
      id: `agent-user-${Date.now()}`,
      role: "user",
      content: trimmedQuery,
      status: "complete",
      citations: [],
      sources: []
    };
    const assistantId = `agent-assistant-${Date.now()}`;
    const assistantMessage: UiChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
      citations: [],
      sources: []
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    enableFollow();
    setQuery("");
    setStreaming(true);
    setAgentStatus("正在分析问题");
    setError("");
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamAgentChat(
        {
          agent_id: agentId,
          query: trimmedQuery,
          conversation_id: conversationId,
          knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
          metadata: {
            workspace: "frontend-web"
          }
        },
        {
          onEvent: (event) => {
            if (event.event === "message_start" && event.data.conversation_id) {
              setConversationId(event.data.conversation_id);
            }
            if (event.event === "status") {
              setAgentStatus(event.data.message);
            }
            if (event.event === "answer_delta") {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + event.data.delta }
                    : message
                )
              );
            }
            if (event.event === "citations") {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        citations: event.data.citations,
                        sources: event.data.sources
                      }
                    : message
                )
              );
            }
            if (event.event === "completed") {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        id: event.data.message_id ? String(event.data.message_id) : message.id,
                        content: event.data.answer || message.content,
                        citations: event.data.citations,
                        sources: event.data.sources,
                        status: "complete"
                      }
                    : message
                )
              );
              if (event.data.conversation_id) {
                setConversationId(event.data.conversation_id);
              }
              setAgentStatus("");
            }
            if (event.event === "error") {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        status: "error",
                        error: event.data.message || "智能助手执行失败，请稍后重试。"
                      }
                    : message
                )
              );
              setAgentStatus("");
            }
          }
        },
        controller.signal
      );
    } catch (requestError) {
      if (controller.signal.aborted) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, status: "aborted" } : message
          )
        );
      } else {
        console.error(requestError);
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  status: "error",
                  error: "网络或服务异常，智能助手执行失败。"
                }
              : message
          )
        );
      }
      setAgentStatus("");
    } finally {
      setStreaming(false);
      abortControllerRef.current = null;
    }
  }

  function stopGeneration() {
    abortControllerRef.current?.abort();
    setAgentStatus("");
  }

  async function copyAnswer(content: string) {
    try {
      await navigator.clipboard.writeText(content);
      showToast("success", "回答已复制。");
    } catch (error) {
      console.error(error);
      showToast("error", "复制失败，请稍后重试。");
    }
  }

  function startNewConversation() {
    if (streaming) {
      return;
    }
    setConversationId(null);
    setMessages([]);
    setError("");
    setAgentStatus("");
  }

  function renderCitations(message: UiChatMessage) {
    if (message.citations.length === 0) {
      return null;
    }
    const citationViews = message.citations.map((citation, index) =>
      buildCitationView(citation, message.sources, index)
    );
    return (
      <div className="citation-list">
        {citationViews.map((citation) => (
          <button
            className="citation-chip"
            key={citation.id}
            type="button"
            onClick={() => setSelectedCitation(citation)}
          >
            <span>{citation.title}</span>
            {citation.section && <small>{citation.section}</small>}
            {citation.article && <small>{citation.article}</small>}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="page-stack agent-chat-page-stack">
      <PageHeader
        eyebrow="Agent Chat"
        title={agent?.name ?? "AI Assistant"}
        description={agent?.description ?? "让智能助手处理真实任务，并查看可理解的执行状态。"}
      />

      {agentLoading ? (
        <Loading label="Loading assistant..." />
      ) : error ? (
        <ErrorState title="Agent error" message={error} />
      ) : !agent ? (
        <EmptyState title="Assistant not found" description="当前智能助手不存在或不可用。" />
      ) : (
        <section className="agent-chat-layout">
          <aside className="agent-side-panel">
            <div className="panel-header">
              <h2>Assistant</h2>
              <button className="secondary-button" type="button" onClick={startNewConversation}>
                New
              </button>
            </div>
            <p>{agent.description}</p>
            <label className="kb-field">
              Knowledge Base ID
              <input
                value={knowledgeBaseId}
                onChange={(event) => setKnowledgeBaseId(event.target.value)}
                disabled={streaming}
                placeholder="可为空"
              />
            </label>
            <div className="agent-status-card">
              <span>Status</span>
              <strong>{agentStatus || (streaming ? "正在处理任务" : "Ready")}</strong>
            </div>
            <Link className="text-link" to={`/agents/${agent.id}`}>
              View Details
            </Link>
          </aside>

          <main className="chat-main">
            <div className="chat-title-row">
              <div>
                <h2>{conversationId ? `Conversation ${conversationId}` : "New Conversation"}</h2>
                <p>Agent answers stream from the server</p>
              </div>
              {streaming && (
                <button className="secondary-button" type="button" onClick={stopGeneration}>
                  Stop Generation
                </button>
              )}
            </div>
            <div className="message-list workspace-message-list" ref={messageListRef} onScroll={handleScroll}>
              {messages.length === 0 ? (
                <EmptyState
                  title="Start with a task"
                  description="输入任务后，智能助手会开始执行并返回答案。"
                />
              ) : (
                messages.map((message) => (
                  <article className={`message-bubble ${message.role}`} key={message.id}>
                    <div className="message-role">
                      {message.role === "user" ? "You" : agent.name}
                      {message.status === "streaming" && <span>{agentStatus || "Working..."}</span>}
                      {message.status === "aborted" && <span>Stopped</span>}
                      {message.status === "error" && <span>Failed</span>}
                    </div>
                    {message.role === "assistant" ? (
                      <MarkdownMessage content={message.content || (message.status === "streaming" ? "正在等待智能助手输出..." : "")} />
                    ) : (
                      <p>{message.content}</p>
                    )}
                    {message.error && <div className="message-error">{message.error}</div>}
                    {renderCitations(message)}
                    {message.role === "assistant" && message.content && message.status !== "streaming" && (
                      <div className="message-actions">
                        <button
                          type="button"
                          className="ghost-button"
                          onClick={() => void copyAnswer(message.content)}
                        >
                          Copy Answer
                        </button>
                      </div>
                    )}
                  </article>
                ))
              )}
            </div>
            <MessageComposer
              value={query}
              disabled={streaming}
              placeholder="请输入任务，例如：分析劳动法第十条的核心要求"
              onChange={setQuery}
              onSubmit={handleSubmit}
            />
          </main>

          {selectedCitation && (
            <>
              <button
                type="button"
                aria-label="Close source drawer"
                className="drawer-overlay"
                onClick={() => setSelectedCitation(null)}
              />
              <aside className="citation-drawer">
                <div className="panel-header">
                  <h2>Source</h2>
                  <button type="button" className="secondary-button" onClick={() => setSelectedCitation(null)}>
                    Close
                  </button>
                </div>
                <div className="citation-detail">
                  <div>
                    <span>Document</span>
                    <strong>{selectedCitation.title}</strong>
                  </div>
                  {selectedCitation.section && (
                    <div>
                      <span>Section</span>
                      <strong>{selectedCitation.section}</strong>
                    </div>
                  )}
                  {selectedCitation.article && (
                    <div>
                      <span>Article</span>
                      <strong>{selectedCitation.article}</strong>
                    </div>
                  )}
                  <div>
                    <span>Original Text</span>
                    <p>{selectedCitation.text}</p>
                  </div>
                </div>
              </aside>
            </>
          )}
        </section>
      )}
    </div>
  );
}
