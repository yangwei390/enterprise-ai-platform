import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { streamChat } from "../api/chat";
import {
  createConversation,
  listConversationMessages,
  listConversations
} from "../api/conversations";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import Loading from "../components/Loading";
import MarkdownMessage from "../components/MarkdownMessage";
import MessageComposer from "../components/MessageComposer";
import PageHeader from "../components/PageHeader";
import { useToast } from "../components/Toast";
import { DEFAULT_KNOWLEDGE_BASE_ID } from "../config/defaults";
import { useAutoScroll } from "../hooks/useAutoScroll";
import type { CitationView } from "../types/citation";
import type {
  ChatCitation,
  ChatSource,
  Conversation,
  ConversationMessage,
  UiChatMessage
} from "../types/chat";
import { buildCitationView, isRecord } from "../utils/citations";

function toUiMessage(message: ConversationMessage): UiChatMessage {
  const metadata = message.metadata ?? {};
  const citations = Array.isArray(metadata.citations)
    ? metadata.citations.filter(isRecord) as ChatCitation[]
    : [];
  const sources = Array.isArray(metadata.sources)
    ? metadata.sources.filter(isRecord) as ChatSource[]
    : [];
  return {
    id: String(message.id),
    role: message.role === "assistant" ? "assistant" : "user",
    content: message.content,
    status: "complete",
    citations,
    sources,
    created_at: message.created_at
  };
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

export default function ChatPage() {
  const { showToast } = useToast();
  const [searchParams, setSearchParams] = useSearchParams();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [messages, setMessages] = useState<UiChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState(DEFAULT_KNOWLEDGE_BASE_ID);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<CitationView | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const { enableFollow, handleScroll, scrollToBottom } = useAutoScroll(messageListRef);
  const linkedConversationId = useMemo(
    () => parseConversationId(searchParams.get("conversationId")),
    [searchParams]
  );

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId) ?? null,
    [activeConversationId, conversations]
  );

  useEffect(() => {
    void refreshConversations();
  }, []);

  useEffect(() => {
    if (linkedConversationId !== null && linkedConversationId !== activeConversationId) {
      setActiveConversationId(linkedConversationId);
    }
  }, [linkedConversationId, activeConversationId]);

  useEffect(() => {
    if (activeConversationId !== null) {
      void loadMessages(activeConversationId);
    }
  }, [activeConversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

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

  async function refreshConversations() {
    setConversationLoading(true);
    setError("");
    try {
      const response = await listConversations();
      setConversations(response.items);
      const requestedConversationId = parseConversationId(searchParams.get("conversationId"));
      if (requestedConversationId !== null) {
        setActiveConversationId(requestedConversationId);
      } else if (activeConversationId === null && response.items.length > 0) {
        setActiveConversationId(response.items[0].id);
      }
    } catch (requestError) {
      console.error(requestError);
      setError("加载会话列表失败，请稍后重试。");
    } finally {
      setConversationLoading(false);
    }
  }

  async function loadMessages(conversationId: number) {
    setMessagesLoading(true);
    setError("");
    try {
      const response = await listConversationMessages(conversationId);
      setMessages(response.map(toUiMessage));
    } catch (requestError) {
      console.error(requestError);
      setError("加载会话消息失败，请稍后重试。");
    } finally {
      setMessagesLoading(false);
    }
  }

  async function handleNewConversation() {
    setError("");
    try {
      const conversation = await createConversation({
        title: "New Chat",
        knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null
      });
      setConversations((current) => [conversation, ...current]);
      setActiveConversationId(conversation.id);
      setSearchParams({ conversationId: String(conversation.id) });
      setMessages([]);
    } catch (requestError) {
      console.error(requestError);
      setError("创建会话失败，请稍后重试。");
    }
  }

  async function handleSubmit() {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || streaming) {
      return;
    }

    const userMessage: UiChatMessage = {
      id: `local-user-${Date.now()}`,
      role: "user",
      content: trimmedQuery,
      status: "complete",
      citations: [],
      sources: []
    };
    const assistantId = `local-assistant-${Date.now()}`;
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
    setError("");
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChat(
        {
          query: trimmedQuery,
          conversation_id: activeConversationId,
          knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
          enable_memory: true,
          enable_tools: false
        },
        {
          onEvent: (event) => {
            if (event.event === "message_start" && event.data.conversation_id) {
              setActiveConversationId(event.data.conversation_id);
              setSearchParams({ conversationId: String(event.data.conversation_id) });
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
                setActiveConversationId(event.data.conversation_id);
                setSearchParams({ conversationId: String(event.data.conversation_id) });
              }
              void refreshConversations();
            }
            if (event.event === "error") {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        status: "error",
                        error: event.data.message || "生成回答失败，请稍后重试。"
                      }
                    : message
                )
              );
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
                  error: "网络或服务异常，生成回答失败。"
                }
              : message
          )
        );
      }
    } finally {
      setStreaming(false);
      abortControllerRef.current = null;
    }
  }

  function stopGeneration() {
    abortControllerRef.current?.abort();
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
    <div className="page-stack chat-page-stack">
      <PageHeader
        eyebrow="Chat"
        title="Workspace Chat"
        description="Ask enterprise knowledge questions with streaming RAG answers and readable citations."
      />

      <section className="chat-workspace">
        <aside className="conversation-panel">
          <div className="panel-header">
            <h2>Conversations</h2>
            <button type="button" onClick={() => void handleNewConversation()}>
              New
            </button>
          </div>
          <label className="kb-field">
            Knowledge Base ID
            <input
              value={knowledgeBaseId}
              onChange={(event) => setKnowledgeBaseId(event.target.value)}
              placeholder="可为空"
            />
          </label>
          {conversationLoading ? (
            <Loading label="Loading conversations..." />
          ) : conversations.length === 0 ? (
            <EmptyState title="No conversations" description="Create a new conversation to start." />
          ) : (
            <div className="conversation-list">
              {conversations.map((conversation) => (
                <button
                  className={conversation.id === activeConversationId ? "conversation-item active" : "conversation-item"}
                  key={conversation.id}
                  type="button"
                  onClick={() => selectConversation(conversation.id)}
                >
                  <strong>{conversation.title || `Conversation ${conversation.id}`}</strong>
                  <span>{formatTime(conversation.updated_at)}</span>
                </button>
              ))}
            </div>
          )}
        </aside>

        <main className="chat-main">
          <div className="chat-title-row">
            <div>
              <h2>{activeConversation?.title || "New Conversation"}</h2>
              <p>Streaming answer with citations</p>
            </div>
            {streaming && (
              <button className="secondary-button" type="button" onClick={stopGeneration}>
                Stop Generation
              </button>
            )}
          </div>

          {error && <ErrorState title="Chat error" message={error} />}

          <div className="message-list workspace-message-list" ref={messageListRef} onScroll={handleScroll}>
            {messagesLoading ? (
              <Loading label="Loading messages..." />
            ) : messages.length === 0 ? (
              <EmptyState
                title="Empty conversation"
                description="Ask a question to start a grounded RAG conversation."
              />
            ) : (
              messages.map((message) => (
                <article className={`message-bubble ${message.role}`} key={message.id}>
                  <div className="message-role">
                    {message.role === "user" ? "You" : "AI"}
                    {message.status === "streaming" && <span>Streaming...</span>}
                    {message.status === "aborted" && <span>Stopped</span>}
                    {message.status === "error" && <span>Failed</span>}
                  </div>
                  {message.role === "assistant" ? (
                    <MarkdownMessage content={message.content || (message.status === "streaming" ? "正在生成回答..." : "")} />
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
            placeholder="请输入问题，例如：劳动法第十条讲什么？"
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
    </div>
  );

  function selectConversation(conversationId: number) {
    setActiveConversationId(conversationId);
    setSearchParams({ conversationId: String(conversationId) });
  }
}

function parseConversationId(value: string | null) {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}
