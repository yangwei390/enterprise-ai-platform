import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { listAgents } from "../api/agents";
import { listConversations } from "../api/conversations";
import AgentCard from "../components/AgentCard";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import Skeleton from "../components/Skeleton";
import type { AgentAssistant } from "../types/agent";
import type { Conversation } from "../types/chat";
import { formatRelativeTime } from "../utils/date";

export default function HomePage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState("");
  const [agents, setAgents] = useState<AgentAssistant[]>([]);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [agentsError, setAgentsError] = useState("");

  const recentConversations = useMemo(
    () =>
      [...conversations]
        .sort((left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime())
        .slice(0, 5),
    [conversations]
  );
  const featuredAgents = agents.slice(0, 3);

  useEffect(() => {
    async function loadConversations() {
      setConversationsLoading(true);
      setConversationsError("");
      try {
        const response = await listConversations();
        setConversations(response.items);
      } catch (requestError) {
        console.error(requestError);
        setConversationsError("最近对话加载失败，请稍后重试。");
      } finally {
        setConversationsLoading(false);
      }
    }

    async function loadAgents() {
      setAgentsLoading(true);
      setAgentsError("");
      try {
        const response = await listAgents();
        setAgents(response.items);
      } catch (requestError) {
        console.error(requestError);
        setAgentsError("智能助手加载失败，请稍后重试。");
      } finally {
        setAgentsLoading(false);
      }
    }

    void loadConversations();
    void loadAgents();
  }, []);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Workspace"
        title="Enterprise AI Workspace"
        description="你好，今天想处理什么？使用知识库问答、AI 助手或上传资料开始工作。"
      />

      <section className="hero-panel">
        <div>
          <h2>Start from the task, not the tool.</h2>
          <p>
            Continue a recent conversation, ask your knowledge base, or choose an assistant for guided work.
          </p>
        </div>
        <div className="quick-actions">
          <Link className="quick-action-card" to="/chat">
            <strong>知识库问答</strong>
            <span>进入 Chat，基于企业资料获得答案。</span>
          </Link>
          <Link className="quick-action-card" to="/agents">
            <strong>使用 AI Agent</strong>
            <span>选择真实可用的 AI Assistant。</span>
          </Link>
          <Link className="quick-action-card" to="/knowledge">
            <strong>上传文档</strong>
            <span>进入 Knowledge，选择知识库后上传资料。</span>
          </Link>
        </div>
      </section>

      <section className="home-grid">
        <div className="workspace-panel home-section">
          <div className="panel-header">
            <h2>Recent Conversations</h2>
            <Link className="text-link" to="/history">View all</Link>
          </div>
          {conversationsLoading ? (
            <Skeleton rows={4} />
          ) : conversationsError ? (
            <ErrorState title="Conversations error" message={conversationsError} />
          ) : recentConversations.length === 0 ? (
            <EmptyState
              title="还没有对话"
              description="开始第一次知识库问答后，最近对话会显示在这里。"
            />
          ) : (
            <div className="history-list compact">
              {recentConversations.map((conversation) => (
                <Link
                  className="history-item"
                  key={conversation.id}
                  to={`/chat?conversationId=${conversation.id}`}
                >
                  <strong>{conversation.title || "新对话"}</strong>
                  <span>{formatRelativeTime(conversation.updated_at || conversation.created_at)}</span>
                </Link>
              ))}
            </div>
          )}
        </div>

        <div className="workspace-panel home-section">
          <div className="panel-header">
            <h2>AI Assistants</h2>
            <Link className="text-link" to="/agents">View all</Link>
          </div>
          {agentsLoading ? (
            <Skeleton rows={3} variant="card" />
          ) : agentsError ? (
            <ErrorState title="Assistants error" message={agentsError} />
          ) : featuredAgents.length === 0 ? (
            <EmptyState title="No assistants" description="当前系统没有可用智能助手。" />
          ) : (
            <div className="home-agent-grid">
              {featuredAgents.map((agent) => (
                <AgentCard agent={agent} key={agent.id} />
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
