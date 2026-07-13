import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { listConversations } from "../api/conversations";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import Skeleton from "../components/Skeleton";
import type { Conversation } from "../types/chat";
import { formatDateTime } from "../utils/date";

export default function HistoryPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const sortedConversations = useMemo(
    () =>
      [...conversations].sort(
        (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime()
      ),
    [conversations]
  );

  useEffect(() => {
    async function loadHistory() {
      setLoading(true);
      setError("");
      try {
        const response = await listConversations();
        setConversations(response.items);
      } catch (requestError) {
        console.error(requestError);
        setError("加载历史会话失败，请稍后重试。");
      } finally {
        setLoading(false);
      }
    }

    void loadHistory();
  }, []);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="History"
        title="Conversation History"
        description="查看当前系统已保存的真实会话，并继续之前的问答。"
      />

      <section className="workspace-panel">
        {loading ? (
          <Skeleton rows={6} />
        ) : error ? (
          <ErrorState title="History error" message={error} />
        ) : sortedConversations.length === 0 ? (
          <EmptyState
            title="暂无历史会话"
            description="开始一次知识库问答后，历史记录会显示在这里。"
          />
        ) : (
          <div className="history-list">
            {sortedConversations.map((conversation) => (
              <Link
                className="history-item"
                key={conversation.id}
                to={`/chat?conversationId=${conversation.id}`}
              >
                <div>
                  <strong>{conversation.title || "新对话"}</strong>
                  <span>{formatDateTime(conversation.updated_at || conversation.created_at)}</span>
                </div>
                <span className="history-action">Open</span>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
