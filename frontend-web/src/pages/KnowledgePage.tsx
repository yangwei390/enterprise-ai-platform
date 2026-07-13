import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import Skeleton from "../components/Skeleton";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listKnowledgeBases } from "../api/knowledge";
import type { KnowledgeBase } from "../types/knowledge";

export default function KnowledgePage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadKnowledgeBases();
  }, []);

  async function loadKnowledgeBases() {
    setLoading(true);
    setError(null);
    try {
      const response = await listKnowledgeBases();
      setKnowledgeBases(response.items);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法加载知识库，请稍后重试。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Knowledge"
        title="Knowledge Workspace"
        description="Browse available knowledge bases and manage documents for grounded AI answers."
      />

      {loading && <Skeleton rows={3} variant="card" />}
      {error && <ErrorState title="无法加载知识库" message={error} />}

      {!loading && !error && knowledgeBases.length === 0 && (
        <section className="workspace-panel">
          <EmptyState
            title="暂无知识库"
            description="当前没有可用知识库。请联系管理员创建知识库后再上传文档。"
          />
        </section>
      )}

      {!loading && !error && knowledgeBases.length > 0 && (
        <section className="knowledge-grid">
          {knowledgeBases.map((knowledgeBase) => (
            <article className="card knowledge-card" key={knowledgeBase.id}>
              <div>
                <h3>{knowledgeBase.name}</h3>
                <p>{knowledgeBase.description || "暂无描述。"}</p>
              </div>
              <div className="knowledge-card-footer">
                <span>Updated {formatDate(knowledgeBase.updated_at)}</span>
                <Link className="button-link" to={`/knowledge/${knowledgeBase.id}`}>
                  Open
                </Link>
              </div>
            </article>
          ))}
        </section>
      )}
    </div>
  );
}

function formatDate(value?: string | null) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}
