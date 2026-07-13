import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  deleteDocument,
  getKnowledgeBase,
  listKnowledgeDocuments,
  parseDocument,
  uploadDocument
} from "../api/knowledge";
import ConfirmDialog from "../components/ConfirmDialog";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import Loading from "../components/Loading";
import PageHeader from "../components/PageHeader";
import { useToast } from "../components/Toast";
import type { KnowledgeBase, KnowledgeDocument } from "../types/knowledge";

const SUPPORTED_FILE_ACCEPT = ".pdf,.txt,.md";
const SUPPORTED_FILE_LABEL = "PDF, TXT, Markdown (.md)";

type DocumentStatusView = {
  label: string;
  tone: "waiting" | "processing" | "ready" | "failed" | "unknown";
  isPending: boolean;
};

export default function KnowledgeDetailPage() {
  const { showToast } = useToast();
  const { knowledgeBaseId } = useParams();
  const numericKnowledgeBaseId = Number(knowledgeBaseId);
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeDocument | null>(null);

  const hasPendingDocuments = useMemo(
    () => documents.some((document) => getDocumentStatus(document).isPending),
    [documents]
  );

  useEffect(() => {
    if (!Number.isFinite(numericKnowledgeBaseId)) {
      setError("知识库不存在或链接无效。");
      setLoading(false);
      return;
    }
    void loadPage();
  }, [numericKnowledgeBaseId]);

  useEffect(() => {
    if (!hasPendingDocuments || !Number.isFinite(numericKnowledgeBaseId)) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      void loadDocuments({ silent: true });
    }, 4000);

    return () => window.clearInterval(timer);
  }, [hasPendingDocuments, numericKnowledgeBaseId]);

  async function loadPage() {
    setLoading(true);
    setError(null);
    try {
      const [knowledgeBaseResult, documentResult] = await Promise.all([
        getKnowledgeBase(numericKnowledgeBaseId),
        listKnowledgeDocuments(numericKnowledgeBaseId)
      ]);
      setKnowledgeBase(knowledgeBaseResult);
      setDocuments(documentResult.items);
    } catch (caught) {
      setError(toUserError(caught, "无法加载知识库，请稍后重试。"));
    } finally {
      setLoading(false);
    }
  }

  async function loadDocuments(options: { silent?: boolean } = {}) {
    if (!options.silent) {
      setDocumentsLoading(true);
    }
    try {
      const result = await listKnowledgeDocuments(numericKnowledgeBaseId);
      setDocuments(result.items);
    } catch (caught) {
      if (!options.silent) {
        setError(toUserError(caught, "无法加载文档列表。"));
      }
    } finally {
      if (!options.silent) {
        setDocumentsLoading(false);
      }
    }
  }

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setUploadError("请先选择一个文档。");
      return;
    }

    setUploading(true);
    setUploadError(null);
    try {
      const uploadedDocument = await uploadDocument(numericKnowledgeBaseId, file);
      setFile(null);
      setDocuments((current) => [uploadedDocument, ...current]);
      showToast("success", "文档已上传，正在处理。");
      void processUploadedDocument(uploadedDocument.id);
      void loadDocuments({ silent: true });
    } catch (caught) {
      const message = toUserError(caught, "文件上传失败，请检查文件格式后重试。");
      setUploadError(message);
      showToast("error", message);
    } finally {
      setUploading(false);
    }
  }

  async function processUploadedDocument(documentId: number) {
    try {
      await parseDocument(documentId);
    } catch (caught) {
      const message = toUserError(caught, "文档处理失败，请检查文件内容或稍后重试。");
      setUploadError(message);
      showToast("error", message);
    } finally {
      void loadDocuments({ silent: true });
    }
  }

  async function confirmDelete() {
    if (!deleteTarget) {
      return;
    }

    setDeletingId(deleteTarget.id);
    setError(null);
    try {
      await deleteDocument(deleteTarget.id);
      setDocuments((current) => current.filter((item) => item.id !== deleteTarget.id));
      showToast("success", "文档已删除。");
      setDeleteTarget(null);
    } catch (caught) {
      const message = toUserError(caught, "文档删除失败，请稍后重试。");
      setError(message);
      showToast("error", message);
      void loadDocuments({ silent: true });
    } finally {
      setDeletingId(null);
    }
  }

  if (loading) {
    return (
      <div className="page-stack">
        <PageHeader
          eyebrow="Knowledge"
          title="Knowledge Base"
          description="Loading knowledge base details."
        />
        <Loading label="正在加载知识库..." />
      </div>
    );
  }

  if (error && !knowledgeBase) {
    return (
      <div className="page-stack">
        <PageHeader
          eyebrow="Knowledge"
          title="Knowledge Base"
          description="The requested knowledge base could not be loaded."
        />
        <ErrorState title="无法加载知识库" message={error} />
        <Link className="button-link" to="/knowledge">返回知识库列表</Link>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Knowledge"
        title={knowledgeBase?.name ?? "Knowledge Base"}
        description={knowledgeBase?.description || "上传文档后，系统会处理并用于知识库问答。"}
      />

      <div className="knowledge-toolbar">
        <Link className="text-link" to="/knowledge">返回知识库列表</Link>
        <button
          type="button"
          className="secondary-button"
          disabled={documentsLoading}
          onClick={() => void loadDocuments()}
        >
          Refresh
        </button>
      </div>

      {error && <ErrorState title="操作失败" message={error} />}

      <section className="knowledge-detail-grid">
        <form className="card upload-card" onSubmit={handleUpload}>
          <h3>Upload Document</h3>
          <p>支持 {SUPPORTED_FILE_LABEL}。上传成功后会开始处理，处理完成后可用于问答。</p>
          <label className="file-field">
            <span>File</span>
            <input
              type="file"
              accept={SUPPORTED_FILE_ACCEPT}
              disabled={uploading}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          {file && <span className="file-name">{file.name}</span>}
          {uploadError && <p className="inline-error">{uploadError}</p>}
          <button type="submit" disabled={uploading || !file}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </form>

        <section className="card knowledge-summary-card">
          <h3>Knowledge Base</h3>
          <dl className="summary-list">
            <div>
              <dt>Documents</dt>
              <dd>{documents.length}</dd>
            </div>
            <div>
              <dt>Updated</dt>
              <dd>{formatDate(knowledgeBase?.updated_at)}</dd>
            </div>
          </dl>
        </section>
      </section>

      <section className="workspace-panel document-panel">
        <div className="panel-header">
          <h2>Documents</h2>
          {documentsLoading && <span className="subtle-text">Refreshing...</span>}
        </div>

        {documents.length === 0 ? (
          <EmptyState
            title="暂无文档"
            description="上传文档后，它们会显示在这里。"
          />
        ) : (
          <div className="document-list">
            {documents.map((document) => {
              const status = getDocumentStatus(document);
              return (
                <article className="document-row" key={document.id}>
                  <div className="document-main">
                    <strong>{getDocumentName(document)}</strong>
                    <span>
                      {formatFileSize(document.file_size)}
                      {" · "}
                      Uploaded {formatDate(document.created_at)}
                    </span>
                  </div>
                  <span className={`document-status ${status.tone}`}>
                    {status.label}
                  </span>
                  <div className="document-actions">
                    <button
                      type="button"
                      className="danger-ghost-button"
                      disabled={deletingId === document.id}
                      onClick={() => setDeleteTarget(document)}
                    >
                      {deletingId === document.id ? "Deleting..." : "Delete"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
      <ConfirmDialog
        open={deleteTarget !== null}
        title="删除文档"
        message={
          deleteTarget
            ? `确定删除“${getDocumentName(deleteTarget)}”吗？删除后该文档将不再用于知识库问答。`
            : ""
        }
        confirmLabel="删除"
        loading={deletingId !== null}
        onCancel={() => {
          if (deletingId === null) {
            setDeleteTarget(null);
          }
        }}
        onConfirm={() => void confirmDelete()}
      />
    </div>
  );
}

function getDocumentName(document: KnowledgeDocument) {
  return document.original_filename || document.filename || `Document ${document.id}`;
}

function getDocumentStatus(document: KnowledgeDocument): DocumentStatusView {
  if (document.parse_status === "success") {
    return { label: "已就绪", tone: "ready", isPending: false };
  }
  if (document.parse_status === "failed") {
    return { label: "处理失败", tone: "failed", isPending: false };
  }
  if (document.parse_status === "processing") {
    return { label: "处理中", tone: "processing", isPending: true };
  }
  if (document.parse_status === "pending") {
    return { label: "等待处理", tone: "waiting", isPending: true };
  }
  return { label: "状态未知", tone: "unknown", isPending: false };
}

function toUserError(error: unknown, fallback: string) {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return fallback;
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

function formatFileSize(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
