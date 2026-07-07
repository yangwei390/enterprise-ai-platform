import { FormEvent, useState } from "react";
import { runRagTrace } from "../api/debug";
import type { RagTraceChunk, RagTraceResult } from "../api/debug";

type TraceResultState = RagTraceResult | { error: string } | null;

function ChunkCard({ chunk }: { chunk: RagTraceChunk }) {
  return (
    <article className="trace-chunk-card">
      <div className="trace-chunk-meta">
        <span>source: {chunk.source ?? "-"}</span>
        <span>chunk: {String(chunk.chunk_index)}</span>
        <span>score: {chunk.score === null ? "-" : chunk.score.toFixed(6)}</span>
      </div>
      <div className="trace-chunk-meta">
        <span>document_id: {String(chunk.document_id)}</span>
        <span>dense_rank: {String(chunk.dense_rank ?? "-")}</span>
        <span>sparse_rank: {String(chunk.sparse_rank ?? "-")}</span>
        <span>fusion_score: {String(chunk.fusion_score ?? "-")}</span>
        <span>rerank_score: {String(chunk.rerank_score ?? "-")}</span>
      </div>
      <pre className="trace-preview">{chunk.text_preview}</pre>
      <details>
        <summary>metadata</summary>
        <pre>{JSON.stringify(chunk.metadata, null, 2)}</pre>
      </details>
    </article>
  );
}

function ChunkSection({
  title,
  chunks
}: {
  title: string;
  chunks: RagTraceChunk[];
}) {
  return (
    <div className="card">
      <div className="section-header">
        <h3>{title}</h3>
        <span className="muted">{chunks.length} chunks</span>
      </div>
      <div className="trace-chunk-list">
        {chunks.length === 0 ? (
          <p className="muted">No chunks captured in this stage.</p>
        ) : (
          chunks.map((chunk, index) => <ChunkCard key={`${chunk.id ?? title}-${index}`} chunk={chunk} />)
        )}
      </div>
    </div>
  );
}

export default function RagDebugPage() {
  const [query, setQuery] = useState("ViewModel 是什么？");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("4");
  const [topK, setTopK] = useState("10");
  const [scoreThreshold, setScoreThreshold] = useState("0.0");
  const [metadataFilter, setMetadataFilter] = useState("");
  const [result, setResult] = useState<TraceResultState>(null);
  const [loading, setLoading] = useState(false);
  const traceResult = result && "query" in result ? result : null;

  function parseMetadataFilter() {
    if (!metadataFilter.trim()) {
      return null;
    }
    return JSON.parse(metadataFilter) as Record<string, unknown>;
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    try {
      const response = await runRagTrace({
        query,
        knowledge_base_id: knowledgeBaseId ? Number(knowledgeBaseId) : null,
        top_k: topK ? Number(topK) : 10,
        score_threshold: scoreThreshold ? Number(scoreThreshold) : null,
        metadata_filter: parseMetadataFilter()
      });
      setResult(response);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-title">
        <h2>RAG Debug</h2>
        <p>Trace retrieval, rerank, context, and compression without calling the LLM.</p>
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
            Top K
            <input value={topK} onChange={(event) => setTopK(event.target.value)} />
          </label>
          <label>
            Score Threshold
            <input
              value={scoreThreshold}
              onChange={(event) => setScoreThreshold(event.target.value)}
            />
          </label>
          <label>
            Metadata Filter JSON
            <textarea
              placeholder='{"source":"Android.txt"}'
              value={metadataFilter}
              onChange={(event) => setMetadataFilter(event.target.value)}
            />
          </label>
          <button type="submit" disabled={loading}>
            Run Trace
          </button>
        </form>

        <div className="stack">
          <div className="card">
            <h3>Trace Summary</h3>
            {traceResult ? (
              <div className="trace-summary">
                <p>rewritten_query: {traceResult.rewritten_query ?? "-"}</p>
                <p>retriever_mode: {traceResult.retriever_mode ?? "-"}</p>
                <p>knowledge_base_id: {String(traceResult.knowledge_base_id)}</p>
              </div>
            ) : (
              <p className="muted">Run trace to inspect retrieved chunks.</p>
            )}
            {result && "error" in result && <p className="error-text">{result.error}</p>}
          </div>
          <div className="card result-card">
            <h3>Metadata</h3>
            <pre>{JSON.stringify(traceResult?.metadata ?? result ?? {}, null, 2)}</pre>
          </div>
        </div>
      </div>

      {traceResult && (
        <div className="trace-grid">
          <ChunkSection title="Dense Chunks" chunks={traceResult.dense_chunks} />
          <ChunkSection title="Sparse Chunks" chunks={traceResult.sparse_chunks} />
          <ChunkSection title="Fused Chunks" chunks={traceResult.fused_chunks} />
          <ChunkSection title="Reranked Chunks" chunks={traceResult.reranked_chunks} />
          <ChunkSection title="Context Chunks" chunks={traceResult.context_chunks} />
          <div className="card">
            <h3>Context Text Preview</h3>
            <pre>{traceResult.context_text_preview ?? ""}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
