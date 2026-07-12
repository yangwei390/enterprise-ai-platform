import { apiRequest } from "./client";
import type {
  CacheDebugSnapshot,
  CheckpointsDebugSnapshot,
  McpDebugSnapshot,
  MemoryDebugSnapshot
} from "../types/common";

export type RagTraceChunk = {
  id: string | null;
  document_id: number | null;
  knowledge_base_id: number | null;
  chunk_index: number | null;
  source: string | null;
  text_preview: string;
  score: number | null;
  dense_rank?: number | null;
  sparse_rank?: number | null;
  fusion_score?: number | null;
  sparse_score?: number | null;
  rerank_score?: number | null;
  metadata: Record<string, unknown>;
};

export type RagTraceResult = {
  query: string;
  rewritten_query: string | null;
  knowledge_base_id: number | null;
  retriever_mode: string | null;
  dense_chunks: RagTraceChunk[];
  sparse_chunks: RagTraceChunk[];
  fused_chunks: RagTraceChunk[];
  reranked_chunks: RagTraceChunk[];
  context_chunks: RagTraceChunk[];
  context_text_preview: string | null;
  metadata: Record<string, unknown>;
};

export type RagTraceRequest = {
  query: string;
  knowledge_base_id: number | null;
  top_k: number;
  score_threshold: number | null;
  metadata_filter?: Record<string, unknown> | null;
};

export function runRagTrace(request: RagTraceRequest) {
  return apiRequest<RagTraceResult>("/debug/rag-trace", {
    method: "POST",
    body: request
  });
}

export function runRetrieverCompare(request: RagTraceRequest) {
  return apiRequest<RagTraceResult>("/debug/retriever-compare", {
    method: "POST",
    body: request
  });
}

export function getMemoryDebug() {
  return apiRequest<MemoryDebugSnapshot>("/debug/memory");
}

export function getCacheDebug() {
  return apiRequest<CacheDebugSnapshot>("/debug/cache");
}

export function getCheckpointsDebug() {
  return apiRequest<CheckpointsDebugSnapshot>("/debug/checkpoints");
}

export function getMcpDebug() {
  return apiRequest<McpDebugSnapshot>("/debug/mcp");
}
