import { apiRequest } from "./client";
import type {
  DocumentParseResult,
  KnowledgeBase,
  KnowledgeBaseListResponse,
  KnowledgeDocument,
  KnowledgeDocumentListResponse
} from "../types/knowledge";

export function listKnowledgeBases() {
  return apiRequest<KnowledgeBaseListResponse>("/kb");
}

export function getKnowledgeBase(knowledgeBaseId: number) {
  return apiRequest<KnowledgeBase>(`/kb/${knowledgeBaseId}`);
}

export function listKnowledgeDocuments(knowledgeBaseId: number) {
  return apiRequest<KnowledgeDocumentListResponse>(`/kb/${knowledgeBaseId}/documents`);
}

export function uploadDocument(knowledgeBaseId: number, file: File) {
  const formData = new FormData();
  formData.append("knowledge_base_id", String(knowledgeBaseId));
  formData.append("file", file);

  return apiRequest<KnowledgeDocument>("/documents/upload", {
    method: "POST",
    body: formData
  });
}

export function parseDocument(documentId: number) {
  return apiRequest<DocumentParseResult>(`/documents/${documentId}/parse`, {
    method: "POST"
  });
}

export function deleteDocument(documentId: number) {
  return apiRequest<{ deleted: boolean }>(`/documents/${documentId}`, {
    method: "DELETE"
  });
}
