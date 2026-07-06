import { apiRequest } from "./client";

export function createKnowledgeBase(data: {
  name: string;
  description?: string;
  embedding_model?: string;
}) {
  return apiRequest<unknown>("/kb", {
    method: "POST",
    body: data
  });
}

export function uploadDocument(knowledgeBaseId: number, file: File) {
  const formData = new FormData();
  formData.append("knowledge_base_id", String(knowledgeBaseId));
  formData.append("file", file);
  return apiRequest<unknown>("/documents/upload", {
    method: "POST",
    body: formData
  });
}

export function parseDocument(documentId: number) {
  return apiRequest<unknown>(`/documents/${documentId}/parse`, {
    method: "POST"
  });
}
