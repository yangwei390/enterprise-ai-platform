import { apiRequest } from "./client";
import type { ChatResponse } from "../types/common";

export type ChatRequest = {
  query: string;
  knowledge_base_id?: number | null;
  conversation_id?: number | null;
  enable_memory: boolean;
  enable_tools: boolean;
  score_threshold?: number | null;
};

export function sendChat(request: ChatRequest) {
  return apiRequest<ChatResponse>("/chat", {
    method: "POST",
    body: request
  });
}
