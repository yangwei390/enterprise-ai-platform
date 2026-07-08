import { apiRequest } from "./client";
import type { AgentChatResponseData, AgentResult } from "../types/common";

export type AgentChatRequest = {
  query: string;
  knowledge_base_id?: number | null;
  conversation_id?: number | null;
  memory_context?: string | null;
  metadata?: Record<string, unknown>;
};

export function agentChat(request: AgentChatRequest) {
  return apiRequest<AgentChatResponseData>("/agent/chat", {
    method: "POST",
    body: request
  });
}

export function runAgent(data: {
  task: string;
  knowledge_base_id?: number | null;
  enable_tools: boolean;
  enable_memory: boolean;
}) {
  return apiRequest<AgentResult>("/agents/run", {
    method: "POST",
    body: data
  });
}
