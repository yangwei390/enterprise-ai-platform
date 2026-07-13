import { API_BASE_URL, apiRequest } from "./client";
import { readSseStream } from "./stream";
import type {
  AgentAssistant,
  AgentAssistantListResponse,
  AgentStreamEvent,
  AgentStreamRequest
} from "../types/agent";

type StreamHandlers = {
  onEvent: (event: AgentStreamEvent) => void;
};

export function listAgents() {
  return apiRequest<AgentAssistantListResponse>("/agent/assistants");
}

export function getAgent(agentId: string) {
  return apiRequest<AgentAssistant>(`/agent/assistants/${agentId}`);
}

export async function streamAgentChat(
  request: AgentStreamRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
) {
  const response = await fetch(`${API_BASE_URL}/agent/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request),
    signal
  });

  await readSseStream<AgentStreamEvent>(response, handlers.onEvent);
}
