import { API_BASE_URL } from "./client";
import { readSseStream } from "./stream";
import type { ChatStreamEvent, ChatStreamRequest } from "../types/chat";

type StreamHandlers = {
  onEvent: (event: ChatStreamEvent) => void;
};

export async function streamChat(
  request: ChatStreamRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
) {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  await readSseStream<ChatStreamEvent>(response, handlers.onEvent);
}
