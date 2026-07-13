import { apiRequest } from "./client";
import type {
  Conversation,
  ConversationListResponse,
  ConversationMessage
} from "../types/chat";

export function createConversation(data: {
  title?: string | null;
  knowledge_base_id?: number | null;
}) {
  return apiRequest<Conversation>("/conversations", {
    method: "POST",
    body: data
  });
}

export function listConversations() {
  return apiRequest<ConversationListResponse>("/conversations");
}

export function listConversationMessages(conversationId: number) {
  return apiRequest<ConversationMessage[]>(`/conversations/${conversationId}/messages`);
}
