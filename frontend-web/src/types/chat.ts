export type Conversation = {
  id: number;
  title: string | null;
  knowledge_base_id: number | null;
  created_at: string;
  updated_at: string;
};

export type ConversationListResponse = {
  items: Conversation[];
  total: number;
};

export type ConversationMessage = {
  id: number;
  conversation_id: number;
  role: "user" | "assistant" | string;
  content: string;
  metadata?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type ChatCitation = {
  source?: string | null;
  document_id?: number | null;
  knowledge_base_id?: number | null;
  chunk_index?: number | null;
  text_preview?: string | null;
  metadata?: Record<string, unknown>;
};

export type ChatSource = {
  id?: string | null;
  text?: string | null;
  document_id?: number | null;
  knowledge_base_id?: number | null;
  chunk_index?: number | null;
  source?: string | null;
  metadata?: Record<string, unknown>;
};

export type ChatStreamRequest = {
  query: string;
  conversation_id?: number | null;
  knowledge_base_id?: number | null;
  enable_memory?: boolean;
  enable_tools?: boolean;
};

export type ChatStreamEvent =
  | {
      event: "message_start";
      data: {
        conversation_id: number | null;
        role: string;
      };
    }
  | {
      event: "answer_delta";
      data: {
        delta: string;
      };
    }
  | {
      event: "citations";
      data: {
        citations: ChatCitation[];
        sources: ChatSource[];
      };
    }
  | {
      event: "completed";
      data: {
        conversation_id: number | null;
        message_id: number | null;
        answer: string;
        citations: ChatCitation[];
        sources: ChatSource[];
      };
    }
  | {
      event: "error";
      data: {
        message: string;
      };
    };

export type UiChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "complete" | "streaming" | "error" | "aborted";
  citations: ChatCitation[];
  sources: ChatSource[];
  error?: string;
  created_at?: string;
};
