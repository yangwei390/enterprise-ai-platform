import type { ChatCitation, ChatSource } from "./chat";

export type AgentAssistant = {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  recommended: boolean;
  metadata?: Record<string, unknown>;
};

export type AgentAssistantListResponse = {
  items: AgentAssistant[];
  total: number;
};

export type AgentStreamRequest = {
  agent_id: string;
  query: string;
  conversation_id?: number | null;
  knowledge_base_id?: number | null;
  memory_context?: string | null;
  metadata?: Record<string, unknown>;
};

export type AgentStreamEvent =
  | {
      event: "message_start";
      data: {
        conversation_id: number | null;
        role: string;
        agent_id?: string | null;
      };
    }
  | {
      event: "status";
      data: {
        status: string;
        message: string;
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
        status: string;
      };
    }
  | {
      event: "error";
      data: {
        message: string;
      };
    };
