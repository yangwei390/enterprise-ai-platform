export type ApiResponse<T> = {
  code: number;
  message: string;
  data: T;
};

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type ToolDefinition = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  source: string;
  permission: string;
};

export type ToolResult = {
  name: string;
  success: boolean;
  result?: unknown;
  error?: string | null;
  metadata?: Record<string, unknown>;
};

export type ChatResponse = {
  query: string;
  answer: string;
  conversation_id?: number | null;
  message_id?: number | null;
  citations?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
};

export type WorkflowResult = {
  workflow_id: string;
  status: string;
  state: Record<string, unknown>;
  artifacts: Array<Record<string, unknown>>;
  logs: Array<Record<string, unknown>>;
  error?: string | null;
};

export type AgentResult = {
  task: string;
  status: string;
  answer?: string | null;
  steps: Array<Record<string, unknown>>;
  artifacts: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
  error?: string | null;
};
