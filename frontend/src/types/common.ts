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

export type WorkflowTraceStep = {
  step: number;
  node_id: string;
  node_type: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  duration_ms: number;
  status: string;
  error?: string | null;
};

export type WorkflowRunResponseData = {
  answer?: string | null;
  output: Record<string, unknown>;
  node_outputs: Record<string, unknown>;
  trace: WorkflowTraceStep[];
  metadata: Record<string, unknown>;
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

export type AgentTraceStep = {
  step?: string | number;
  node?: string;
  name?: string;
  action?: string;
  tool_name?: string;
  input?: Record<string, unknown> | unknown;
  output?: Record<string, unknown> | unknown;
  duration_ms?: number;
  duration?: number;
  status?: string;
  error?: string | null;
  metadata?: Record<string, unknown>;
};

export type AgentTraceResult = {
  trace_id?: string | null;
  runtime?: string;
  agent_id?: string | null;
  agent_definition?: Record<string, unknown>;
  planner?: Record<string, unknown>;
  plan_steps?: Array<Record<string, unknown>>;
  graph_nodes?: Array<Record<string, unknown>>;
  tool_scope?: Record<string, unknown>;
  tool_calls?: Array<Record<string, unknown>>;
  retrieval?: Record<string, unknown>;
  evidence?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  checkpoint?: Record<string, unknown>;
  reflection?: Record<string, unknown>;
  final_answer?: Record<string, unknown>;
  errors?: Array<Record<string, unknown>>;
  timing?: Record<string, unknown>;
  token_usage?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export type AgentChatResponseData = {
  answer: string;
  action: string;
  tool_calls: Array<Record<string, unknown>>;
  observations: Array<Record<string, unknown>>;
  sources: Array<Record<string, unknown>>;
  citations: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
  trace: AgentTraceStep[];
};

export type MemoryDebugSnapshot = {
  provider?: string;
  session_count?: number | null;
  cache_count?: number | null;
  checkpoint_count?: number | null;
  metadata?: Record<string, unknown>;
};

export type CacheDebugSnapshot = {
  provider?: string;
  cache_count?: number | null;
  metadata?: Record<string, unknown>;
};

export type CheckpointsDebugSnapshot = {
  checkpoints: string[];
};

export type McpDebugSnapshot = {
  enabled?: boolean;
  configured_servers?: Array<Record<string, unknown>>;
  health?: Record<string, unknown> | Array<Record<string, unknown>>;
  discovered_tool_count?: number;
  tools?: Array<Record<string, unknown>>;
  registry_version?: string | number | null;
  audit?: Array<Record<string, unknown>>;
};
