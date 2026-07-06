import { apiRequest } from "./client";
import type { AgentResult } from "../types/common";

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
