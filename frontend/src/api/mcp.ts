import { apiRequest } from "./client";
import type { ToolDefinition } from "../types/common";

export function registerMcpTool(data: {
  name: string;
  description: string;
  endpoint: string;
  method: string;
  headers: Record<string, string>;
  parameters: Record<string, unknown>;
  timeout: number;
}) {
  return apiRequest<ToolDefinition>("/mcp/tools/register", {
    method: "POST",
    body: data
  });
}

export function listMcpTools() {
  return apiRequest<ToolDefinition[]>("/mcp/tools");
}
