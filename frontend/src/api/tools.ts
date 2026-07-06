import { apiRequest } from "./client";
import type { ToolDefinition, ToolResult } from "../types/common";

export function listTools() {
  return apiRequest<ToolDefinition[]>("/tools");
}

export function executeTool(name: string, argumentsValue: Record<string, unknown>) {
  return apiRequest<ToolResult>("/tools/execute", {
    method: "POST",
    body: {
      name,
      arguments: argumentsValue
    }
  });
}
