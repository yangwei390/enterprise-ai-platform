import { apiRequest } from "./client";
import type { WorkflowResult, WorkflowRunResponseData } from "../types/common";

export type WorkflowRunRequest = {
  query: string;
  knowledge_base_id?: number | null;
  workflow_id?: string | null;
  inputs?: Record<string, unknown>;
};

export function runWorkflow(request: WorkflowRunRequest) {
  return apiRequest<WorkflowRunResponseData>("/workflow/run", {
    method: "POST",
    body: request
  });
}

export function planAndRunWorkflow(task: string) {
  return apiRequest<WorkflowResult>("/workflows/plan-and-run", {
    method: "POST",
    body: { task }
  });
}
