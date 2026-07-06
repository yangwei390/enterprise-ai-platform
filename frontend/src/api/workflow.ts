import { apiRequest } from "./client";
import type { WorkflowResult } from "../types/common";

export function planAndRunWorkflow(task: string) {
  return apiRequest<WorkflowResult>("/workflows/plan-and-run", {
    method: "POST",
    body: { task }
  });
}
