import type { ApiResponse } from "../types/common";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type RequestOptions = {
  method?: string;
  body?: unknown;
  headers?: Record<string, string>;
};

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const headers = options.body instanceof FormData
    ? options.headers
    : { "Content-Type": "application/json", ...options.headers };

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body instanceof FormData
      ? options.body
      : options.body === undefined
        ? undefined
        : JSON.stringify(options.body)
  });

  const payload = (await response.json()) as ApiResponse<T>;
  if (!response.ok || payload.code !== 0) {
    throw new Error(payload.message || `HTTP ${response.status}`);
  }
  return payload.data;
}
