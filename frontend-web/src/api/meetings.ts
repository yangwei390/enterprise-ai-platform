import { API_BASE_URL, apiRequest } from "./client";

export const OWNER_ID = localStorage.getItem("meetingOwnerId") ?? "local-user";
const headers = { "X-Owner-Id": OWNER_ID };

export type Meeting = {
  id: number; title: string; participants: string[]; status: string; progress: number;
  error_message?: string; audio_filename?: string; duration_seconds?: number;
  current_version: number; approved_version?: number;
};
export type Segment = { id: number; speaker_id: string; speaker_name: string; start_time: number; end_time: number; text: string; confidence?: number };
export type Minutes = { version: number; content_revision: number; status: string; summary: string; topics: Array<Record<string, unknown>>; unresolved_questions: string[]; risks: string[]; decisions: Array<Record<string, unknown>>; action_items: Array<Record<string, unknown>> };
export type EmailDraft = { id: number; revision: number; to_addresses: string[]; cc_addresses: string[]; subject: string; body: string };

export const meetingsApi = {
  list: () => apiRequest<{ items: Meeting[]; total: number }>("/meetings", { headers }),
  create: (title: string, participants: string[]) => apiRequest<Meeting>("/meetings", { method: "POST", headers, body: { title, participants } }),
  upload: (id: number, file: File) => { const body = new FormData(); body.append("file", file); return apiRequest<Meeting>(`/meetings/${id}/audio`, { method: "POST", headers, body }); },
  process: (id: number) => apiRequest(`/meetings/${id}/process`, { method: "POST", headers }),
  retry: (id: number) => apiRequest(`/meetings/${id}/retry`, { method: "POST", headers }),
  get: (id: number) => apiRequest<Meeting>(`/meetings/${id}`, { headers }),
  transcript: (id: number) => apiRequest<{ segments: Segment[] }>(`/meetings/${id}/transcript`, { headers }),
  minutes: (id: number) => apiRequest<Minutes | null>(`/meetings/${id}/minutes`, { headers }),
  saveMinutes: (id: number, value: Minutes) => apiRequest<Minutes>(`/meetings/${id}/minutes`, { method: "PUT", headers, body: value }),
  speakers: (id: number, value: Array<{ speaker_id: string; display_name: string; representative_segment_id?: number }>) => apiRequest(`/meetings/${id}/speakers`, { method: "PUT", headers, body: value }),
  approve: (id: number) => apiRequest<Minutes>(`/meetings/${id}/approve`, { method: "POST", headers }),
  createDraft: (id: number) => apiRequest<EmailDraft>(`/meetings/${id}/email-draft`, { method: "POST", headers }),
  saveDraft: (id: number, value: EmailDraft) => apiRequest<EmailDraft>(`/meetings/${id}/email-draft`, { method: "PUT", headers, body: value }),
  send: (id: number, revision: number) => apiRequest(`/meetings/${id}/send`, { method: "POST", headers, body: { confirm: true, expected_draft_revision: revision } }),
  audio: async (id: number) => { const response = await fetch(`${API_BASE_URL}/meetings/${id}/audio`, { headers }); if (!response.ok) throw new Error("音频加载失败"); return URL.createObjectURL(await response.blob()); }
};
