import type { ChatCitation, ChatSource } from "../types/chat";
import type { CitationView } from "../types/citation";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function fieldText(value: unknown): string {
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  if (typeof value === "number") {
    return String(value);
  }
  return "";
}

export function buildCitationView(
  citation: ChatCitation,
  sources: ChatSource[],
  index: number
): CitationView {
  const source = sources.find((item) => sourceKey(item) === sourceKey(citation));
  const metadata = {
    ...(source?.metadata ?? {}),
    ...(citation.metadata ?? {})
  };
  const articleLabels = Array.isArray(metadata.article_labels)
    ? metadata.article_labels.map((item) => String(item)).join("、")
    : "";
  const articleRange = metadata.article_start || metadata.article_end
    ? `第${metadata.article_start ?? "?"}条${
        metadata.article_end && metadata.article_end !== metadata.article_start
          ? ` - 第${metadata.article_end}条`
          : ""
      }`
    : "";

  return {
    id: `${sourceKey(citation)}:${index}`,
    title: fieldText(citation.source ?? source?.source) || "未知文档",
    section: fieldText(metadata.chapter_title ?? metadata.heading_title ?? metadata.section_title),
    article: articleLabels || articleRange,
    text: fieldText(source?.text) || fieldText(citation.text_preview) || "当前来源没有返回原文。",
    citation,
    source
  };
}

function sourceKey(value: ChatCitation | ChatSource) {
  return `${value.document_id ?? "-"}:${value.chunk_index ?? "-"}:${value.source ?? "-"}`;
}
