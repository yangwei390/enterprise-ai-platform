import type { ChatCitation, ChatSource } from "./chat";

export type CitationView = {
  id: string;
  title: string;
  section: string;
  article: string;
  text: string;
  citation: ChatCitation;
  source?: ChatSource;
};
