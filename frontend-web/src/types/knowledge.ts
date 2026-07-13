export type KnowledgeBase = {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
};

export type KnowledgeBaseListResponse = {
  items: KnowledgeBase[];
  total: number;
};

export type KnowledgeDocument = {
  id: number;
  knowledge_base_id: number;
  filename: string;
  file_size: number;
  status: string;
  chunk_count: number;
  original_filename: string | null;
  mime_type: string | null;
  parse_status: string;
  parse_message: string | null;
  created_at: string;
  updated_at: string;
};

export type KnowledgeDocumentListResponse = {
  items: KnowledgeDocument[];
  total: number;
};

export type DocumentParseResult = {
  document_id: number;
  preview: string;
  text_length: number;
};
