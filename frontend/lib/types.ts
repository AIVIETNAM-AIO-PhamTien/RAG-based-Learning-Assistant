export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export type DocumentRead = {
  id: string;
  name: string;
  status: DocumentStatus;
  page_count: number | null;
  error_message: string | null;
  created_at: string;
};

export type ChatSession = {
  id: string;
  title: string | null;
  created_at: string;
};

export type Citation = {
  index: number;
  chunk_id: string;
  doc_id: string;
  doc_name: string;
  page: number;
  text: string;
  snippet: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
};

export type ChatStreamEvent =
  | { type: "token"; text: string }
  | { type: "citations"; citations: Citation[]; citation_coverage?: number }
  | { type: "done" }
  | { type: "error"; message: string };
