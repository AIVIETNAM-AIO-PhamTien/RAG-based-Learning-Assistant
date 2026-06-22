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

export type FlashcardStatus = "not_reviewed" | "learning" | "known";

export type Flashcard = {
  id: string;
  question: string;
  answer: string;
  status: FlashcardStatus;
  source_doc_name: string;
  source_page: number;
  created_at: string;
};

export type FlashcardStats = {
  total: number;
  not_reviewed: number;
  learning: number;
  known: number;
};

export type FlashcardsResponse = { flashcards: Flashcard[]; stats: FlashcardStats };
