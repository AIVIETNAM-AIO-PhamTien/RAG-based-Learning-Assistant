"use client";

import { useEffect, useMemo, useState } from "react";

import { ChatInput } from "@/components/chat/ChatInput";
import { DocumentStatusList } from "@/components/chat/DocumentStatusList";
import { DocumentUploader } from "@/components/chat/DocumentUploader";
import { FlashcardsPanel } from "@/components/chat/FlashcardsPanel";
import { MessageList } from "@/components/chat/MessageList";
import { createSession, generateFlashcards, getFlashcards, getSessionDocuments, streamChat, updateFlashcardStatus, uploadDocument } from "@/lib/api";
import type { ChatMessage, ChatSession, DocumentRead, Flashcard, FlashcardStats, FlashcardStatus } from "@/lib/types";

const SESSION_STORAGE_KEY = "aio-chat-session";
const EMPTY_FLASHCARD_STATS: FlashcardStats = { total: 0, not_reviewed: 0, learning: 0, known: 0 };

function nextMessageId() {
  return globalThis.crypto?.randomUUID?.() ?? String(Date.now());
}

function readStoredSession(): ChatSession | null {
  const rawSession = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (!rawSession) return null;

  try {
    const parsed = JSON.parse(rawSession) as Partial<ChatSession>;
    if (typeof parsed.id !== "string") return null;
    return {
      id: parsed.id,
      title: typeof parsed.title === "string" ? parsed.title : null,
      created_at: typeof parsed.created_at === "string" ? parsed.created_at : new Date().toISOString(),
    };
  } catch {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    return null;
  }
}

function storeSession(session: ChatSession): void {
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
}

export default function Home() {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFlashcardsOpen, setIsFlashcardsOpen] = useState(false);
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [flashcardStats, setFlashcardStats] = useState<FlashcardStats>(EMPTY_FLASHCARD_STATS);
  const [isGeneratingFlashcards, setIsGeneratingFlashcards] = useState(false);

  const hasReadyDocument = useMemo(
    () => documents.some((document) => document.status === "ready"),
    [documents],
  );

  useEffect(() => {
    let isMounted = true;

    async function loadSession() {
      const storedSession = readStoredSession();
      if (storedSession) {
        try {
          const storedDocuments = await getSessionDocuments(storedSession.id);
          if (!isMounted) return;
          setSession(storedSession);
          setDocuments(storedDocuments);
          const cards = await getFlashcards(storedSession.id);
          if (!isMounted) return;
          setFlashcards(cards.flashcards);
          setFlashcardStats(cards.stats);
          return;
        } catch (caught: unknown) {
          if (!isMounted) return;
          setSession(storedSession);
          setError(caught instanceof Error ? caught.message : String(caught));
          return;
        }
      }

      try {
        const createdSession = await createSession();
        if (!isMounted) return;
        storeSession(createdSession);
        setSession(createdSession);
      } catch (caught: unknown) {
        if (isMounted) setError(caught instanceof Error ? caught.message : String(caught));
      }
    }

    void loadSession();
    return () => {
      isMounted = false;
    };
  }, []);

  async function refreshDocuments(sessionId: string) {
    setDocuments(await getSessionDocuments(sessionId));
  }

  async function onUpload(file: File) {
    if (!session) return;
    setIsUploading(true);
    setError(null);
    try {
      const uploaded = await uploadDocument(session.id, file);
      setDocuments((current) => [uploaded, ...current.filter((item) => item.id !== uploaded.id)]);
      await refreshDocuments(session.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsUploading(false);
    }
  }

  async function onSend(content: string) {
    if (!session || isStreaming) return;
    setError(null);
    const assistantId = nextMessageId();
    setMessages((current) => [
      ...current,
      { id: nextMessageId(), role: "user", content },
      { id: assistantId, role: "assistant", content: "", citations: [], isStreaming: true },
    ]);
    setIsStreaming(true);

    try {
      await streamChat(session.id, content, (event) => {
        if (event.type === "token") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: `${message.content}${event.text}` }
                : message,
            ),
          );
        }
        if (event.type === "citations") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, citations: event.citations } : message,
            ),
          );
        }
        if (event.type === "error") {
          setError(event.message);
        }
        if (event.type === "done") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, isStreaming: false } : message,
            ),
          );
        }
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, isStreaming: false } : message,
        ),
      );
    } finally {
      setIsStreaming(false);
    }
  }

  async function onGenerateFlashcards(topic: string, count: 5 | 10 | 15) {
    if (!session) return;
    setIsGeneratingFlashcards(true);
    setError(null);
    try {
      const result = await generateFlashcards(session.id, topic, count);
      setFlashcards(result.flashcards);
      setFlashcardStats(result.stats);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsGeneratingFlashcards(false);
    }
  }

  async function onFlashcardStatusChange(flashcardId: string, status: FlashcardStatus) {
    if (!session) return;
    try {
      await updateFlashcardStatus(session.id, flashcardId, status);
      const refreshed = await getFlashcards(session.id);
      setFlashcards(refreshed.flashcards);
      setFlashcardStats(refreshed.stats);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  return (
    <main className="min-h-screen text-foreground">
      <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="hidden border-r border-border bg-background p-4 lg:flex lg:flex-col">
          <div className="flex items-center gap-3 rounded-3xl border border-border bg-card p-3">
            <div className="flex size-11 items-center justify-center rounded-2xl border border-border/70 bg-muted text-xl">
              🎓
            </div>
            <div>
              <p className="text-base font-semibold tracking-[-0.03em]">AIO</p>
              <p className="text-xs text-muted-foreground">AI Tutor</p>
            </div>
          </div>

          <button
            type="button"
            disabled
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-2xl bg-muted px-4 py-3 text-sm font-semibold text-foreground disabled:cursor-not-allowed disabled:opacity-65"
          >
            <span>＋</span>
            New Session
          </button>

          <section className="mt-7 min-h-0 flex-1">
            <p className="px-1 font-mono text-[11px] tracking-[0.2em] text-muted-foreground uppercase">History</p>
            <div className="mt-3 rounded-2xl border border-border/70 bg-muted/35 p-3">
              <div className="rounded-xl bg-accent/12 p-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="truncate text-sm font-medium text-foreground">
                    {session?.title ?? documents[0]?.name ?? "Current study session"}
                  </p>
                  <span className="text-xs text-accent">●</span>
                </div>
                <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                  {session ? session.id.slice(0, 8) : "starting"}
                </p>
              </div>
              <p className="mt-3 px-1 text-xs leading-5 text-muted-foreground">
                Saved history, rename, and delete controls need session APIs and are not wired in this UI-only pass.
              </p>
            </div>
          </section>

          <div className="mt-5 flex items-center justify-between rounded-2xl border border-border bg-card p-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex size-9 items-center justify-center rounded-full bg-muted text-xs text-muted-foreground">
                TN
              </div>
              <div className="min-w-0">
                <p className="truncate text-xs font-medium text-foreground">student@aio.local</p>
                <p className="text-[11px] text-muted-foreground">Focus mode</p>
              </div>
            </div>
            <span className="text-muted-foreground">↪</span>
          </div>
        </aside>

        <section className={`grid min-h-screen min-w-0 transition-[grid-template-columns] duration-300 ${isFlashcardsOpen ? "xl:grid-cols-[minmax(0,1fr)_380px]" : "xl:grid-cols-[minmax(0,1fr)_0px]"}`}>
          <div className="flex min-h-screen min-w-0 flex-col bg-background">
            <header className="flex min-h-16 items-center justify-between gap-4 border-b border-border bg-background px-5 md:px-7">
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-muted text-lg text-accent">
                  ✦
                </span>
                <div className="min-w-0">
                  <h1 className="truncate text-sm font-semibold tracking-[-0.02em]">AIO</h1>
                  <p className="truncate text-xs text-muted-foreground">AI Tutor · always in focus mode</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="hidden rounded-full border border-border/70 px-3 py-1.5 font-mono text-[11px] text-muted-foreground sm:inline-flex">
                  {session ? session.id.slice(0, 8) : "starting"}
                </span>
                <button
                  type="button"
                  onClick={() => setIsFlashcardsOpen((current) => !current)}
                  className="inline-flex items-center gap-2 rounded-full border border-accent/45 bg-accent/12 px-4 py-2 text-xs font-semibold text-accent transition hover:bg-accent/20"
                  aria-pressed={isFlashcardsOpen}
                >
                  📘 Flashcards ✨
                </button>
              </div>
            </header>

            {error ? (
              <div
                role="alert"
                className="border-b border-danger/30 bg-danger/10 px-6 py-3 text-xs text-danger"
              >
                {error}
              </div>
            ) : null}

            <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[320px_minmax(0,1fr)]">
              <aside className="border-b border-border/70 bg-background/30 p-4 xl:border-r xl:border-b-0">
                <div className="space-y-5 xl:sticky xl:top-4">
                  <DocumentUploader disabled={!session || isUploading} onUpload={onUpload} />
                  <section className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h2 className="text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase">
                        Session documents
                      </h2>
                      {isUploading ? (
                        <span className="rounded-full bg-accent/12 px-2 py-1 font-mono text-[11px] text-accent">ingesting</span>
                      ) : null}
                    </div>
                    <DocumentStatusList documents={documents} />
                  </section>
                </div>
              </aside>

              <div className="min-h-0 overflow-y-auto aio-scrollbar">
                <MessageList messages={messages} />
              </div>
            </div>
            <ChatInput disabled={!session} sendDisabled={!hasReadyDocument || isStreaming} onSend={onSend} />
          </div>

          <aside className={`overflow-hidden border-l border-border bg-background transition-opacity duration-300 ${isFlashcardsOpen ? "opacity-100" : "pointer-events-none opacity-0"}`}>
            {/* Replaced by the connected flashcards panel below.
            <div className="flex h-full w-[380px] flex-col p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold tracking-[-0.03em]">Flashcards</h2>
                    <span className="rounded-full border border-border/70 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">0 cards</span>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">Scoped to this session</p>
                </div>
                <button
                  type="button"
                  onClick={() => setIsFlashcardsOpen(false)}
                  className="flex size-9 items-center justify-center rounded-full border border-border/70 text-muted-foreground transition hover:border-accent hover:text-accent"
                  aria-label="Close flashcards panel"
                >
                  ×
                </button>
              </div>

              <div className="mt-5 rounded-2xl border border-border bg-card p-4">
                <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                  <span>Known</span>
                  <span>Still learning</span>
                  <span>Not reviewed</span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-0 rounded-full bg-accent" />
                </div>
              </div>

              <div className="mt-5 flex flex-1 items-center justify-center rounded-[2rem] border border-dashed border-border bg-card p-6 text-center">
                <div>
                  <div className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-border/70 bg-muted text-2xl text-accent">✨</div>
                  <h3 className="mt-5 text-base font-semibold">No flashcards yet for this session</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                    Flashcard generation needs a follow-up data/API implementation. This panel is ready for that workflow.
                  </p>
                  <button
                    type="button"
                    disabled
                    className="mt-5 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    Generate flashcards
                  </button>
                </div>
              </div>
            </div> */}
            <FlashcardsPanel
              cards={flashcards}
              stats={flashcardStats}
              disabled={!session || !hasReadyDocument}
              generating={isGeneratingFlashcards}
              onClose={() => setIsFlashcardsOpen(false)}
              onGenerate={onGenerateFlashcards}
              onStatusChange={onFlashcardStatusChange}
            />
          </aside>
        </section>
      </div>
    </main>
  );
}
