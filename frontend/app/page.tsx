"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatFlashcardsPanel } from "@/components/chat/ChatFlashcardsPanel";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { DocumentStatusList } from "@/components/chat/DocumentStatusList";
import { DocumentUploader } from "@/components/chat/DocumentUploader";
import { MessageList } from "@/components/chat/MessageList";
import {
  createSession,
  deleteSession,
  generateSessionFlashcards,
  getSessionDocuments,
  getSessionMessages,
  getSessions,
  renameSession,
  streamChat,
  uploadDocument,
} from "@/lib/api";
import type { ChatMessage, ChatSession, DocumentRead, Flashcard } from "@/lib/types";

const SESSION_STORAGE_KEY = "aio-chat-session";
const SIDEBAR_COLLAPSED_STORAGE_KEY = "aio-sidebar-collapsed";
const SIDEBAR_EXPANDED_WIDTH_CLASS = "lg:grid-cols-[280px_minmax(0,1fr)]";
const SIDEBAR_COLLAPSED_WIDTH_CLASS = "lg:grid-cols-[64px_minmax(0,1fr)]";
const COMPOSER_SIDEBAR_EXPANDED_OFFSET_CLASS = "lg:left-[280px] xl:left-[600px]";
const COMPOSER_SIDEBAR_COLLAPSED_OFFSET_CLASS = "lg:left-[64px] xl:left-[384px]";
const DEFAULT_FLASHCARD_COUNT = 6;

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

function readStoredSidebarCollapsed(): boolean {
  const rawValue = window.localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY);
  if (!rawValue) return false;

  try {
    const parsed = JSON.parse(rawValue);
    if (typeof parsed !== "boolean") return false;
    return parsed;
  } catch {
    window.localStorage.removeItem(SIDEBAR_COLLAPSED_STORAGE_KEY);
    return false;
  }
}

function storeSidebarCollapsed(isCollapsed: boolean): void {
  window.localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, JSON.stringify(isCollapsed));
}

const sidebarAccount = {
  initials: "AI",
  label: "AIO Workspace",
  subtitle: "Local study session",
};

export default function Home() {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [sessionHistory, setSessionHistory] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [flashcardSourceCount, setFlashcardSourceCount] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [isGeneratingFlashcards, setIsGeneratingFlashcards] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flashcardsError, setFlashcardsError] = useState<string | null>(null);
  const [isFlashcardsOpen, setIsFlashcardsOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const hasReadyDocument = useMemo(
    () => documents.some((document) => document.status === "ready"),
    [documents],
  );

  const resetFlashcardsState = useCallback((): void => {
    setFlashcards([]);
    setFlashcardSourceCount(0);
    setFlashcardsError(null);
    setIsGeneratingFlashcards(false);
  }, []);

  const activateSession = useCallback(async (nextSession: ChatSession) => {
    const [nextDocuments, nextMessages] = await Promise.all([
      getSessionDocuments(nextSession.id),
      getSessionMessages(nextSession.id),
    ]);
    storeSession(nextSession);
    setSession(nextSession);
    setDocuments(nextDocuments);
    setMessages(nextMessages);
    resetFlashcardsState();
    setError(null);
  }, [resetFlashcardsState]);

  useEffect(() => {
    let isMounted = true;

    async function loadSessionHistory() {
      const history = await getSessions();
      if (!isMounted) return history;
      setSessionHistory(history);
      return history;
    }

    async function createAndActivateSession() {
      setIsCreatingSession(true);
      try {
        const createdSession = await createSession();
        if (!isMounted) return;
        storeSession(createdSession);
        setSession(createdSession);
        setDocuments([]);
        setMessages([]);
        resetFlashcardsState();
        setError(null);
        await loadSessionHistory();
      } catch (caught: unknown) {
        if (isMounted) setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        if (isMounted) setIsCreatingSession(false);
      }
    }

    async function loadSession() {
      const storedSession = readStoredSession();
      setIsSidebarCollapsed(readStoredSidebarCollapsed());
      const history = await loadSessionHistory();

      if (storedSession) {
        const matchedSession = history.find((item) => item.id === storedSession.id) ?? storedSession;
        try {
          await activateSession(matchedSession);
          return;
        } catch (caught: unknown) {
          if (!isMounted) return;
          storeSession(matchedSession);
          setSession(matchedSession);
          setDocuments([]);
          setMessages([]);
          resetFlashcardsState();
          setError(caught instanceof Error ? caught.message : String(caught));
          return;
        }
      }

      await createAndActivateSession();
    }

    void loadSession();
    return () => {
      isMounted = false;
    };
  }, [activateSession, resetFlashcardsState]);

  useEffect(() => {
    storeSidebarCollapsed(isSidebarCollapsed);
  }, [isSidebarCollapsed]);

  async function refreshDocuments(sessionId: string) {
    setDocuments(await getSessionDocuments(sessionId));
  }

  async function onNewSession() {
    if (!session || isCreatingSession || isStreaming || isUploading || isGeneratingFlashcards) return;

    setIsCreatingSession(true);
    setError(null);
    try {
      const createdSession = await createSession();
      storeSession(createdSession);
      setSession(createdSession);
      setDocuments([]);
      setMessages([]);
      resetFlashcardsState();
      setError(null);
      setSessionHistory(await getSessions());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsCreatingSession(false);
    }
  }

  async function onSelectSession(sessionId: string) {
    if (
      !session ||
      isCreatingSession ||
      isStreaming ||
      isUploading ||
      isGeneratingFlashcards ||
      session.id === sessionId
    ) {
      return;
    }

    const nextSession = sessionHistory.find((item) => item.id === sessionId);
    if (!nextSession) return;

    setError(null);
    try {
      await activateSession(nextSession);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function onRenameSession(sessionId: string, title: string) {
    if (!session || isCreatingSession || isStreaming || isUploading || isGeneratingFlashcards) return;

    setError(null);
    try {
      const updatedSession = await renameSession(sessionId, title);
      setSessionHistory((current) =>
        current.map((item) => (item.id === updatedSession.id ? updatedSession : item)),
      );
      if (session.id === updatedSession.id) {
        storeSession(updatedSession);
        setSession(updatedSession);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function onDeleteSession(sessionId: string) {
    if (!session || isCreatingSession || isStreaming || isUploading || isGeneratingFlashcards) return;

    setError(null);
    try {
      await deleteSession(sessionId);
      const remainingSessions = sessionHistory.filter((item) => item.id !== sessionId);
      setSessionHistory(remainingSessions);

      if (session.id !== sessionId) {
        return;
      }

      const nextSession = remainingSessions[0] ?? null;
      if (nextSession) {
        await activateSession(nextSession);
        return;
      }

      const createdSession = await createSession();
      storeSession(createdSession);
      setSession(createdSession);
      setDocuments([]);
      setMessages([]);
      resetFlashcardsState();
      setSessionHistory([createdSession]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function onUpload(file: File) {
    if (!session || isGeneratingFlashcards) return;
    setIsUploading(true);
    setError(null);
    try {
      const uploaded = await uploadDocument(session.id, file);
      setDocuments((current) => [uploaded, ...current.filter((item) => item.id !== uploaded.id)]);
      resetFlashcardsState();
      await refreshDocuments(session.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsUploading(false);
    }
  }

  async function onGenerateFlashcards() {
    if (!session || !hasReadyDocument || isGeneratingFlashcards) return;

    setFlashcardsError(null);
    setIsGeneratingFlashcards(true);
    try {
      const response = await generateSessionFlashcards(session.id, {
        flashcard_count: DEFAULT_FLASHCARD_COUNT,
      });
      setFlashcards(response.flashcards);
      setFlashcardSourceCount(response.sources.length);
    } catch (caught) {
      setFlashcardsError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsGeneratingFlashcards(false);
    }
  }

  async function onSend(content: string) {
    if (!session || isStreaming) return;
    const activeSessionId = session.id;
    const shouldRefreshSessionTitle = !session.title;
    setError(null);
    const assistantId = nextMessageId();
    setMessages((current) => [
      ...current,
      { id: nextMessageId(), role: "user", content },
      { id: assistantId, role: "assistant", content: "", citations: [], isStreaming: true },
    ]);
    setIsStreaming(true);

    try {
      await streamChat(activeSessionId, content, (event) => {
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

      if (shouldRefreshSessionTitle) {
        const updatedSessions = await getSessions();
        setSessionHistory(updatedSessions);
        const updatedSession = updatedSessions.find((item) => item.id === activeSessionId);
        if (updatedSession) {
          storeSession(updatedSession);
          setSession(updatedSession);
        }
      }
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

  const sessionIdLabel = session ? session.id.slice(0, 8) : "starting";
  const isSessionActionDisabled =
    !session || isCreatingSession || isStreaming || isUploading || isGeneratingFlashcards;
  const sidebarWidthClass = isSidebarCollapsed ? SIDEBAR_COLLAPSED_WIDTH_CLASS : SIDEBAR_EXPANDED_WIDTH_CLASS;
  const composerSidebarOffsetClass = isSidebarCollapsed
    ? COMPOSER_SIDEBAR_COLLAPSED_OFFSET_CLASS
    : COMPOSER_SIDEBAR_EXPANDED_OFFSET_CLASS;

  return (
    <main className="min-h-screen text-foreground">
      <div className={`grid min-h-screen grid-cols-1 transition-[grid-template-columns] duration-300 ${sidebarWidthClass}`}>
        <ChatSidebar
          sessions={sessionHistory}
          activeSessionId={session?.id ?? null}
          onSelectSession={onSelectSession}
          onRenameSession={onRenameSession}
          onDeleteSession={onDeleteSession}
          onNewSession={onNewSession}
          onToggleCollapse={() => setIsSidebarCollapsed((current) => !current)}
          isCollapsed={isSidebarCollapsed}
          newSessionDisabled={isSessionActionDisabled}
          account={sidebarAccount}
        />

        <section className={`grid min-h-screen min-w-0 transition-[grid-template-columns] duration-300 ${isFlashcardsOpen ? "xl:grid-cols-[minmax(0,1fr)_380px]" : "xl:grid-cols-[minmax(0,1fr)_0px]"}`}>
          <div className="flex min-h-screen min-w-0 flex-col bg-background">
            <header className="sticky top-0 z-20 flex min-h-16 items-center justify-between gap-4 border-b border-border bg-background px-5 md:px-7">
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
                  {sessionIdLabel}
                </span>
                <button
                  type="button"
                  onClick={() => setIsFlashcardsOpen((current) => !current)}
                  className="hidden items-center gap-2 rounded-full border border-accent/45 bg-accent/12 px-4 py-2 text-xs font-semibold text-accent transition hover:bg-accent/20 xl:inline-flex"
                  aria-pressed={isFlashcardsOpen}
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="size-4"
                  >
                    <path d="M4.75 5.75A2.75 2.75 0 0 1 7.5 3h11.75v15.25H7.5a2.75 2.75 0 0 0-2.75 2.75z" />
                    <path d="M7.5 3A2.75 2.75 0 0 0 4.75 5.75v15.5" />
                    <path d="M8.75 7.5h7" />
                    <path d="M8.75 10.5h5.5" />
                  </svg>
                  Flashcards
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
                <div className="space-y-5 xl:sticky xl:top-[5rem]">
                  <DocumentUploader disabled={!session || isUploading || isGeneratingFlashcards} onUpload={onUpload} />
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

              <div className={`flex min-h-0 min-w-0 flex-col pb-40 transition-[padding] duration-300 ${isFlashcardsOpen ? "xl:pb-44" : "xl:pb-52"}`}>
                <div className="min-h-0 flex-1 overflow-y-auto aio-scrollbar">
                  <MessageList messages={messages} />
                </div>
              </div>
            </div>
          </div>

          <ChatFlashcardsPanel
            isOpen={isFlashcardsOpen}
            cards={flashcards}
            isLoading={isGeneratingFlashcards}
            error={flashcardsError}
            canGenerate={Boolean(session) && hasReadyDocument && !isGeneratingFlashcards}
            sourceCount={flashcardSourceCount}
            onClose={() => setIsFlashcardsOpen(false)}
            onGenerate={onGenerateFlashcards}
          />
        </section>

        <ChatComposer
          disabled={!session}
          sendDisabled={!hasReadyDocument || isStreaming}
          isFlashcardsOpen={isFlashcardsOpen}
          sidebarOffsetClass={composerSidebarOffsetClass}
          onSend={onSend}
        />
      </div>
    </main>
  );
}
