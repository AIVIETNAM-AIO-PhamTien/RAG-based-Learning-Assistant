import type { ChatSession } from "@/lib/types";

import { ChatAccountCard } from "./ChatAccountCard";
import { ChatHistoryCard } from "./ChatHistoryCard";

type ChatSidebarProps = {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onRenameSession: (sessionId: string, title: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onNewSession: () => void;
  onToggleCollapse: () => void;
  isCollapsed: boolean;
  newSessionDisabled: boolean;
  account: {
    initials: string;
    label: string;
    subtitle: string;
  };
};

export function ChatSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  onNewSession,
  onToggleCollapse,
  isCollapsed,
  newSessionDisabled,
  account,
}: ChatSidebarProps) {
  return (
    <aside
      className={`sticky top-0 hidden h-screen self-start overflow-hidden border-r border-border bg-background p-3 transition-[padding] duration-300 lg:flex lg:flex-col ${
        isCollapsed ? "items-center px-2" : "p-4"
      }`}
    >
      <div
        className={`w-full rounded-3xl border border-border bg-card transition-[padding,gap] duration-300 ${
          isCollapsed ? "flex flex-col items-center gap-3 p-2.5" : "p-3"
        }`}
      >
        {isCollapsed ? (
          <>
            <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-muted text-xl">
              🎓
            </div>
            <button
              type="button"
              onClick={onToggleCollapse}
              className="flex size-9 items-center justify-center rounded-2xl border border-border/70 bg-background/70 text-muted-foreground transition hover:border-accent hover:bg-accent/10 hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              aria-label="Expand sidebar"
              title="Expand sidebar"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="size-4 shrink-0"
              >
                <path d="m9 6 6 6-6 6" />
              </svg>
            </button>
          </>
        ) : (
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-muted text-xl">
                🎓
              </div>
              <div className="min-w-0">
                <p className="text-base font-semibold tracking-[-0.03em]">AIO</p>
                <p className="text-xs text-muted-foreground">AI Tutor</p>
              </div>
            </div>
            <button
              type="button"
              onClick={onToggleCollapse}
              className="flex size-9 shrink-0 items-center justify-center rounded-2xl border border-border/70 text-muted-foreground transition hover:border-accent hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              aria-label="Collapse sidebar"
              title="Collapse sidebar"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="size-4 shrink-0"
              >
                <path d="m15 6-6 6 6 6" />
              </svg>
            </button>
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={onNewSession}
        disabled={newSessionDisabled}
        className={`mt-3 flex items-center justify-center rounded-2xl bg-muted text-foreground transition hover:bg-accent/10 hover:text-accent disabled:cursor-not-allowed disabled:opacity-65 ${
          isCollapsed ? "size-11" : "w-full gap-2 px-4 py-3 text-sm font-semibold"
        }`}
        aria-label="New Session"
        title="New Session"
      >
        <span>＋</span>
        {!isCollapsed ? <span>New Session</span> : null}
      </button>

      <section className={`mt-6 flex min-h-0 flex-1 flex-col ${isCollapsed ? "w-full items-center" : ""}`}>
        {!isCollapsed ? (
          <>
            <p className="px-1 font-mono text-[11px] tracking-[0.2em] text-muted-foreground uppercase">History</p>
            <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto rounded-2xl border border-border/70 bg-muted/35 p-3">
              {sessions.map((chatSession) => (
                <ChatHistoryCard
                  key={chatSession.id}
                  sessionTitle={chatSession.title ?? "Current study session"}
                  sessionIdLabel={chatSession.id.slice(0, 8)}
                  isActive={chatSession.id === activeSessionId}
                  onClick={() => onSelectSession(chatSession.id)}
                  onRename={(title) => onRenameSession(chatSession.id, title)}
                  onDelete={() => onDeleteSession(chatSession.id)}
                  disabled={newSessionDisabled}
                />
              ))}
              <p className="px-1 pt-1 text-xs leading-5 text-muted-foreground">
                Saved sessions stay local to this browser in this pass.
              </p>
            </div>
          </>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col items-center gap-2 overflow-y-auto rounded-3xl border border-border/70 bg-muted/35 px-2 py-3">
            {sessions.map((chatSession) => {
              const isActive = chatSession.id === activeSessionId;
              return (
                <button
                  key={chatSession.id}
                  type="button"
                  onClick={() => onSelectSession(chatSession.id)}
                  disabled={newSessionDisabled}
                  className={`inline-flex size-10 items-center justify-center rounded-2xl border text-[11px] font-mono transition disabled:cursor-not-allowed disabled:opacity-65 ${
                    isActive
                      ? "border-accent/60 bg-accent/15 text-accent hover:bg-accent/20"
                      : "border-border/70 bg-background text-muted-foreground hover:border-accent/40 hover:bg-accent/10 hover:text-foreground"
                  }`}
                  aria-label={chatSession.title ?? "Current study session"}
                  title={chatSession.title ?? chatSession.id.slice(0, 8)}
                >
                  {chatSession.id.slice(0, 2)}
                </button>
              );
            })}
          </div>
        )}
      </section>
      <div className={`mt-5 ${isCollapsed ? "flex justify-center" : ""}`}>
        {isCollapsed ? (
          <button
            type="button"
            className="flex size-11 items-center justify-center rounded-2xl border border-border/70 bg-card text-sm font-semibold text-foreground transition hover:border-accent/40 hover:bg-accent/10 hover:text-accent"
            aria-label={account.label}
            title={`${account.label} · ${account.subtitle}`}
          >
            {account.initials}
          </button>
        ) : (
          <ChatAccountCard initials={account.initials} label={account.label} subtitle={account.subtitle} />
        )}
      </div>
    </aside>
  );
}
