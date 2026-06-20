"use client";

import { useState } from "react";

type ChatHistoryCardProps = {
  sessionTitle: string;
  sessionIdLabel: string;
  isActive: boolean;
  onClick: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
  disabled?: boolean;
};

export function ChatHistoryCard({
  sessionTitle,
  sessionIdLabel,
  isActive,
  onClick,
  onRename,
  onDelete,
  disabled,
}: ChatHistoryCardProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  function openRenamePrompt() {
    setIsMenuOpen(false);
    const nextTitle = window.prompt("Rename session", sessionTitle)?.trim();
    if (!nextTitle || nextTitle === sessionTitle) return;
    onRename(nextTitle);
  }

  function confirmDelete() {
    setIsMenuOpen(false);
    if (!window.confirm(`Delete session \"${sessionTitle}\"?`)) return;
    onDelete();
  }

  return (
    <div
      className={`relative rounded-xl transition ${
        isActive ? "bg-accent/12" : "bg-background/40 hover:bg-background/70"
      } ${disabled ? "opacity-60" : ""}`}
    >
      <div className="flex items-start gap-2 p-3">
        <button
          type="button"
          onClick={onClick}
          disabled={disabled}
          className="min-w-0 flex-1 rounded-lg text-left transition hover:text-accent disabled:cursor-not-allowed"
        >
          <div className="flex items-center justify-between gap-3">
            <p className="truncate text-sm font-medium text-foreground">{sessionTitle}</p>
            <span className={`text-xs ${isActive ? "text-emerald-400" : "text-muted-foreground"}`}>●</span>
          </div>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">{sessionIdLabel}</p>
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation();
            setIsMenuOpen((current) => !current);
          }}
          className="inline-flex size-8 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-background/70 text-sm text-muted-foreground transition hover:border-accent/40 hover:bg-accent/10 hover:text-foreground disabled:cursor-not-allowed"
          aria-label={`Open actions for ${sessionTitle}`}
        >
          ⋯
        </button>
      </div>
      {isMenuOpen ? (
        <div className="absolute top-12 right-3 z-10 min-w-32 rounded-xl border border-border/70 bg-card p-1 shadow-[0_12px_24px_rgba(0,0,0,0.2)]">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              openRenamePrompt();
            }}
            className="flex w-full items-center rounded-lg px-3 py-2 text-left text-sm text-foreground transition hover:bg-muted"
          >
            Rename
          </button>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              confirmDelete();
            }}
            className="flex w-full items-center rounded-lg px-3 py-2 text-left text-sm text-danger transition hover:bg-danger/10"
          >
            Delete
          </button>
        </div>
      ) : null}
    </div>
  );
}
