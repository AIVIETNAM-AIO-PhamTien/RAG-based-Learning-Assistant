import { ArrowLeft, ArrowRight, RefreshCw, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import type { Flashcard } from "@/lib/types";

type ChatFlashcardsPanelProps = {
  isOpen: boolean;
  cards: Flashcard[];
  isLoading: boolean;
  error: string | null;
  canGenerate: boolean;
  sourceCount: number;
  onClose: () => void;
  onGenerate: () => void;
};

export function ChatFlashcardsPanel({
  isOpen,
  cards,
  isLoading,
  error,
  canGenerate,
  sourceCount,
  onClose,
  onGenerate,
}: ChatFlashcardsPanelProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const hasCards = cards.length > 0;
  const currentCard = hasCards ? cards[currentIndex] : null;
  const currentProgress = hasCards ? `${currentIndex + 1}/${cards.length}` : "0/0";
  const isFirstCard = currentIndex === 0;
  const isLastCard = currentIndex === cards.length - 1;

  useEffect(() => {
    setCurrentIndex(0);
    setIsFlipped(false);
  }, [cards]);

  useEffect(() => {
    setIsFlipped(false);
  }, [currentIndex]);

  useEffect(() => {
    if (!isOpen) {
      setIsFlipped(false);
    }
  }, [isOpen]);

  function showPreviousCard() {
    setCurrentIndex((current) => Math.max(current - 1, 0));
  }

  function showNextCard() {
    setCurrentIndex((current) => Math.min(current + 1, cards.length - 1));
  }

  return (
    <aside
      className={`sticky top-0 hidden h-screen overflow-hidden border-l border-border bg-background transition-opacity duration-300 xl:block ${
        isOpen ? "opacity-100" : "pointer-events-none opacity-0"
      }`}
    >
      <div className="flex h-screen w-[380px] flex-col p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold tracking-[-0.03em]">Flashcards</h2>
              <span className="rounded-full border border-border/70 bg-muted/45 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                {cards.length} cards
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Scoped to this session{sourceCount > 0 ? ` · ${sourceCount} source chunks` : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex size-9 items-center justify-center rounded-full border border-border/70 bg-muted/30 text-muted-foreground transition hover:border-accent/60 hover:bg-accent/10 hover:text-accent"
            aria-label="Close flashcards panel"
          >
            ×
          </button>
        </div>

        {error ? (
          <div
            role="alert"
            className="mt-5 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
          >
            {error}
          </div>
        ) : null}

        {hasCards && currentCard ? (
          <div className="relative mt-5 flex min-h-0 flex-1 flex-col gap-4 overflow-hidden rounded-[2rem] border border-border/80 bg-card p-4 shadow-[0_20px_50px_rgba(0,0,0,0.22)]">
            <div className="pointer-events-none absolute inset-x-5 top-0 h-28 rounded-full bg-accent/14 blur-3xl" />
            <div className="pointer-events-none absolute -right-12 top-20 h-40 w-40 rounded-full bg-secondary-accent/12 blur-3xl" />

            <div className="relative rounded-[1.5rem] border border-border/60 bg-background/35 p-2">
              <button
                type="button"
                onClick={onGenerate}
                disabled={!canGenerate}
                className="inline-flex w-full items-center justify-center gap-2 rounded-full border border-accent/30 bg-accent/90 px-4 py-2.5 text-sm font-semibold text-accent-foreground shadow-[0_10px_24px_rgba(132,90,255,0.24)] transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-45"
              >
                <RefreshCw className={`size-4 ${isLoading ? "animate-spin" : ""}`} aria-hidden="true" />
                {isLoading ? "Generating..." : "Regenerate"}
              </button>
            </div>

            <div className="relative flex min-h-0 flex-1 flex-col gap-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 rounded-full border border-border/60 bg-background/35 px-3 py-1.5">
                  <span className="size-2 rounded-full bg-accent shadow-[0_0_12px_rgba(132,90,255,0.65)]" aria-hidden="true" />
                  <p className="text-[11px] font-semibold tracking-[0.18em] text-muted-foreground uppercase">
                    {isFlipped ? "Answer side" : "Question side"}
                  </p>
                </div>
                <span className="rounded-full border border-border/60 bg-background/40 px-3 py-1 font-mono text-[11px] text-muted-foreground">
                  {currentProgress}
                </span>
              </div>

              <div className="flex min-h-0 flex-1 items-center justify-center" style={{ perspective: "1400px" }}>
                <button
                  type="button"
                  onClick={() => setIsFlipped((current) => !current)}
                  className="group w-full cursor-pointer rounded-[2rem] text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  aria-label={isFlipped ? "Show question side of flashcard" : "Show answer side of flashcard"}
                  aria-pressed={isFlipped}
                >
                  <div className="rounded-[2rem] bg-linear-to-br from-accent/20 via-accent/8 to-secondary-accent/16 p-[1px] shadow-[0_24px_55px_rgba(8,10,17,0.48)] transition duration-300 group-hover:shadow-[0_28px_65px_rgba(87,63,160,0.32)]">
                    <div
                      className="relative h-[380px] w-full motion-reduce:transition-none"
                      style={{
                        transform: isFlipped ? "rotateY(180deg)" : "rotateY(0deg)",
                        transformStyle: "preserve-3d",
                        transition: "transform 520ms cubic-bezier(0.22, 1, 0.36, 1)",
                      }}
                    >
                      <div
                        className="absolute inset-0 flex h-full flex-col overflow-hidden rounded-[calc(2rem-1px)] border border-white/6 bg-[radial-gradient(circle_at_top_left,rgba(132,90,255,0.28),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(86,71,160,0.18),transparent_26%),linear-gradient(180deg,rgba(27,31,49,0.98),rgba(13,16,27,0.98))] p-5"
                        style={{ backfaceVisibility: "hidden" }}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="rounded-full border border-accent/25 bg-accent/14 px-3 py-1 text-[11px] font-semibold tracking-[0.18em] text-accent uppercase">
                            Question
                          </span>
                          <span className="rounded-full border border-white/8 bg-white/6 px-3 py-1 text-[11px] text-muted-foreground">
                            Tap to reveal
                          </span>
                        </div>
                        <div className="pointer-events-none absolute inset-x-10 top-14 h-20 rounded-full bg-accent/18 blur-3xl" />
                        <div className="relative mt-5 flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-2">
                          <div className="max-w-[18rem] text-center">
                            <h3 className="text-xl font-semibold leading-9 text-foreground">
                              {currentCard.question}
                            </h3>
                          </div>
                        </div>
                        <div className="relative flex items-center justify-between gap-3 rounded-[1.25rem] border border-white/6 bg-background/25 px-4 py-3 text-xs text-muted-foreground">
                          <span>Pause, recall, then flip.</span>
                          <span className="font-mono text-[11px] text-accent/90">{currentProgress}</span>
                        </div>
                      </div>

                      <div
                        className="absolute inset-0 flex h-full flex-col overflow-hidden rounded-[calc(2rem-1px)] border border-accent/16 bg-[radial-gradient(circle_at_top_right,rgba(132,90,255,0.28),transparent_34%),radial-gradient(circle_at_bottom_left,rgba(94,114,228,0.18),transparent_28%),linear-gradient(180deg,rgba(20,24,42,0.98),rgba(10,13,23,0.98))] p-5"
                        style={{
                          backfaceVisibility: "hidden",
                          transform: "rotateY(180deg)",
                        }}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="rounded-full border border-secondary-accent/30 bg-secondary-accent/20 px-3 py-1 text-[11px] font-semibold tracking-[0.18em] text-foreground uppercase">
                            Answer
                          </span>
                          <span className="rounded-full border border-white/8 bg-white/6 px-3 py-1 text-[11px] text-muted-foreground">
                            Tap to flip back
                          </span>
                        </div>
                        <div className="pointer-events-none absolute left-8 top-12 h-24 w-24 rounded-full bg-secondary-accent/18 blur-3xl" />
                        <div className="relative mt-5 min-h-0 flex-1 overflow-y-auto rounded-[1.5rem] border border-white/6 bg-background/20 px-4 py-5">
                          <p className="text-[11px] font-semibold tracking-[0.28em] text-muted-foreground uppercase">
                            Explanation
                          </p>
                          <p className="mt-4 text-sm leading-7 text-foreground">
                            {currentCard.answer}
                          </p>
                        </div>
                        <div className="relative mt-4 flex items-center justify-between gap-3 rounded-[1.25rem] border border-white/6 bg-background/25 px-4 py-3 text-xs text-muted-foreground">
                          <span>Use the arrows to continue reviewing.</span>
                          <span className="font-mono text-[11px] text-secondary-accent/90">{currentProgress}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </button>
              </div>

              <div className="relative flex items-center justify-between gap-3 rounded-[1.5rem] border border-border/60 bg-background/35 p-2">
                <button
                  type="button"
                  onClick={showPreviousCard}
                  disabled={isFirstCard}
                  className="inline-flex min-w-[7.25rem] items-center justify-center gap-2 rounded-full border border-border/70 bg-background/55 px-4 py-2.5 text-sm font-medium text-foreground transition hover:border-accent/50 hover:bg-accent/10 hover:text-accent disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <ArrowLeft className="size-4" aria-hidden="true" />
                  Previous
                </button>
                <p className="rounded-full border border-accent/15 bg-accent/10 px-3 py-1.5 font-mono text-sm text-accent-foreground/90">
                  {currentProgress}
                </p>
                <button
                  type="button"
                  onClick={showNextCard}
                  disabled={isLastCard}
                  className="inline-flex min-w-[7.25rem] items-center justify-center gap-2 rounded-full border border-border/70 bg-background/55 px-4 py-2.5 text-sm font-medium text-foreground transition hover:border-accent/50 hover:bg-accent/10 hover:text-accent disabled:cursor-not-allowed disabled:opacity-45"
                >
                  Next
                  <ArrowRight className="size-4" aria-hidden="true" />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-5 flex min-h-0 flex-1 items-center justify-center rounded-[2rem] border border-dashed border-border bg-card p-6 text-center">
            <div>
              <div className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-accent/20 bg-accent/10 text-accent shadow-[0_14px_32px_rgba(87,63,160,0.2)]">
                <Sparkles className="size-6" aria-hidden="true" />
              </div>
              <h3 className="mt-5 text-base font-semibold">No flashcards yet for this session</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Generate study cards from the ready documents in this session.
              </p>
              <button
                type="button"
                onClick={onGenerate}
                disabled={!canGenerate}
                className="mt-5 inline-flex items-center gap-2 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground shadow-[0_12px_28px_rgba(132,90,255,0.24)] transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-45"
              >
                <Sparkles className="size-4" aria-hidden="true" />
                {isLoading ? "Generating..." : "Generate flashcards"}
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
