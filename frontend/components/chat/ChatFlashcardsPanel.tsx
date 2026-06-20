type ChatFlashcardsPanelProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function ChatFlashcardsPanel({ isOpen, onClose }: ChatFlashcardsPanelProps) {
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
              <span className="rounded-full border border-border/70 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                0 cards
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">Scoped to this session</p>
          </div>
          <button
            type="button"
            onClick={onClose}
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

        <div className="mt-5 flex min-h-0 flex-1 items-center justify-center rounded-[2rem] border border-dashed border-border bg-card p-6 text-center">
          <div>
            <div className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-border/70 bg-muted text-2xl text-accent">
              ✨
            </div>
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
      </div>
    </aside>
  );
}
