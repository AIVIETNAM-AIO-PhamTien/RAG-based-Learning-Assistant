"use client";

import { useEffect, useState } from "react";

import type { Flashcard, FlashcardStats, FlashcardStatus } from "@/lib/types";

type Props = {
  cards: Flashcard[];
  stats: FlashcardStats;
  disabled: boolean;
  generating: boolean;
  onClose: () => void;
  onGenerate: (topic: string, count: 5 | 10 | 15) => Promise<void>;
  onStatusChange: (id: string, status: FlashcardStatus) => Promise<void>;
};

const emptyStats: FlashcardStats = { total: 0, not_reviewed: 0, learning: 0, known: 0 };

export function FlashcardsPanel({
  cards,
  stats = emptyStats,
  disabled,
  generating,
  onClose,
  onGenerate,
  onStatusChange,
}: Props) {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState<5 | 10 | 15>(10);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [showReplaceWarning, setShowReplaceWarning] = useState(false);
  const card = cards[index];
  const knownPercent = stats.total ? Math.round((stats.known / stats.total) * 100) : 0;

  useEffect(() => {
    if (index >= cards.length) setIndex(Math.max(0, cards.length - 1));
    setFlipped(false);
  }, [cards.length, index]);

  async function submit() {
    if (!topic.trim()) return;
    if (cards.length && !showReplaceWarning) {
      setShowReplaceWarning(true);
      return;
    }
    await onGenerate(topic.trim(), count);
    setShowReplaceWarning(false);
    setIndex(0);
  }

  return (
    <div className="flex h-full w-[380px] flex-col p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold tracking-[-0.03em]">Flashcards</h2>
            <span className="rounded-full border border-border/70 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
              {stats.total} cards
            </span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Scoped to this session</p>
        </div>
        <button type="button" onClick={onClose} className="flex size-9 items-center justify-center rounded-full border border-border/70 text-muted-foreground" aria-label="Close flashcards panel">×</button>
      </div>

      <div className="mt-5 rounded-2xl border border-border bg-card p-4">
        <div className="flex justify-between text-[11px] text-muted-foreground"><span>Known {stats.known}</span><span>Learning {stats.learning}</span><span>New {stats.not_reviewed}</span></div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-accent" style={{ width: `${knownPercent}%` }} /></div>
      </div>

      <div className="mt-4 space-y-2 rounded-2xl border border-border bg-card p-3">
        <input value={topic} onChange={(event) => { setTopic(event.target.value); setShowReplaceWarning(false); }} placeholder="Chủ đề cần ôn, ví dụ: định luật Newton" className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none" disabled={disabled || generating} />
        <div className="flex gap-2">
          <select value={count} onChange={(event) => setCount(Number(event.target.value) as 5 | 10 | 15)} className="rounded-xl border border-border bg-background px-2 text-sm" disabled={disabled || generating}><option value={5}>5 thẻ</option><option value={10}>10 thẻ</option><option value={15}>15 thẻ</option></select>
          <button type="button" onClick={() => void submit()} disabled={disabled || generating || !topic.trim()} className="flex-1 rounded-xl bg-accent px-3 py-2 text-sm font-semibold text-accent-foreground disabled:opacity-45">{generating ? "Đang tạo..." : showReplaceWarning ? "Xác nhận thay thế" : "Generate"}</button>
        </div>
        {showReplaceWarning ? <p className="text-xs text-danger">Bộ thẻ hiện tại sẽ bị thay thế. Bấm “Xác nhận thay thế” để tiếp tục.</p> : null}
      </div>

      {card ? (
        <div className="mt-5 flex flex-1 flex-col">
          <button type="button" onClick={() => setFlipped((value) => !value)} className="flex min-h-56 flex-1 flex-col justify-center rounded-[2rem] border border-border bg-card p-6 text-left transition hover:border-accent">
            <span className="text-xs text-muted-foreground">{flipped ? "Đáp án" : "Câu hỏi"} · chạm để lật</span>
            <p className="mt-4 text-lg font-medium leading-7">{flipped ? card.answer : card.question}</p>
            {flipped ? <p className="mt-5 text-xs text-muted-foreground">Nguồn: {card.source_doc_name} · trang {card.source_page}</p> : null}
          </button>
          <div className="mt-3 flex gap-2">
            {(["not_reviewed", "learning", "known"] as FlashcardStatus[]).map((status) => <button key={status} type="button" disabled={!flipped} onClick={() => void onStatusChange(card.id, status)} className={`flex-1 rounded-xl border px-2 py-2 text-xs disabled:opacity-45 ${card.status === status ? "border-accent bg-accent/10 text-accent" : "border-border"}`}>{status === "not_reviewed" ? "New" : status === "learning" ? "Learning" : "Known"}</button>)}
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground"><button type="button" disabled={index === 0} onClick={() => setIndex((value) => value - 1)}>← Previous</button><span>{index + 1} / {cards.length}</span><button type="button" disabled={index === cards.length - 1} onClick={() => setIndex((value) => value + 1)}>Next →</button></div>
        </div>
      ) : <div className="mt-5 flex flex-1 items-center justify-center rounded-[2rem] border border-dashed border-border bg-card p-6 text-center text-sm text-muted-foreground">{disabled ? "Upload and finish ingesting a PDF before generating flashcards." : "Enter a topic to generate flashcards for this session."}</div>}
    </div>
  );
}
