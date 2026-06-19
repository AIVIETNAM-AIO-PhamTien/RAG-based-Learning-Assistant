import type { Citation } from "@/lib/types";

export function CitationPill({ citation }: { citation?: Citation }) {
  return (
    <span className="group relative inline-flex align-baseline">
      <span className="mx-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-border/70 bg-muted px-1.5 font-mono text-[11px] leading-none text-accent transition group-hover:border-accent/60">
        [{citation?.index ?? "?"}]
      </span>
      {citation ? (
        <span className="pointer-events-none absolute bottom-7 left-1/2 z-20 hidden w-80 -translate-x-1/2 rounded-2xl border border-border bg-[#14162b] p-4 text-left shadow-[0_24px_80px_rgba(0,0,0,0.45)] group-hover:block">
          <span className="block truncate text-xs font-semibold text-foreground">{citation.doc_name}</span>
          <span className="mt-1 block font-mono text-[11px] text-muted-foreground">page {citation.page}</span>
          <span className="mt-3 line-clamp-6 block text-xs leading-5 text-muted-foreground">
            {citation.snippet || citation.text}
          </span>
        </span>
      ) : null}
    </span>
  );
}
