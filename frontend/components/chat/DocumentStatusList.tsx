import type { DocumentRead } from "@/lib/types";

const statusStyles: Record<DocumentRead["status"], string> = {
  pending: "border-border/70 bg-muted/50 text-muted-foreground",
  processing: "border-accent/30 bg-accent/10 text-accent",
  ready: "border-emerald-300/30 bg-emerald-300/10 text-emerald-200",
  failed: "border-danger/30 bg-danger/10 text-danger",
};

export function DocumentStatusList({ documents }: { documents: DocumentRead[] }) {
  if (documents.length === 0) {
    return (
      <p className="rounded-2xl border border-border/60 bg-muted/30 p-4 text-xs leading-5 text-muted-foreground">
        No documents attached yet. Uploaded PDFs will appear here as session files.
      </p>
    );
  }

  return (
    <div className="space-y-2.5">
      {documents.map((document) => (
        <div key={document.id} className="rounded-2xl border border-border/70 bg-card/80 p-3.5 shadow-[0_12px_36px_rgba(0,0,0,0.2)]">
          <div className="flex items-start gap-3">
            <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent/12 text-sm text-accent">
              PDF
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-3">
                <p className="truncate text-sm font-medium text-foreground">{document.name}</p>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase ${statusStyles[document.status]}`}>
                  {document.status}
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {document.page_count ? `${document.page_count} pages indexed` : "Page count pending"}
              </p>
              {document.error_message ? (
                <p className="mt-2 text-xs leading-5 text-danger">{document.error_message}</p>
              ) : null}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
