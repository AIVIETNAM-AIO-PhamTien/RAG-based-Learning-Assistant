type ChatAccountCardProps = {
  initials: string;
  label: string;
  subtitle: string;
};

export function ChatAccountCard({ initials, label, subtitle }: ChatAccountCardProps) {
  return (
    <div className="mt-5 rounded-2xl border border-border bg-card p-3 transition hover:border-accent/40 hover:bg-accent/10">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex size-9 items-center justify-center rounded-full bg-muted text-xs text-muted-foreground">
          {initials}
        </div>
        <div className="min-w-0">
          <p className="truncate text-xs font-medium text-foreground">{label}</p>
          <p className="text-[11px] text-muted-foreground">{subtitle}</p>
        </div>
      </div>
    </div>
  );
}
