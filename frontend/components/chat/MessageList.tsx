import { CitationPill } from "./CitationPill";

import { parseCitations } from "@/lib/citations";
import type { ChatMessage } from "@/lib/types";

function AioMark() {
  return (
    <span className="flex size-9 shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-muted text-sm text-accent">
      ✦
    </span>
  );
}

function AssistantText({ message }: { message: ChatMessage }) {
  const citationsByIndex = new Map((message.citations ?? []).map((citation) => [citation.index, citation]));
  return (
    <p className="whitespace-pre-wrap text-[15px] leading-7 text-foreground/90">
      {parseCitations(message.content).map((part, index) =>
        part.type === "text" ? (
          <span key={index}>{part.text}</span>
        ) : (
          <CitationPill key={index} citation={citationsByIndex.get(part.index)} />
        ),
      )}
      {message.isStreaming ? <span className="ml-1 text-accent">▋</span> : null}
    </p>
  );
}

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  if (messages.length === 0) {
    return (
      <div className="flex min-h-full items-center justify-center px-6 py-12 text-center">
        <div className="mx-auto max-w-xl">
          <div className="mx-auto flex size-20 items-center justify-center rounded-[2rem] border border-border/70 bg-muted text-4xl text-accent">
            ⇧
          </div>
          <h2 className="mt-7 text-3xl font-semibold tracking-[-0.04em] text-foreground">
            Start by uploading a document
          </h2>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            AIO reads PDFs like lecture slides, textbooks, and notes so every answer can stay grounded in your study material.
          </p>
          <div className="mt-7 rounded-3xl border border-border/70 bg-card/65 p-5 text-left shadow-[0_24px_80px_rgba(4,7,24,0.26)]">
            <p className="text-sm font-medium text-foreground">Or type your question directly below</p>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">
              Ask AIO to summarize concepts, explain confusing sections, or generate study prompts once a PDF is ready.
            </p>
            <p className="mt-4 text-xs text-muted-foreground">✦ Generate flashcards from your document to review later</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-4xl space-y-8 px-6 py-8">
      {messages.map((message) =>
        message.role === "assistant" ? (
          <article key={message.id} className="flex items-start gap-4">
            <AioMark />
            <div className="min-w-0 flex-1 pt-1">
              <AssistantText message={message} />
            </div>
          </article>
        ) : (
          <article key={message.id} className="flex justify-end">
            <p className="max-w-[75%] whitespace-pre-wrap rounded-[1.6rem] rounded-br-md bg-accent px-5 py-3 text-sm leading-6 text-white">
              {message.content}
            </p>
          </article>
        ),
      )}
    </div>
  );
}
