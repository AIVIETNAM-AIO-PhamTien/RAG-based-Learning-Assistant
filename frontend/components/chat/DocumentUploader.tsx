"use client";

import type { ChangeEvent, DragEvent } from "react";

export function DocumentUploader({ disabled, onUpload }: { disabled?: boolean; onUpload: (file: File) => void }) {
  function uploadFile(file?: File) {
    if (!file || disabled) return;
    onUpload(file);
  }

  function onChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    uploadFile(file);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    uploadFile(event.dataTransfer.files?.[0]);
  }

  function onDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
  }

  return (
    <label
      onDrop={onDrop}
      onDragOver={onDragOver}
      className="group block cursor-pointer rounded-3xl border border-dashed border-border bg-card p-5 text-center shadow-[0_8px_24px_rgba(0,0,0,0.16)] transition-colors hover:border-accent/45 hover:bg-muted/65 has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-60"
    >
      <span className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-border bg-muted text-2xl text-accent/85">
        ⇧
      </span>
      <span className="mt-4 block text-sm font-semibold text-foreground">Tap to select a PDF file</span>
      <span className="mt-2 block text-xs leading-5 text-muted-foreground">
        Drag and drop lecture slides, notes, or textbook PDFs here.
      </span>
      <span className="mt-3 block font-mono text-[11px] text-muted-foreground">
        Max 10MB · First 30 pages will be read
      </span>
      <input
        className="sr-only"
        type="file"
        accept="application/pdf,.pdf"
        disabled={disabled}
        onChange={onChange}
      />
    </label>
  );
}
