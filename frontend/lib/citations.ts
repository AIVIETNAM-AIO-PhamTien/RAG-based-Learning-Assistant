export type CitationTextPart = { type: "text"; text: string };
export type CitationIndexPart = { type: "citation"; index: number };
export type CitationPart = CitationTextPart | CitationIndexPart;

const CITATION_PATTERN = /\[(\d+)]/g;

export function parseCitations(text: string): CitationPart[] {
  const parts: CitationPart[] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(CITATION_PATTERN)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      parts.push({ type: "text", text: text.slice(lastIndex, index) });
    }
    parts.push({ type: "citation", index: Number(match[1]) });
    lastIndex = index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: "text", text: text.slice(lastIndex) });
  }
  return parts;
}
