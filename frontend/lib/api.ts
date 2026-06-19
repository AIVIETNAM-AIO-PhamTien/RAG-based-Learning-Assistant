import type { ChatSession, ChatStreamEvent, DocumentRead } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function createSession(): Promise<ChatSession> {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Baseline RAG" }),
  });
  return parseResponse<ChatSession>(response);
}

export async function uploadDocument(sessionId: string, file: File): Promise<DocumentRead> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/documents`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<DocumentRead>(response);
}

export async function getSessionDocuments(sessionId: string): Promise<DocumentRead[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/documents`);
  return parseResponse<DocumentRead[]>(response);
}

export async function streamChat(
  sessionId: string,
  message: string,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const eventLine = frame.split("\n").find((line) => line.startsWith("event: "));
      const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      const type = eventLine.replace("event: ", "") as ChatStreamEvent["type"];
      const data = JSON.parse(dataLine.replace("data: ", "")) as Record<string, unknown>;
      onEvent({ type, ...data } as ChatStreamEvent);
    }
  }
}
