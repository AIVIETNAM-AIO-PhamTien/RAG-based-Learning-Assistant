Bạn có thể demo bằng:


docker compose up -d

`cd backend`

`uv run alembic upgrade head`

``uv run uvicorn app.main:app --reload --port 8000``
---
``uv --directory backend run uvicorn app.main:app --reload --port 8000``

`pnpm --dir frontend dev`
---
``cd frontend``

``pnpm dev``

Rồi mở:


http://localhost:3000

Flow demo:

- Upload PDF
- Đợi ingest xong
- Hỏi câu về PDF
- Nhận answer streaming
- Hover [1] để xem đoạn nguồn + page
---

# AIO — Educational RAG Chatbot

A NotebookLM-style RAG chatbot for studying PDFs. Upload tài liệu (VN/EN), hỏi đáp có **citation hover-able** chỉ tới đúng đoạn, đúng trang, đúng document.

> Project mục tiêu **học RAG end-to-end**. Stack chạy **miễn phí** từ dev → deploy.

---

## TL;DR stack

**Backend** FastAPI · SQLAlchemy 2.0 async · Pydantic v2 · arq + Redis
**RAG** PyMuPDF · intfloat/e5-small-v2 · BAAI/bge-reranker-v2-m3 · hybrid (BM25 + dense) + RRF + MMR
**LLM** Gemini 2.5 Flash (primary) · Cerebras Qwen3-32B (fallback) · Ollama Qwen2.5-7B (offline dev)
**DB** Postgres + pgvector (Neon free)
**Frontend** Next.js 15 · Tailwind v4 · shadcn/ui · Framer Motion · `@react-pdf-viewer` · Vercel AI SDK 5
**Storage** Cloudflare R2 (10 GB free, $0 egress)
**Observability** Langfuse Cloud Hobby
**Deploy** Vercel (FE) · Render (BE) · Neon (DB) · Upstash (Redis) · R2 (PDF)

Cost: **$0/tháng**.

---

## Bạn đang ở đâu trong docs?

| Tình huống | Đọc trước |
|---|---|
| Mới vào, chưa biết RAG là gì | `local/glossary.md` → `local/architect.md` |
| Hiểu rồi, muốn biết hệ thống vận hành | `local/architect.md` |
| Bắt tay code BE | `local/plan.md` rồi follow `local/agent-prompt.md` |
| **Bắt tay code FE (Cursor/Copilot/Windsurf/Codex/manual)** | **`local/frontend-taste.md`** |
| Tune prompt cho LLM | `local/prompts.md` |
| Stuck với jargon (chunk, reranker, RRF, MRR…) | `local/glossary.md` |

> `local/` được **gitignore** — đó là không gian cho notes/agent docs. README này là cái duy nhất public.

---

## Quickstart (sau khi Phase 0 scaffold xong)

```bash
# 1. env examples
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# 2. local services
docker compose up -d        # postgres + redis

# 3. backend
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# 4. frontend
cd ../frontend
pnpm install
pnpm dev                   # http://localhost:3000
```

Sample PDF để thử nằm trong `sample/`. Upload qua UI hoặc:

```bash
curl -F file=@sample/linear-algebra-ch3.pdf http://localhost:8000/api/v1/documents
```

---

## Phases

- **Phase 1 (current):** PDF upload + RAG Q&A + NotebookLM citation (VN+EN, text-only).
- **Phase 2:** Multimodal slides (Docling/ColPali), HyDE, CRAG-lite "I don't know", flashcard generator.
- **Phase 3:** Quiz, RAPTOR tree summaries, multi-user auth, optional GraphRAG.

Chi tiết: `local/plan.md`.

---

## Citation behavior (đặc tả ngắn)

```
User hỏi → BE retrieve → LLM trả lời với [1], [2], [3] inline.
FE parse [n] → render <CitationPill>.
  hover  → tooltip: doc_name + page + snippet.
  click  → /viewer?doc=<id>&page=<n>#bbox=... → PDF viewer jump + highlight.
```

Mỗi citation BẮT BUỘC được **validate** trước khi return (chống hallucinated `[n]`).

---

## License

MIT (sẽ thêm khi public).
