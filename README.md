# AIO

AIO is a study-oriented RAG application for PDF-based learning.

Upload lecture slides, notes, or textbooks, then ask questions against the uploaded material. The app streams grounded answers, attaches inline citations, and lets you generate flashcards from the active study session.

---

## UI Preview

> **Screenshot placeholder**
>
> Add the application UI screenshot here later.
>
> Suggested path: `docs/images/aio-ui.png`
>
> Example markdown to replace this block:
>
> `![AIO UI](docs/images/aio-ui.png)`

---

## What the app does

### Current user-facing features

- **Session-based study workspace**
  - create, rename, switch, and delete chat sessions
  - each session keeps its own uploaded documents and message history
- **PDF upload and ingestion**
  - accepts PDF uploads per session
  - parses text-layer PDFs and stores chunks plus embeddings
- **Grounded chat with streaming responses**
  - answers are streamed token-by-token over SSE
  - responses include inline citation markers like `[1]`, `[2]`
- **Citation hover cards**
  - hovering a citation shows document name, page number, and source snippet
- **Flashcard generation**
  - generates flashcards from ready documents in the current session
  - supports regeneration from the same session context
- **Study summary API**
  - backend exposes a summary endpoint for session study sources
- **Evaluation pipeline**
  - includes retrieval/generation benchmarking utilities under `backend/evaluation/`

### Current constraints

- optimized for **text-layer PDFs**
- **encrypted PDFs are not supported**
- if a PDF has no readable text layer, ingestion fails
- chat and study generation require a valid **Gemini API key**

---

## Architecture at a glance

### Backend

- **Framework:** FastAPI
- **ORM / DB access:** SQLAlchemy 2 async + Alembic
- **Database:** PostgreSQL + pgvector
- **PDF parsing:** PyMuPDF
- **Embeddings:** `intfloat/e5-small-v2`
- **Reranking:** `BAAI/bge-reranker-base`
- **Generation:** Gemini (`gemini-2.5-flash` by default)

### Frontend

- **Framework:** Next.js 15
- **UI:** React 19 + Tailwind CSS 4
- **Interaction model:** session sidebar, upload panel, streaming chat, flashcard panel

### Runtime flow

1. User uploads a PDF into a session.
2. Backend parses PDF text, chunks pages, computes embeddings, and stores chunks in Postgres.
3. User asks a question.
4. Backend embeds the query, retrieves candidate chunks from the session corpus, optionally reranks them, and sends the selected context to Gemini.
5. Gemini streams the answer back.
6. Frontend renders inline citations and shows hoverable source snippets.
7. User can generate flashcards from representative session chunks.

---

## Repository structure

```text
.
├── backend/              # FastAPI app, RAG pipeline, tests, evaluation tools
│   ├── app/
│   │   ├── api/v1/       # health, sessions, documents, chat, study routes
│   │   ├── db/           # SQLAlchemy models and session setup
│   │   ├── rag/          # parser, chunker, embedder, retriever, generator
│   │   └── schemas/      # Pydantic schemas
│   ├── evaluation/       # offline evaluation / benchmarking pipeline
│   └── tests/            # backend tests
├── frontend/             # Next.js application
│   ├── app/              # App Router entrypoints
│   ├── components/chat/  # sidebar, uploader, message list, flashcards panel
│   └── lib/              # API client, types, citation parsing
├── infra/                # local database initialization and deployment notes
├── data/                 # uploaded documents / local data
├── store/                # repository data directory
└── docker-compose.yml    # local Postgres + Redis services
```

---

## API overview

### Health

- `GET /healthz`

### Sessions

- `POST /api/v1/sessions/` — create a session
- `GET /api/v1/sessions/` — list sessions
- `PATCH /api/v1/sessions/{session_id}` — rename a session
- `DELETE /api/v1/sessions/{session_id}` — delete a session
- `GET /api/v1/sessions/{session_id}/documents` — list documents in a session
- `GET /api/v1/sessions/{session_id}/messages` — list message history

### Documents

- `POST /api/v1/sessions/{session_id}/documents` — upload a PDF into a session
- `GET /api/v1/documents/{document_id}` — get document metadata/status

### Chat

- `POST /api/v1/sessions/{session_id}/chat` — stream an answer with SSE events
  - emits `token`, `citations`, `done`, and `error`

### Study tools

- `POST /api/v1/sessions/{session_id}/summary` — generate a study summary
- `POST /api/v1/sessions/{session_id}/flashcards` — generate flashcards for the session

---

## Quickstart

### 1. Start local infrastructure

```bash
docker compose up -d
```

This starts:

- PostgreSQL + pgvector
- Redis

### 2. Configure environment files

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Set at least:

- `GEMINI_API_KEY` in `backend/.env`

Optional backend knobs include:

- `GEMINI_MODEL`
- `EMBEDDING_MODEL_NAME`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_TOP_K`
- `RERANK_ENABLED`
- `RERANK_MODEL_NAME`
- `RERANK_CANDIDATE_K`

### 3. Run the backend

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

Backend default URL:

- `http://localhost:8000`

### 4. Run the frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend default URL:

- `http://localhost:3000`

---

## Typical demo flow

1. Open `http://localhost:3000`
2. Let the app create or restore a chat session
3. Upload a PDF
4. Wait until the document status becomes **ready**
5. Ask a question about the uploaded material
6. Watch the answer stream into the chat
7. Hover citation pills to inspect document/page/snippet context
8. Open the flashcards panel and generate review cards

---

## Development notes

### Useful backend commands

```bash
cd backend
uv run pytest
uv run pytest tests/test_study.py -v
```

### Useful frontend commands

```bash
cd frontend
pnpm lint
pnpm build
```

---

## Evaluation pipeline

The repository includes an evaluation toolkit under [backend/evaluation/](backend/evaluation/).

It supports:

- dataset / Q&A generation from PDFs
- retrieval-only benchmarks
- generation-only runs from saved retrieval artifacts
- full end-to-end evaluation
- experiment comparison and reporting

Quick example:

```bash
cd backend
uv sync --group eval
uv run python -m evaluation.cli retrieval --dataset hotpotqa --num-samples 15
```

For more details, see [backend/evaluation/README.md](backend/evaluation/README.md).

---

## Source-backed implementation highlights

These README notes were aligned with the current codebase, including:

- FastAPI app setup in [backend/app/main.py](backend/app/main.py)
- session, document, chat, and study routes in [backend/app/api/v1/](backend/app/api/v1/)
- ingestion pipeline in [backend/app/rag/ingest.py](backend/app/rag/ingest.py)
- retrieval logic in [backend/app/rag/retriever.py](backend/app/rag/retriever.py)
- Gemini generation in [backend/app/rag/generator.py](backend/app/rag/generator.py)
- frontend chat experience in [frontend/app/page.tsx](frontend/app/page.tsx)
- citation rendering in [frontend/components/chat/CitationPill.tsx](frontend/components/chat/CitationPill.tsx)

---

## License

MIT
