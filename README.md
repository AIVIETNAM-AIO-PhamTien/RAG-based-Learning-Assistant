# AIO

<p align="center">
  A study-oriented RAG application for learning from PDFs with grounded chat, inline citations, and session-based flashcards.
</p>

<p align="center">
  <img alt="FastAPI" src="https://img.shields.io/badge/backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Next.js 15" src="https://img.shields.io/badge/frontend-Next.js%2015-111111?style=flat-square&logo=nextdotjs&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/database-PostgreSQL%20%2B%20pgvector-336791?style=flat-square&logo=postgresql&logoColor=white">
  <img alt="Gemini" src="https://img.shields.io/badge/generation-Gemini%202.5%20Flash-4f46e5?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20prototype-7c3aed?style=flat-square">
</p>

AIO lets you upload lecture slides, notes, or textbooks as PDFs, then ask questions against the uploaded material. The app streams answers in real time, attaches inline citations such as `[1]` and `[2]`, and supports flashcard generation from the active study session.

---

## Table of contents

- [Current status](#current-status)
- [UI preview](#ui-preview)
- [What works today](#what-works-today)
- [Known constraints](#known-constraints)
- [Architecture at a glance](#architecture-at-a-glance)
- [Quickstart](#quickstart)
- [Demo flow](#demo-flow)
- [API overview](#api-overview)
- [Evaluation pipeline](#evaluation-pipeline)
- [Repository structure](#repository-structure)
- [Implementation references](#implementation-references)
- [License](#license)

---

## Current status

### In the current app

- Session-based chat workspace is implemented
- PDF upload and ingestion is implemented
- Retrieval-augmented chat with streaming SSE responses is implemented
- Inline citation hover cards are implemented
- Flashcard generation for a session is implemented
- Evaluation tooling exists under [backend/evaluation/](backend/evaluation/)

### Important scope notes

- The **study summary** endpoint exists on the backend, but the **current frontend does not expose it yet**
- **Redis is present in local infrastructure**, but **the current running app code does not use Redis yet**
- The current product is best described as an **active prototype / learning project**, not a finished production platform

---

## UI preview

> **Screenshot placeholder**
>
> Replace this block once you have a UI screenshot.
>
> Suggested asset path:
>
> `docs/images/aio-ui.png`
>
> Suggested markdown:
>
> `![AIO UI](docs/images/aio-ui.png)`

---

## What works today

### User-facing features

- **Session-based study workspace**
  - create, rename, switch, and delete sessions
  - each session keeps its own documents and message history
- **PDF upload and ingestion**
  - uploads are attached to a session
  - text-layer PDFs are parsed into chunks and embedded for retrieval
- **Grounded chat with citations**
  - answers stream token-by-token over SSE
  - citations are rendered inline as `[n]`
- **Citation hover cards**
  - hover reveals document name, page number, and snippet text
- **Flashcard generation**
  - generates flashcards from ready documents in the current session
  - supports regeneration from the same session context
- **Evaluation pipeline**
  - includes retrieval and generation benchmarking tools under [backend/evaluation/](backend/evaluation/)

---

## Known constraints

- Optimized for **text-layer PDFs**
- **Encrypted PDFs are not supported**
- If a PDF has no readable text layer, ingestion fails
- Chat and study generation require a valid **Gemini API key**
- Redis is configured in local infra but is **not yet used** for jobs, caching, or sessions
- The frontend currently surfaces **chat + citations + flashcards**, but not the backend summary flow

---

## Architecture at a glance

### Backend

- **Framework:** FastAPI
- **ORM / migrations:** SQLAlchemy 2 async + Alembic
- **Database:** PostgreSQL + pgvector
- **PDF parsing:** PyMuPDF
- **Embeddings:** `intfloat/e5-small-v2`
- **Reranking:** `BAAI/bge-reranker-base`
- **Generation:** Gemini (`gemini-2.5-flash` by default)

### Frontend

- **Framework:** Next.js 15
- **UI stack:** React 19 + Tailwind CSS 4
- **Interaction model:** sidebar sessions, upload panel, streaming chat, flashcards panel

### Runtime flow

1. A user opens the app and a chat session is restored or created.
2. The user uploads a PDF into that session.
3. The backend parses PDF text, chunks pages, computes embeddings, and stores chunks in PostgreSQL.
4. The user asks a question.
5. The backend embeds the query, retrieves relevant chunks from the session corpus, optionally reranks them, and builds the prompt for Gemini.
6. Gemini streams the answer back through SSE.
7. The frontend renders inline citations and hoverable source snippets.
8. The user can generate flashcards from representative chunks in the same session.

---

## Quickstart

### 1. Start local infrastructure

```bash
docker compose up -d
```

This starts:

- PostgreSQL + pgvector
- Redis *(included in local infra, currently unused by the app runtime)*

### 2. Configure environment files

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Minimum required setup:

- set `GEMINI_API_KEY` in `backend/.env`

Useful backend settings include:

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

Backend URL:

- `http://localhost:8000`

### 4. Run the frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend URL:

- `http://localhost:3000`

### 5. Upload a PDF

The easiest way to test the app is through the UI:

1. Open `http://localhost:3000`
2. Let the app create a session automatically
3. Upload your own text-based PDF
4. Wait until the document status becomes `ready`
5. Start asking questions

If you want to test upload via API instead, create a session first:

```bash
curl -X POST http://localhost:8000/api/v1/sessions/
```

Then upload a PDF using the returned `session_id`:

```bash
curl -F file=@/path/to/your.pdf http://localhost:8000/api/v1/sessions/<session_id>/documents
```

---

## Demo flow

A simple manual demo looks like this:

1. Start backend and frontend locally
2. Open `http://localhost:3000`
3. Upload a text-based PDF
4. Wait for the document status to turn `ready`
5. Ask a question about the uploaded material
6. Watch the answer stream into the chat UI
7. Hover citation pills to inspect document/page/snippet context
8. Open the flashcards panel and generate study cards for the session

---

## API overview

### Health

- `GET /healthz`

### Sessions

- `POST /api/v1/sessions/` — create a session
- `GET /api/v1/sessions/` — list sessions
- `PATCH /api/v1/sessions/{session_id}` — rename a session
- `DELETE /api/v1/sessions/{session_id}` — delete a session
- `GET /api/v1/sessions/{session_id}/documents` — list session documents
- `GET /api/v1/sessions/{session_id}/messages` — list session messages

### Documents

- `POST /api/v1/sessions/{session_id}/documents` — upload a PDF to a session
- `GET /api/v1/documents/{document_id}` — get document metadata and status

### Chat

- `POST /api/v1/sessions/{session_id}/chat` — stream an answer with SSE events
  - emits `token`, `citations`, `done`, and `error`

### Study tools

- `POST /api/v1/sessions/{session_id}/summary` — generate a study summary *(backend-only for now)*
- `POST /api/v1/sessions/{session_id}/flashcards` — generate flashcards for the session

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

For full usage details, see [backend/evaluation/README.md](backend/evaluation/README.md).

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
├── docs/                 # documentation assets and project docs
├── infra/                # local database initialization and deployment notes
├── data/                 # uploaded documents / local data
└── docker-compose.yml    # local Postgres + Redis services
```

---

## Implementation references

This README was aligned with the current codebase, including:

- FastAPI app setup in [backend/app/main.py](backend/app/main.py)
- session, document, chat, and study routes in [backend/app/api/v1/](backend/app/api/v1/)
- ingestion pipeline in [backend/app/rag/ingest.py](backend/app/rag/ingest.py)
- retrieval logic in [backend/app/rag/retriever.py](backend/app/rag/retriever.py)
- Gemini generation in [backend/app/rag/generator.py](backend/app/rag/generator.py)
- frontend chat experience in [frontend/app/page.tsx](frontend/app/page.tsx)
- citation rendering in [frontend/components/chat/CitationPill.tsx](frontend/components/chat/CitationPill.tsx)

---

## License

This project is licensed under the [MIT License](LICENSE).
