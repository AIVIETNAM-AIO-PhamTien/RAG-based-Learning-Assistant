# Báo cáo cấu trúc source code AIO

## 1. Tổng quan ngắn

AIO là app học tập dạng RAG Chatbot: người dùng upload PDF, hệ thống trích xuất nội dung, chia chunk, tạo embedding, lưu vào Postgres + pgvector, rồi trả lời câu hỏi bằng LLM kèm citation `[1]`, `[2]` có thể hover để xem nguồn.

---

## 2. Tech stack hiện tại

### 2.1 Backend dùng gì?

Backend là FastAPI app chạy bằng Uvicorn, viết Python 3.12+.

- API framework: FastAPI.
- Config/env: Pydantic Settings đọc `.env`.
- Database ORM: SQLAlchemy 2.0 async.
- Migration: Alembic.
- DB driver: asyncpg.
- Vector extension: pgvector qua `pgvector.sqlalchemy.Vector`.
- PDF parser: PyMuPDF.
- Embedding local: `sentence-transformers` với model `intfloat/e5-small-v2`.
- LLM: Google GenAI, model mặc định `gemini-2.5-flash`.
- Test/lint: pytest, pytest-asyncio, ruff, black.

### 2.2 Frontend dùng gì?

Frontend là Next.js app chạy React 19.

- Framework: Next.js 15 App Router.
- UI runtime: React 19.
- Styling: Tailwind CSS v4.
- Component utilities: Radix Slot, class-variance-authority, clsx, tailwind-merge.
- Icon: lucide-react.
- Client API layer: `frontend/lib/api.ts` gọi REST và đọc SSE stream từ backend.
- Citation parser: `frontend/lib/citations.ts` parse pattern `[n]` để render citation pill.

### 2.3 Infra local dùng gì?

- Docker Compose chạy 2 service local: Postgres và Redis.
- Postgres image: `pgvector/pgvector:pg16` để có sẵn extension vector.
- Redis image: `redis:7-alpine`; hiện có trong infra nhưng ingest worker `arq` chưa được implement.
- File upload hiện lưu local vào `data/uploads`, chưa dùng object storage.

### 2.4 RAG pipeline đang chạy qua các step nào?

Ingest PDF:

```text
Frontend upload PDF
→ Backend validate PDF cơ bản
→ Lưu file vào data/uploads
→ Tạo row Document trong Postgres
→ Link Document với ChatSession qua ChatSessionDocument
→ PyMuPDF đọc text từng page
→ Chunk text theo chunk_size=1600, overlap=250
→ sentence-transformers tạo embedding 384d bằng intfloat/e5-small-v2
→ Lưu từng Chunk gồm page, text, parent_text, embedding vào Postgres/pgvector
→ Update Document.status = ready hoặc failed
```

Query/chat:

```text
User gửi câu hỏi từ frontend
→ Backend lưu ChatMessage role=user
→ Embed query bằng cùng model intfloat/e5-small-v2
→ pgvector cosine distance tìm top_k chunk liên quan trong các document của session
→ Build citation payload: index, chunk_id, doc_id, doc_name, page, text, snippet
→ Build prompt từ câu hỏi + retrieved chunks
→ Gemini stream answer qua SSE
→ Backend lưu ChatMessage role=assistant + citations JSONB
→ Frontend nhận token stream, render answer và citation hover
```

Hiện tại retrieval là dense-only baseline. Chưa có BM25, hybrid search, RRF, reranker, MMR, HyDE hoặc citation validator nâng cao.

### 2.5 Database lưu trữ như thế nào?

Postgres là nguồn lưu chính cho metadata, session, message và vector chunk.

- `documents`: metadata file PDF, path local, mime type, status ingest, page_count, error_message.
- `chunks`: nội dung chunk, page, parent_text và embedding vector 384 chiều để tìm kiếm ngữ nghĩa bằng pgvector.
- `chat_sessions`: phiên chat/học tập.
- `chat_session_documents`: bảng nối session ↔ document để mỗi session biết đang hỏi trên tài liệu nào.
- `chat_messages`: lưu user/assistant message; assistant message có citations dạng JSONB.

File PDF gốc không nằm trong DB. DB chỉ lưu `storage_path`; file thật nằm trong `data/uploads`.

### 2.6 SQL trong repo dùng để làm gì?

Repo có 2 loại SQL/schema setup:

1. `infra/postgres/init.sql`
   - Chạy tự động khi Docker tạo Postgres container lần đầu.
   - Mục đích: `CREATE EXTENSION IF NOT EXISTS vector;` để bật pgvector trong database.
   - Nếu thiếu extension này, cột `Vector(384)` và truy vấn cosine distance sẽ không hoạt động.

2. Alembic migrations trong `backend/alembic/versions/`
   - Tạo và cập nhật schema thật của app: bảng `documents`, `chunks`, `chat_sessions`, `chat_session_documents`, `chat_messages`, index.
   - Chạy bằng `uv run alembic upgrade head` trước khi start backend.

Tóm lại: `init.sql` bật năng lực vector cho Postgres; Alembic tạo bảng và index cho application.

---

## 3. Cấu trúc thư mục cấp cao

```text
aio-conquer-microwave/
├── backend/              # FastAPI API + RAG pipeline + DB models + tests
├── frontend/             # Next.js UI chat, upload PDF, citation hover
├── infra/                # Init script cho Postgres/pgvector
├── local/                # Tài liệu nội bộ: plan, architect, UI spec, prompts
├── sample/               # PDF mẫu để demo/test thủ công
├── data/                 # Dữ liệu local, upload PDF
├── docker-compose.yml    # Postgres + Redis local
├── README.md             # Hướng dẫn demo, stack, phase roadmap
└── .env.example          # Biến môi trường mẫu cấp root
```

---

## 4. Backend

### 4.1 Mục đích

Backend chịu trách nhiệm:

- Tạo chat session.
- Upload PDF và ingest vào vector database.
- Retrieve chunk liên quan theo câu hỏi.
- Gọi Gemini để sinh câu trả lời streaming.
- Lưu message và citation.
- Expose API cho frontend.

### 4.2 Cấu trúc chính

```text
backend/
├── app/
│   ├── main.py              # FastAPI entrypoint, CORS, include router
│   ├── config.py            # Pydantic Settings, env, model config, RAG config
│   ├── api/v1/              # API endpoints
│   ├── db/                  # SQLAlchemy models + session factory
│   ├── rag/                 # RAG pipeline basic hiện tại
│   └── schemas/             # Pydantic request/response models
├── alembic/                 # Database migrations
├── tests/                   # Unit/integration tests hiện có
├── pyproject.toml           # Python deps, ruff, black, pytest config
└── Dockerfile
```

### 4.3 API quan trọng

| File | Mục đích |
|---|---|
| `backend/app/api/v1/health.py` | `GET /healthz`, health check đơn giản. |
| `backend/app/api/v1/sessions.py` | Tạo chat session, list documents theo session. |
| `backend/app/api/v1/documents.py` | Upload PDF vào session, lưu file local, gọi ingest. |
| `backend/app/api/v1/chat.py` | Nhận câu hỏi, retrieve chunk, stream answer qua SSE. |
| `backend/app/api/v1/__init__.py` | Gom router API v1. |

### 4.4 Database quan trọng

| File | Mục đích |
|---|---|
| `backend/app/db/models.py` | Model `Document`, `Chunk`, `ChatSession`, `ChatSessionDocument`, `ChatMessage`. |
| `backend/app/db/session.py` | Tạo async engine/session từ `database_url`. |
| `backend/alembic/versions/0001_initial.py` | Migration schema ban đầu. |
| `backend/alembic/versions/0002_fix_chunk_embedding_dimension.py` | Sửa dimension embedding. |

Bảng chính:

- `documents`: metadata PDF, status ingest, path file.
- `chunks`: chunk text, page, parent_text, embedding vector 384d.
- `chat_sessions`: phiên học/chat.
- `chat_session_documents`: mapping session ↔ document.
- `chat_messages`: user/assistant messages + citations JSONB.

---

## 5. Frontend

### 5.1 Mục đích

Frontend là giao diện chat dạng NotebookLM-lite:

- Tạo hoặc khôi phục chat session từ `localStorage`.
- Upload PDF bằng click hoặc drag/drop.
- Hiển thị trạng thái ingest document.
- Gửi câu hỏi và nhận answer streaming.
- Parse citation `[n]` và render pill hover tooltip.
- Có panel Flashcards UI nhưng chưa có backend/data thật.

### 5.2 Cấu trúc chính

```text
frontend/
├── app/
│   ├── page.tsx          # Trang chính: layout, session, upload, chat streaming
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Theme/tokens/global style
├── components/chat/
│   ├── ChatInput.tsx
│   ├── CitationPill.tsx
│   ├── DocumentStatusList.tsx
│   ├── DocumentUploader.tsx
│   └── MessageList.tsx
├── lib/
│   ├── api.ts            # Client gọi backend REST/SSE
│   ├── citations.ts      # Parse `[1]` thành text/citation parts
│   ├── types.ts          # TypeScript types FE
│   └── utils.ts
├── package.json
└── Dockerfile
```

### 5.3 File frontend quan trọng

| File | Mục đích |
|---|---|
| `frontend/app/page.tsx` | Điều phối toàn bộ UI: session, upload, refresh document, stream chat, flashcard panel. |
| `frontend/lib/api.ts` | Gọi `POST /sessions`, upload document, list documents, đọc SSE chat. |
| `frontend/lib/citations.ts` | Regex parse citation pattern `[n]`. |
| `frontend/components/chat/MessageList.tsx` | Render user/assistant messages, citation pill, streaming cursor. |
| `frontend/components/chat/CitationPill.tsx` | Tooltip hover hiển thị `doc_name`, `page`, `snippet`. |
| `frontend/components/chat/DocumentUploader.tsx` | Upload PDF bằng input hoặc drag/drop. |
| `frontend/components/chat/DocumentStatusList.tsx` | Hiển thị status `pending`, `processing`, `ready`, `failed`. |

---

## 6. Infra và cấu hình

| File/folder | Mục đích |
|---|---|
| `docker-compose.yml` | Chạy `postgres` image `pgvector/pgvector:pg16` và `redis:7-alpine`. |
| `infra/postgres/init.sql` | `CREATE EXTENSION IF NOT EXISTS vector;`. |
| `backend/.env.example` | Env mẫu backend. |
| `frontend/.env.example` | Env mẫu frontend. |
| `README.md` | Demo commands, stack mục tiêu, flow demo, phase roadmap. |
| `local/plan.md` | Roadmap chi tiết, tech stack, folder target, checklist Phase 1/2. |
| `local/architect.md` | Kiến trúc RAG từ naive → advanced → modular. |
| `local/ui-spec.md` | Spec UI dark focus mode + flashcards. |

---

## 7. RAG pipeline basic hiện tại

### 7.1 Luồng ingest PDF hiện tại

```text
Upload PDF
→ validate file PDF
→ lưu vào data/uploads
→ tạo Document + link vào ChatSessionDocument
→ ingest_document()
→ parse_pdf_text()
→ chunk_pages()
→ embed_texts()
→ lưu Chunk vào Postgres/pgvector
→ set Document.status = ready hoặc failed
```

File liên quan:

- `backend/app/api/v1/documents.py`
- `backend/app/rag/ingest.py`
- `backend/app/rag/parser.py`
- `backend/app/rag/chunker.py`
- `backend/app/rag/embedder.py`
- `backend/app/db/models.py`

### 7.2 Luồng query/chat hiện tại

```text
User hỏi
→ POST /api/v1/sessions/{session_id}/chat
→ lưu user message
→ embed query
→ dense search top-k bằng pgvector cosine distance
→ build citations payload
→ build prompt từ chunks
→ Gemini stream answer token-by-token
→ lưu assistant message + citations
→ gửi event citations
→ gửi event done
```

File liên quan:

- `backend/app/api/v1/chat.py`
- `backend/app/rag/retriever.py`
- `backend/app/rag/generator.py`
- `backend/app/rag/prompts.py`
- `backend/app/rag/metrics.py`
- `frontend/lib/api.ts`
- `frontend/app/page.tsx`

### 7.3 Những gì đã làm

- Đã có upload PDF theo session.
- Đã validate file PDF cơ bản bằng extension/content-type.
- Đã parse text-layer PDF bằng PyMuPDF.
- Đã reject PDF encrypted hoặc không có text layer.
- Đã chunk text theo fixed-size + overlap.
- Đã tạo embedding bằng `intfloat/e5-small-v2`.
- Đã normalize embedding và lưu vector 384d vào pgvector.
- Đã retrieve dense-only top-k bằng cosine distance.
- Đã build citation gồm `index`, `chunk_id`, `doc_id`, `doc_name`, `page`, `text`, `snippet`.
- Đã gọi Gemini để stream answer qua SSE.
- Đã có prompt yêu cầu trả lời chỉ dựa trên chunk và cite `[n]`.
- Đã tính `citation_coverage` cơ bản.
- Đã lưu chat messages vào DB.
- Đã có frontend render streaming answer.
- Đã có citation hover tooltip hiển thị document, page, snippet.
- Đã có tests cơ bản cho chunker, health, document, schema.

### 7.4 Những gì chưa làm trong RAG pipeline

- Chưa có background job queue thật cho ingest; upload request đang ingest trực tiếp, dễ block request lâu.
- Chưa có Redis/arq worker dù Redis đã có trong Docker Compose và plan.
- Chưa có BM25/Postgres full-text search.
- Chưa có hybrid retrieval BM25 + dense.
- Chưa có RRF fusion.
- Chưa có reranker `bge-reranker-v2-m3`.
- Chưa có MMR diversify.
- Chưa có multi-query rewrite.
- Chưa có HyDE.
- Chưa có citation validator để strip citation `[n]` bị hallucinate.
- Chưa có confidence threshold / CRAG-lite để trả lời “không tìm thấy” khi retrieval yếu.
- Chưa có bbox/highlight metadata từ PDF.
- Chưa có PDF viewer và click citation để mở đúng trang/highlight.
- Chưa có gold set 30 câu và eval hit@5 tự động đầy đủ.
- Chưa có Langfuse/RAGAS observability/eval.
- Chưa có provider fallback Gemini → Cerebras → Ollama.
- Chưa có storage R2; hiện đang lưu file local.
- Chưa có upload size limit thực thi ở backend, trong UI chỉ ghi “Max 10MB”.
- Chưa có giới hạn “First 30 pages” thực thi; parser đang đọc toàn bộ text-layer pages.
- Chưa có prompt-injection defense cho nội dung chunk.

### 7.5 Lệch giữa plan và code hiện tại

- Plan ban đầu nói embedding `bge-m3` 1024d; code hiện dùng `intfloat/e5-small-v2` 384d.
- Plan có parent/child chunking; code hiện dùng fixed-size chunk và `parent_text = chunk_text`.
- Plan Phase 1 có gold set baseline; repo hiện chưa thấy `gold_set.jsonl`.
- Plan có arq worker; code hiện chưa có `workers/`.
- README nói stack có reranker, hybrid, RRF, MMR; code hiện mới là dense-only baseline.

---

## 8. Tính năng hiện tại chưa complete

### 8.1 Frontend/UI

- `New Session` button đang disabled.
- History chỉ hiển thị current session, chưa có API list/rename/delete sessions.
- Account/footer là mock (`student@aio.local`), chưa có auth/logout.
- `Attach PDF` trong chat input đang disabled; upload chỉ qua sidebar uploader.
- Flashcards panel mới là UI placeholder, chưa có API/model/data.
- Chưa hỗ trợ Markdown đầy đủ cho assistant answer.
- Chưa có PDF viewer route `/viewer`.
- Citation chỉ hover, chưa click-to-open PDF/page/highlight.
- UI nói “Max 10MB · First 30 pages will be read” nhưng backend chưa enforce.

### 8.2 Backend/API

- Chưa có API list tất cả sessions.
- Chưa có rename/delete session.
- Chưa có delete document/remove attachment khỏi session.
- Chưa có endpoint stream/download PDF.
- Chưa có flashcard APIs.
- Chưa có auth/multi-user isolation.
- Chưa có rate limiting.
- Chưa có production security headers/CSP.
- Chưa có structured API response envelope thống nhất.

### 8.3 Data/RAG/Eval

- Chưa có hybrid search và reranking.
- Chưa có eval dataset chuẩn.
- Chưa có tracing/observability.
- Chưa có robust citation validation.
- Chưa có async ingestion worker.
- Chưa có object storage production.

---

## 9. Planning future đề xuất

### Phase gần nhất: hoàn thiện Baseline RAG

1. Enforce upload limits ở backend: PDF-only, max size, max pages.
2. Chuyển ingest sang background worker `arq + Redis`.
3. Thêm API polling document status ổn định hơn.
4. Thêm citation validator basic: chỉ giữ `[n]` nằm trong citations trả về.
5. Tạo `gold_set.jsonl` khoảng 30 câu hỏi và script eval `hit@5`, `citation_coverage`.
6. Bổ sung test cho retriever, ingest failure, chat SSE.
7. Làm rõ trong README: hiện tại là dense-only baseline, chưa phải advanced RAG.

### Phase 2: Advanced RAG

1. Thêm `tsvector` cho chunks và GIN index.
2. Implement BM25 + dense retrieval.
3. Fuse bằng RRF.
4. Rerank candidates bằng `bge-reranker-v2-m3`.
5. MMR để giảm chunk trùng lặp.
6. Multi-query rewrite bằng Gemini.
7. Confidence threshold để trả lời “không tìm thấy trong tài liệu” khi context yếu.
8. Log trace/eval vào Langfuse hoặc file local trước.

### Phase 3: UX NotebookLM-lite

1. Lưu bbox/page-level block từ PyMuPDF.
2. Thêm endpoint serve PDF.
3. Thêm frontend route `/viewer`.
4. Click citation → mở đúng PDF/page.
5. Highlight đoạn citation nếu có bbox.
6. Render assistant message bằng Markdown.

### Phase 4: Study features

1. Flashcard generation theo session.
2. Review flashcards: known/learning/not_reviewed.
3. Quiz mode: MCQ + free-response.
4. Session history đầy đủ: create/list/rename/delete.
5. Auth/multi-user nếu deploy cho nhiều người dùng.

### Phase 5: Production/deploy

1. Cloudflare R2 cho PDF storage.
2. Neon Postgres production.
3. Upstash Redis cho worker queue.
4. Render/Fly/Oracle VM cho backend.
5. Vercel cho frontend.
6. CSP/security headers/rate limit.
7. CI checks: backend tests, frontend lint/build, migration check.

---

## 10. Kết luận

Source hiện đã có một baseline RAG end-to-end chạy được: upload PDF → ingest → dense retrieve → Gemini streaming answer → citation hover trên frontend.

Phần còn thiếu chủ yếu nằm ở 4 nhóm:

1. Advanced retrieval: BM25, RRF, reranker, MMR, query rewrite.
2. Reliability: background worker, limits, eval, observability, citation validator.
3. UX NotebookLM: PDF viewer, click citation, bbox highlight.
4. Product features: flashcards, session history, auth, deployment production.
