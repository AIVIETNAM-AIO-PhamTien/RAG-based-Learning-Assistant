# Setup end-to-end để chạy source code

## 1. Yêu cầu trước khi chạy

Cài sẵn:

- Docker Desktop
- Python 3.12+
- `uv`
- Node.js 20+
- `pnpm`

## 2. Tạo file môi trường

Tại thư mục gốc project:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Mở `backend/.env` và điền nếu có:

```env
GEMINI_API_KEY=your_key_here
```

Không có key vẫn có thể khởi động app, nhưng tính năng hỏi đáp bằng Gemini có thể không chạy đầy đủ.

## 3. Chạy database và Redis

```bash
docker compose up -d
```

Lệnh này mở:

- Postgres + pgvector: `localhost:5432`
- Redis: `localhost:6379`

## 4. Setup và chạy backend

Terminal 1:

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

Backend chạy tại:

```text
http://localhost:8000
```

## 5. Setup và chạy frontend

Terminal 2:

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend chạy tại:

```text
http://localhost:3000
```

Frontend sẽ gọi backend qua biến:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## 6. Kiểm tra flow end-to-end

Mở trình duyệt:

```text
http://localhost:3000
```

Thử theo thứ tự:

1. Upload một file PDF trong thư mục `sample/`.
2. Đợi backend ingest tài liệu.
3. Hỏi một câu liên quan nội dung PDF.
4. Kiểm tra câu trả lời streaming.
5. Hover citation `[1]`, `[2]` để xem nguồn và số trang.

Có thể test upload nhanh bằng API:

```bash
curl -F file=@sample/linear-algebra-ch3.pdf http://localhost:8000/api/v1/documents
```

## 7. Dừng hệ thống

Dừng frontend/backend bằng `Ctrl+C` ở từng terminal.

Dừng Postgres và Redis:

```bash
docker compose down
```

Nếu muốn xoá luôn dữ liệu local:

```bash
docker compose down -v
```

## 8. Lệnh kiểm tra nhanh

Backend:

```bash
cd backend
uv run pytest
uv run ruff check .
```

Frontend:

```bash
cd frontend
pnpm lint
pnpm build
```
