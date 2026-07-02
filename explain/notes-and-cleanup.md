# Lưu ý sau migration & dọn dẹp

Ghi chú các điểm cần biết sau khi chuyển embedding sang Qdrant.

---

## 1. Hai lưu ý bắt buộc

### ⚠️ `backend/uv.lock` đang lệch với `pyproject.toml`
Đã thêm `qdrant-client>=1.12.0` vào `backend/pyproject.toml`, **nhưng chưa cập nhật `uv.lock`**
(máy dev lúc làm không có `uv` để re-lock). Chạy **một lần**:

```bash
cd backend
uv sync        # hoặc: uv lock
```

để `uv.lock` khớp lại với `pyproject.toml`. Nếu không, môi trường cài từ lock sẽ **thiếu
`qdrant-client`** và app fail ngay khi import `app/rag/vector_store.py`.

### 🚫 Giữ `pgvector` trong deps — đừng xoá
Model `Chunk` không còn dùng `pgvector` nữa, nhưng **migration cũ vẫn import nó**:

- `backend/alembic/versions/0001_initial.py` → `from pgvector.sqlalchemy import Vector`
- `backend/alembic/versions/0002_fix_chunk_embedding_dimension.py`
- `backend/alembic/versions/0003_drop_chunk_embedding_column.py` (phần `downgrade`)

Xoá `pgvector` khỏi `pyproject.toml` sẽ làm **vỡ `alembic upgrade head`** trên DB mới
(chạy tuần tự 0001 → 0002 → 0003). Cứ để nguyên — nó nhẹ và vô hại.

> Tương tự, `infra/postgres/init.sql` (`CREATE EXTENSION vector`) giờ không còn cần cho
> vector của app, nhưng vẫn hợp lệ (migration 0001 cũng tạo extension này). Không cần xoá.

---

## 2. Code chết có sẵn trong repo (ngoài phạm vi task)

Những thứ dưới đây **không phải do migration này tạo ra** — đã được commit từ trước và **tách
biệt hẳn** khỏi app thật (`backend/`). Chúng trông như code chết, nhưng **tôi không tự xoá** vì
không chắc bạn còn giữ để tham chiếu. Muốn dọn cái nào thì báo:

| Mục | Là gì | Vì sao trông thừa |
|---|---|---|
| `src/` | Prototype RAG cũ dùng **TF-IDF + scikit-learn + OpenAI**, lưu index ra file `joblib` | App thật là `backend/` (FastAPI). `src/` không được backend import, không liên quan Qdrant/pgvector. |
| `store/` | Thư mục index `joblib` của prototype `src/` (tên `store/qdrant` gây hiểu lầm — **không phải** Qdrant) | Chỉ phục vụ `src/`. |
| `requirements.txt` (root) | Deps của `src/` (`openai`, `scikit-learn`, `fastapi`…) | Backend quản deps riêng bằng `backend/pyproject.toml` + `uv`. File root này không dùng cho backend. |
| `.playwright-mcp/` | Artifact còn sót của công cụ Playwright MCP | Không phải source code của app. |

**Gợi ý:** nếu xác nhận không dùng `src/` nữa, xoá gọn cả cụm `src/`, `store/`, và
`requirements.txt` (root) để repo chỉ còn `backend/` + `frontend/` cho rõ ràng. Nhưng đây là
quyết định của bạn — cần tôi làm thì nói.

---

## 3. Nhắc lại — cần làm khi deploy/chạy mới

1. `cd backend && uv sync` (cập nhật lock + cài `qdrant-client`).
2. Set `QDRANT_URL` / `QDRANT_API_KEY` trong `backend/.env` (xem [`qdrant-setup.md`](./qdrant-setup.md)).
3. `uv run alembic upgrade head` (migration 0003 sẽ drop cột `embedding`).
4. **Re-ingest** lại các PDF cũ — embedding cũ trong Postgres đã bị xoá, phải nạp lại vào Qdrant.
