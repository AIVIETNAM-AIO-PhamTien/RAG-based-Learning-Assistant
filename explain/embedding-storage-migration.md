# Vì sao chuyển embedding từ Postgres/pgvector sang Qdrant (bản dễ hiểu)

> TL;DR: **Postgres vẫn giữ nội dung chunk + mọi quan hệ. Qdrant chỉ giữ vector.**
> Đây gọi là kiểu **hybrid**. Không xoá bảng `chunks`, chỉ bỏ đúng **1 cột** `embedding`.

---

## 1. Trước và sau — nhìn 1 phát là hiểu

**TRƯỚC (pgvector):** mọi thứ nằm trong Postgres. Bảng `chunks` vừa giữ chữ, vừa giữ vector.

```
                 ┌──────────────────────── Postgres ────────────────────────┐
   ingest PDF →  │  chunks(id, doc_id, page, text, parent_text, embedding▮)  │
   query      →  │  tìm gần nhất bằng embedding ngay trong bảng (pgvector)   │
                 └───────────────────────────────────────────────────────────┘
```

**SAU (Qdrant hybrid):** tách làm 2 nơi, mỗi nơi làm đúng việc nó giỏi.

```
                 ┌──────────── Postgres (nguồn sự thật) ───────────┐   ┌──── Qdrant (chỉ vector) ────┐
   ingest PDF →  │  chunks(id, doc_id, page, text, parent_text)    │   │  point(id = chunk.id,        │
                 │            ▲ giữ chữ + quan hệ                   │   │        vector = embedding,   │
                 │            │                                    │   │        payload = {doc_id})   │
   query      →  │  ③ lấy row theo id để build citation ───────────┘   │  ① tìm gần nhất (cosine)     │
                 └────────────────────────────────────────────────┘   │     lọc theo doc_id          │
                                        ▲                              └──────────────┬───────────────┘
                                        └──────── ② trả về chunk_id + score ──────────┘
```

Khoá nối 2 nơi: **`chunk.id` (Postgres) == `point.id` (Qdrant)**. Có id là map ngược được ngay.

---

## 2. Luồng chạy sau khi đổi

**Ingest (nạp PDF):**
```
PDF → parse text → chunk → embed (e5-small-v2, 384 chiều)
    → lưu Chunk(text, page, doc_id...) vào Postgres      (KHÔNG còn cột embedding)
    → upsert vector vào Qdrant: id=chunk.id, payload={doc_id}
```

**Query (hỏi đáp):**
```
câu hỏi → embed query
   ① Qdrant: tìm top-k vector gần nhất, CHỈ trong các doc_id của session
   ② Qdrant trả về danh sách (chunk_id, score) đã xếp theo độ giống
   ③ Postgres: lấy các row chunk theo chunk_id (kèm join Document để có tên/trang)
      → sắp lại theo score của Qdrant → rerank → build citation [1][2]
```

Bước rerank, citation, streaming trả lời… **không đổi gì**.

---

## 3. Vì sao KHÔNG "bê hết" sang Qdrant? (câu bạn hỏi về FK)

"Full move" = đưa luôn cả text + metadata vào Qdrant và bỏ bảng `chunks`. Nghe gọn nhưng vỡ 3 thứ:

### a) Mất ràng buộc FK
Hiện tại Postgres ép: **mỗi chunk phải thuộc một document có thật**.

```
documents(id) ──1─────< chunks(doc_id)     ← FK: xoá document thì chunk tự dọn (CASCADE)
```

Qdrant **không có** foreign key. Bỏ bảng `chunks` ⇒ xoá 1 document, vector của nó có thể **bị bỏ lại mồ côi** mà không ai dọn.

### b) Vỡ cách lọc theo session
"Chỉ tìm trong tài liệu của phiên này" hiện làm bằng **JOIN SQL**:

```
chat_session_documents(session_id, document_id)
        │
        └── JOIN documents ── JOIN chunks   → chỉ lấy chunk thuộc session
```

Không còn bảng `chunks` thì **không JOIN được** — phải tự nhét doc_id/session vào payload rồi tự viết lại logic lọc. Tức là **tự dựng lại** đúng cái quan hệ mà DB đang cho miễn phí.

### c) Citation mất điểm neo
Citation `[1] doc_name, trang 3` lấy từ row `chunks` + `documents` đã join. Bê text ra ngoài ⇒ mỗi lần trích dẫn phải gọi Qdrant, và **không còn đảm bảo** chunk được trích có tồn tại.

### Còn nếu "full move mà vẫn giữ bảng chunks"?
Thì FK không vỡ — **nhưng** text nằm ở CẢ hai nơi ⇒ **2 nguồn sự thật**, dễ lệch, tốn gấp đôi.

> Kết: full-move → hoặc **phá quan hệ**, hoặc **nhân đôi dữ liệu**. Hybrid tránh được cả hai.

---

## 4. So sánh nhanh

| | pgvector (cũ) | Qdrant full-move | **Qdrant hybrid (đã chọn)** |
|---|---|---|---|
| Nơi giữ text/metadata | Postgres | Qdrant payload | **Postgres** |
| Nơi giữ vector | Postgres | Qdrant | **Qdrant** |
| FK / quan hệ | ✅ còn | ❌ mất | ✅ **còn nguyên** |
| Nguồn sự thật | 1 (Postgres) | 1 (Qdrant) | **1 (Postgres)** |
| Số chỗ code phải sửa | — | rất nhiều | **ít (3 chỗ)** |

---

## 5. Đổi những gì trong code (đúng 3 chỗ chạm vector)

| File | Trước | Sau |
|---|---|---|
| `backend/app/db/models.py` | có cột `embedding Vector(384)` | **bỏ cột** (Postgres không giữ vector nữa) |
| `backend/app/rag/ingest.py` | `Chunk(..., embedding=...)` | insert chunk + `upsert_chunk_vectors(...)` sang Qdrant |
| `backend/app/rag/retriever.py` | `Chunk.embedding.cosine_distance(...)` | `search_chunk_ids(...)` (Qdrant) rồi fetch row Postgres |
| *(mới)* `backend/app/rag/vector_store.py` | — | wrapper Qdrant: `ensure_collection / upsert / delete / search` |
| `backend/alembic/versions/0003_*.py` | — | migration drop cột `embedding` + HNSW index |

Cấu hình mới: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBEDDING_DIM` (xem `backend/.env.example`).
Chạy Qdrant Cloud free tier; để trống `QDRANT_URL` thì tự dùng Qdrant in-memory (chỉ cho dev, không lưu bền).
