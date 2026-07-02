# Hướng dẫn set up Qdrant — Cloud free tier & chạy local

Sau khi migrate, embedding được lưu ở **Qdrant** (Postgres chỉ giữ text + metadata).
Bạn cần cho backend biết Qdrant nằm ở đâu qua 4 biến trong `backend/.env`:

```env
QDRANT_URL=            # URL cluster; để TRỐNG = dùng in-memory (ephemeral)
QDRANT_API_KEY=        # API key (chỉ cần cho Cloud)
QDRANT_COLLECTION=chunks
EMBEDDING_DIM=384      # khớp model e5-small-v2 (384 chiều)
```

> Collection **tự được tạo** lúc app khởi động (`ensure_collection`) và lần ingest đầu tiên —
> bạn không cần tạo collection bằng tay.

---

## Phần 1 — Qdrant Cloud (free tier)

Free tier cho **1 cluster 1GB, không cần thẻ tín dụng** — thừa sức cho dev/demo.

1. Vào <https://cloud.qdrant.io> → đăng ký / đăng nhập (Google/GitHub được).
2. **Create a cluster** → chọn gói **Free** (1 GB RAM, 1 node). Chọn region gần bạn
   (vd Singapore/`ap-southeast`) để độ trễ thấp. Đặt tên, bấm **Create**.
3. Đợi cluster chuyển trạng thái **Healthy** (~1–2 phút).
4. Lấy **Cluster URL**: mở cluster → copy Endpoint, dạng:
   ```
   https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx.ap-southeast-1.aws.cloud.qdrant.io:6333
   ```
5. Lấy **API key**: tab **API Keys** (hoặc **Data Access Control**) → **Create** → copy ngay
   (chỉ hiện 1 lần).
6. Điền vào `backend/.env`:
   ```env
   QDRANT_URL=https://xxxxxxxx-....aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=<paste-api-key>
   QDRANT_COLLECTION=chunks
   EMBEDDING_DIM=384
   ```
7. Kiểm tra kết nối:
   ```bash
   curl -s "$QDRANT_URL/collections" -H "api-key: $QDRANT_API_KEY"
   # → {"result":{"collections":[...]},"status":"ok", ...}
   ```

Xong. Khi start backend, collection `chunks` sẽ tự xuất hiện trong dashboard Qdrant Cloud.

---

## Phần 2 — Chạy local (full stack)

**Yêu cầu:** Docker, [uv](https://docs.astral.sh/uv/), pnpm, Python 3.12+.

```bash
# 1. Tạo file env từ mẫu
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# 2. Service local: Postgres + Redis
docker compose up -d

# 3. Cấu hình Qdrant trong backend/.env  → chọn 1 trong 3 cách ở mục 2.1 bên dưới

# 4. Backend
cd backend
uv sync                         # cài deps (đã gồm qdrant-client)
uv run alembic upgrade head     # tạo bảng (chunks KHÔNG còn cột embedding)
uv run uvicorn app.main:app --reload --port 8000

# 5. Frontend (terminal khác)
cd frontend
pnpm install
pnpm dev                        # http://localhost:3000
```

### 2.1 — Chọn nơi chạy Qdrant khi dev

| Cách | Set trong `backend/.env` | Ưu / nhược |
|---|---|---|
| **A. Dùng Cloud cluster** (khuyến nghị) | `QDRANT_URL=https://...:6333`<br>`QDRANT_API_KEY=...` | Giống production, không cài gì thêm. Cần mạng. |
| **B. Qdrant local qua Docker** | `QDRANT_URL=http://localhost:6333`<br>`QDRANT_API_KEY=` (để trống) | Offline, lưu bền trên đĩa. Tốn 1 container. |
| **C. In-memory (zero setup)** | `QDRANT_URL=` (để trống) | Chạy ngay, không cài gì. **Vector KHÔNG lưu bền** — restart là mất, phải re-ingest. Chỉ để thử nhanh. |

**Cách B — chạy Qdrant local bằng Docker:**
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/store/qdrant-data:/qdrant/storage" \
  qdrant/qdrant
# Dashboard: http://localhost:6333/dashboard
```
Hoặc thêm hẳn vào `docker-compose.yml` cho tiện `docker compose up`:
```yaml
  qdrant:
    image: qdrant/qdrant
    container_name: aio-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
# và thêm 'qdrant_data:' vào mục volumes ở cuối file
```

---

## Phần 3 — Verify hoạt động

1. Mở <http://localhost:3000>, **upload 1 PDF**, đợi status `ready`.
2. **Hỏi** 1 câu về nội dung PDF → phải nhận answer streaming kèm citation `[1]`.
3. (Tùy chọn) Kiểm tra vector đã vào Qdrant:
   ```bash
   # Cloud:
   curl -s "$QDRANT_URL/collections/chunks" -H "api-key: $QDRANT_API_KEY" | grep points_count
   # Local Docker:
   curl -s "http://localhost:6333/collections/chunks" | grep points_count
   ```
   `points_count` > 0 nghĩa là embedding đã được lưu sang Qdrant thành công.

---

## Ghi chú

- **Đổi model embedding?** Nếu dùng model khác 384 chiều, sửa `EMBEDDING_DIM` cho khớp
  **và** xoá collection cũ (`chunks`) để nó tạo lại đúng số chiều — vector khác chiều không trộn được.
- **Re-ingest luôn an toàn:** ingest lại 1 document sẽ xoá sạch vector cũ của nó trên Qdrant
  rồi ghi lại (idempotent theo `doc_id`).
- **Postgres vẫn là nguồn sự thật** cho text/metadata/citation — xem
  [`embedding-storage-migration.md`](./embedding-storage-migration.md) để hiểu vì sao.
