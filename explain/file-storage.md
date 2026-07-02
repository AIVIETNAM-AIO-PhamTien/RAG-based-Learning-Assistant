# File storage — solution flow & setup (Local + Cloudflare R2)

> TL;DR: file upload không còn dính chặt vào disk local. Có **1 lớp Storage** trừu tượng,
> chọn backend bằng biến `STORAGE_BACKEND`: **`local`** (mặc định, zero-setup) hoặc
> **`r2`** (Cloudflare R2 — free 10GB, $0 egress). Ứng dụng chỉ làm việc với một **key**
> chuỗi, không quan tâm file nằm ở đâu.

---

## 1. Vì sao đổi

**Trước:** `documents.py` ghi thẳng file vào `data/uploads` và lưu **đường dẫn tuyệt đối** vào
`Document.storage_path`. Vấn đề:
- Deploy thật (Render free…) filesystem **ephemeral** → redeploy là **mất file**.
- Đa instance: instance A ghi, instance B không thấy.
- File chỉ được đọc 1 lần lúc ingest, **không có endpoint serve lại** (không mở được PDF viewer).

**Sau:** tách phần "lưu file ở đâu" ra sau một interface. Dev vẫn dùng disk (không cài gì),
prod bật R2 chỉ bằng đổi env — không sửa code.

---

## 2. Solution flow hiện tại

### Upload
```
Frontend gửi PDF (multipart)
  → POST /api/v1/sessions/{sid}/documents
  → đọc bytes (await file.read())
  → key = "documents/{document_id}/{ten-file}.pdf"
  → get_storage().save(key, bytes)      ── Local: ghi data/uploads/<key>
  │                                         R2:    put_object(bucket, key)
  → tạo Document(storage_path = key)  + link ChatSessionDocument  (Postgres)
  → ingest_document(session, id, bytes)   ← truyền THẲNG bytes, không đọc lại disk
        parse (PyMuPDF từ bytes) → chunk → embed → Qdrant + Postgres
  → status = ready / failed
```

### Download / serve (mở khoá PDF viewer)
```
GET /api/v1/documents/{id}/file
  → đọc Document.storage_path (key) từ Postgres
  → data = get_storage().load(key)      ── Local: đọc file;  R2: get_object
  → Response(data, media_type="application/pdf", inline)
```
Stream đồng nhất cho cả 2 backend (R2 egress free nên không cần presigned URL → đỡ CORS/redirect).

---

## 3. Kiến trúc

```
              app/api/v1/documents.py          app/rag/ingest.py
                       │  get_storage()                │
                       ▼                                 (nhận bytes)
        ┌──────────────────────────────┐
        │  app/storage/  (interface)   │   Storage: save / load / delete
        │  ┌────────────┬────────────┐ │
        │  │ LocalStorage│ R2Storage  │ │   get_storage() chọn theo STORAGE_BACKEND
        │  │  (disk)     │ (boto3/S3) │ │
        │  └────────────┴────────────┘ │
        └──────────────────────────────┘
```

| Thành phần | File | Vai trò |
|---|---|---|
| Interface | `app/storage/base.py` | `Storage.save/load/delete`, `StorageError` |
| Dev backend | `app/storage/local.py` | Ghi/đọc trong `upload_dir`, chặn path-traversal |
| Prod backend | `app/storage/r2.py` | R2 (S3-compatible) qua `boto3`, import boto3 lazy |
| Factory | `app/storage/__init__.py` | `get_storage()` (lru_cache) chọn backend + validate config |

**Lưu ý dữ liệu:** `Document.storage_path` giờ chứa **key** (vd `documents/<uuid>/notes.pdf`),
không phải đường dẫn tuyệt đối. Không cần migration (vẫn là cột Text). Đổi backend local↔R2
**không** đổi giá trị key trong DB — chỉ đổi nơi key đó trỏ tới.

---

## 4. Chạy local (mặc định — không cần cài gì)

Trong `backend/.env`:
```env
STORAGE_BACKEND=local
UPLOAD_DIR=../data/uploads
```
File nằm ở `data/uploads/documents/<document_id>/<ten-file>.pdf`. Xong. Không cần R2, không cần mạng.

---

## 5. Setup Cloudflare R2 (free tier)

R2 free: **10 GB lưu trữ + Class A/B operations free + $0 egress**, không cần thẻ (chỉ cần verify).

1. Đăng nhập <https://dash.cloudflare.com> → menu **R2** → (lần đầu) **Enable R2**.
2. **Create bucket** → đặt tên, ví dụ `aio-pdfs` → chọn location gần (APAC). Bấm Create.
3. Lấy **Account ID**: trang R2 Overview góc phải có `Account ID` (chuỗi hex). Copy.
4. Tạo **API Token (S3 keys)**: R2 → **Manage R2 API Tokens** → **Create API Token**
   → quyền **Object Read & Write** (giới hạn vào bucket vừa tạo cho an toàn) → Create.
   Copy **Access Key ID** và **Secret Access Key** (secret chỉ hiện 1 lần).
5. Điền vào `backend/.env`:
   ```env
   STORAGE_BACKEND=r2
   R2_ACCOUNT_ID=<account-id-hex>
   R2_ACCESS_KEY_ID=<access-key-id>
   R2_SECRET_ACCESS_KEY=<secret-access-key>
   R2_BUCKET=aio-pdfs
   # R2_ENDPOINT tự suy ra: https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
   ```
6. Cài dep + chạy:
   ```bash
   cd backend
   uv sync                       # cài boto3 (chỉ R2 mới cần)
   uv run uvicorn app.main:app --reload --port 8000
   ```

> `boto3` được import **lazy** (chỉ khi backend=r2). Nếu để `STORAGE_BACKEND=local` thì
> không cần boto3 chạy — nhẹ cho dev.

---

## 6. Verify

1. Upload 1 PDF qua UI (hoặc `curl -F file=@sample.pdf .../documents`).
2. Mở `GET http://localhost:8000/api/v1/documents/<id>/file` → phải trả về đúng PDF (inline).
3. Backend R2: vào bucket trên dashboard Cloudflare → thấy object `documents/<id>/<ten>.pdf`.
   Backend local: file nằm trong `data/uploads/documents/<id>/`.

---

## 7. Ghi chú / chưa làm (ngoài phạm vi lần này)

- **Chưa** enforce limit 10MB / 30 trang ở backend (UI mới ghi chú). Có thể chặn ngay sau
  `await file.read()` (check `len(data)`).
- **Chưa** xoá file khi xoá document (hiện chưa có API xoá document). Interface đã có
  `delete(key)` sẵn để nối vào khi làm tính năng đó.
- Serve đang stream qua backend cho đơn giản; nếu sau này cần giảm tải, có thể đổi R2 sang
  **presigned URL** mà không đụng tới `documents.py` (chỉ thêm method vào `Storage`).
