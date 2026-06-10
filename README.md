# RAG-based-Learning-Assistant

Ứng dụng trợ lý học tập dựa trên RAG: nạp tài liệu PDF, tách đoạn, lập chỉ mục tìm kiếm, truy xuất ngữ cảnh và trả lời câu hỏi qua FastAPI.

## Cấu trúc

```text
data/                  PDF đầu vào
src/
  prompts/             Prompt template Jinja2
  interface/           FastAPI app + giao diện web
  evaluation/          Script đánh giá đơn giản
  config.py            Cấu hình project
  schemas.py           Pydantic schemas
  indexing.py          Build chỉ mục từ PDF
  store.py             Lưu/đọc index local
  llm.py               Lớp gọi LLM hoặc fallback local
  filters.py           Lọc truy vấn/ngữ cảnh
  retriever.py         Truy xuất đoạn liên quan
  learning.py          Sinh câu trả lời học tập
store/qdrant/          Thư mục lưu index local
```

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo file `.env` từ mẫu:

```powershell
Copy-Item .env.example .env
```

Nếu muốn dùng OpenAI, điền `OPENAI_API_KEY` trong `.env`. Nếu không, app vẫn chạy ở chế độ fallback local.

## Thêm tài liệu

Đặt file PDF vào thư mục `data/`, sau đó build index:

```powershell
python -m src.indexing
```

## Chạy app

```powershell
uvicorn src.interface.app:app --reload
```

Mở trình duyệt tại `http://127.0.0.1:8000`.

## API nhanh

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/ask `
  -ContentType "application/json" `
  -Body '{"question":"Tài liệu nói gì về RAG?"}'
```
