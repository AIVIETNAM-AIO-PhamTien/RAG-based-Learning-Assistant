# RAG Evaluation Pipeline

Evaluation pipeline cho RAG-based Learning Assistant. Chạy hoàn toàn offline (local metrics), chỉ cần Gemini API cho bước sinh answer.

## Cài đặt

```bash
cd backend
uv sync --group eval
```

Yêu cầu:
- `GEMINI_API_KEY` trong file `.env` (cho bước generation)
- Models tự động tải lần đầu: `e5-small-v2` (~130MB), `bge-reranker-base` (~1.1GB), `nli-deberta-v3-small` (~300MB)

## Workflow tổng quan

```
1. Chuẩn bị dataset  ──>  2. Chạy evaluation  ──>  3. Đọc kết quả
   (generate-qa)             (run / compare)          (report)
```

## 1. Chuẩn bị dataset

### Cách 1: Sinh Q/A tự động từ PDF (LLM-assisted)

```bash
uv run python -m evaluation.cli generate-qa \
  --pdf-path evaluation/data/doc.pdf \
  --num-questions 3 \
  --output evaluation/data/qa_dataset.json
```

| Flag | Mô tả |
|------|--------|
| `--pdf-path` | Đường dẫn tới file PDF |
| `--num-questions` | Số câu hỏi sinh ra per chunk (default: 3) |
| `--output` | Đường dẫn file JSON output |

File output sẽ có format:

```json
[
  {
    "question": "RAG là gì?",
    "answer": "RAG là kỹ thuật kết hợp truy xuất thông tin với sinh văn bản.",
    "contexts": ["Đoạn văn bản gốc chứa câu trả lời..."],
    "source_pdf": "doc.pdf",
    "page": 1,
    "difficulty": "medium"
  }
]
```

**Khuyến nghị:** Sau khi sinh, review và chỉnh sửa file JSON thủ công để đảm bảo chất lượng.

### Cách 2: Tự tạo file JSON

Tạo file JSON theo format trên. Đặt vào `evaluation/data/`.

### Cách 3: Dùng research dataset

Các dataset có sẵn: `hotpotqa`, `popqa`, `asqa`, `pubhealth`, `nq`.

```bash
uv run python -m evaluation.cli run --dataset hotpotqa --num-samples 50
```

## 2. Chạy evaluation

### 2.1. Chạy đơn lẻ (`run`)

```bash
uv run python -m evaluation.cli run \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa_dataset.json \
  --num-samples 10
```

Các flag:

| Flag | Default | Mô tả |
|------|---------|--------|
| `--dataset` | `pdf_qa` | Loại dataset (`pdf_qa`, `hotpotqa`, `popqa`, `asqa`, `pubhealth`, `nq`) |
| `--dataset-path` | (bắt buộc cho pdf_qa) | Đường dẫn file/thư mục JSON |
| `--num-samples` | tất cả | Giới hạn số samples |
| `--chunk-size` | 1600 | Kích thước chunk (chars) |
| `--chunk-overlap` | 250 | Overlap giữa chunks |
| `--top-k` | 5 | Số chunks truy xuất |
| `--rerank / --no-rerank` | `--rerank` | Bật/tắt reranker |
| `--rerank-candidate-k` | 20 | Số ứng viên trước rerank |
| `--ragas` | off | Dùng RAGAS metrics thay local metrics (cần API) |
| `--output-dir` | `evaluation/results` | Thư mục lưu kết quả |
| `-v` | off | Verbose logging |

Output: 2 file trong `evaluation/results/`:
- `{name}_report.json` — full report (config + metrics + per-sample results)
- `{name}_samples.csv` — per-sample Q/A và latency

### 2.2. So sánh configs (`compare`)

```bash
uv run python -m evaluation.cli compare \
  --config-file evaluation/configs/default_comparison.yaml \
  --dataset-path evaluation/data/qa_dataset.json \
  --num-samples 20
```

File config YAML định nghĩa nhiều experiments:

```yaml
experiments:
  - name: baseline_no_rerank
    dataset_name: pdf_qa
    chunk_size: 1600
    chunk_overlap: 250
    top_k: 5
    rerank_enabled: false

  - name: with_reranking
    dataset_name: pdf_qa
    chunk_size: 1600
    chunk_overlap: 250
    top_k: 5
    rerank_enabled: true
    rerank_candidate_k: 20
```

Output:
- `comparison.csv` — bảng so sánh (configs x metrics)
- `comparison.tex` — bảng LaTeX cho paper
- `{name}_report.json` — per-experiment

### 2.3. Xuất báo cáo từ kết quả đã lưu (`report`)

```bash
# Bảng terminal
uv run python -m evaluation.cli report --results-dir evaluation/results

# CSV
uv run python -m evaluation.cli report --results-dir evaluation/results --format csv

# LaTeX
uv run python -m evaluation.cli report --results-dir evaluation/results --format latex
```

## 3. Đọc kết quả

### 3.1. Metrics

Pipeline đo 3 nhóm metrics:

#### Retrieval Metrics (đo chất lượng truy xuất)

| Metric | Ý nghĩa | Cách đọc |
|--------|---------|----------|
| `recall_at_5` | % ground-truth contexts tìm thấy trong top-5 | 1.0 = tìm đúng hết. 0.0 = không tìm thấy gì. Target > 0.7 |
| `mrr` | Mean Reciprocal Rank — vị trí trung bình của kết quả đúng đầu tiên | 1.0 = luôn ở vị trí 1. 0.5 = trung bình ở vị trí 2. Target > 0.6 |
| `context_relevance_avg` | Reranker score trung bình cho retrieved contexts | Cao = contexts liên quan đến query. Không có thang cố định, dùng để so sánh giữa configs |

#### Generation Metrics (đo chất lượng câu trả lời)

| Metric | Ý nghĩa | Cách đọc |
|--------|---------|----------|
| `faithfulness_nli` | % câu trong answer được entail bởi contexts (NLI model) | 1.0 = hoàn toàn grounded. 0.0 = hallucination hoàn toàn. Target > 0.7 |
| `answer_similarity` | Cosine similarity giữa question và answer embeddings | 1.0 = rất liên quan. Target > 0.5 |
| `rouge_l` | ROUGE-L F1 giữa answer và ground truth | 1.0 = match hoàn hảo. Thường 0.3-0.6 cho generative answers |
| `token_f1` | Token-level F1 giữa answer và ground truth | Tương tự ROUGE-L nhưng bag-of-words. Thường cao hơn ROUGE-L |
| `citation_coverage_avg` | % citations trong answer là hợp lệ | 1.0 = mọi [n] đều hợp lệ. 0.0 = không dùng citation |

#### Latency Metrics

| Metric | Ý nghĩa |
|--------|---------|
| `retrieval_latency_p50` | Median retrieval time (ms) |
| `retrieval_latency_p95` | 95th percentile retrieval time (ms) |
| `generation_latency_p50` | Median generation time (ms) |
| `generation_latency_p95` | 95th percentile generation time (ms) |

### 3.2. Output files

```
evaluation/results/
  baseline_no_rerank_report.json    # Full report
  baseline_no_rerank_samples.csv    # Per-sample Q/A
  with_reranking_report.json
  with_reranking_samples.csv
  comparison.csv                    # Comparison table
  comparison.tex                    # LaTeX table
```

**`_report.json`** chứa:
```json
{
  "config": { "name": "...", "chunk_size": 1600, ... },
  "aggregate_metrics": {
    "recall_at_5": 0.85,
    "mrr": 0.72,
    "faithfulness_nli": 0.78,
    "answer_similarity": 0.65,
    "rouge_l": 0.42,
    "token_f1": 0.55,
    ...
  },
  "results": [ ... ],
  "duration_seconds": 45.2
}
```

**`_samples.csv`** chứa mỗi hàng là 1 sample:
```
question, ground_truth_answer, generated_answer, retrieval_latency_ms, generation_latency_ms, num_retrieved, citations_used
```

**`comparison.csv`** mỗi hàng là 1 config:
```
name, dataset_name, chunk_size, top_k, rerank_enabled, recall_at_5, mrr, faithfulness_nli, ...
```

## 4. Ví dụ end-to-end

```bash
cd backend

# Bước 1: Sinh Q/A từ PDF
uv run python -m evaluation.cli generate-qa \
  --pdf-path evaluation/data/doc.pdf \
  --num-questions 3 \
  --output evaluation/data/qa.json

# Bước 2: Chạy evaluation đơn lẻ
uv run python -m evaluation.cli run \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa.json \
  --num-samples 10

# Bước 3: So sánh nhiều configs
uv run python -m evaluation.cli compare \
  --config-file evaluation/configs/default_comparison.yaml \
  --dataset-path evaluation/data/qa.json \
  --num-samples 10

# Bước 4: Xuất LaTeX cho paper
uv run python -m evaluation.cli report \
  --results-dir evaluation/results \
  --format latex
```

## 5. Environment variables

| Variable | Mô tả | Bắt buộc |
|----------|--------|----------|
| `GEMINI_API_KEY` | API key cho Gemini (generation + Q/A sinh) | Cho bước generate-qa và run |
| `GEMINI_MODEL` | Model Gemini (default: `gemini-2.5-flash`) | Không |
| `EMBEDDING_MODEL_NAME` | Embedding model (default: `intfloat/e5-small-v2`) | Không |
| `RERANK_MODEL_NAME` | Reranker model (default: `BAAI/bge-reranker-base`) | Không |
| `NLI_MODEL_NAME` | NLI model (default: `cross-encoder/nli-deberta-v3-small`) | Không |

## 6. Chạy tests

```bash
uv run pytest tests/evaluation/ -v
```
