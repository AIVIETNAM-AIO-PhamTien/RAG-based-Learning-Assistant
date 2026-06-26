# RAG Evaluation Pipeline

Evaluation pipeline cho RAG-based Learning Assistant. Chạy hoàn toàn offline (local metrics), chỉ cần Gemini API cho bước sinh answer.

## Quick Start (TL;DR)

```bash
cd backend
uv sync --group eval

# Option A: Benchmark retrieval nhanh (không cần API key)
uv run python -m evaluation.cli retrieval --dataset hotpotqa --num-samples 10

# Option B: Full pipeline với PDF của bạn
uv run python -m evaluation.cli generate-qa --pdf-path evaluation/data/doc.pdf --num-questions 3 --output evaluation/data/qa.json
uv run python -m evaluation.cli full --dataset pdf_qa --dataset-path evaluation/data/qa.json --num-samples 10

# Option C: So sánh nhiều configs
uv run python -m evaluation.cli compare --config-file evaluation/configs/default_comparison.yaml --dataset-path evaluation/data/qa.json --num-samples 20
```

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
   (generate-qa)             (retrieval /          (report)
                              generation /
                              full / compare)
```

Pipeline hỗ trợ 3 execution modes:

| Mode | Command | Mô tả |
|------|---------|--------|
| **Retrieval-only** | `retrieval` | Chỉ chạy retrieval, lưu artifact, tính retrieval metrics |
| **Generation-only** | `generation` | Dùng retrieval artifact có sẵn, chỉ generate + tính full metrics |
| **Full pipeline** | `full` (hoặc `run`) | Chạy toàn bộ: retrieval → generation → metrics |

## 1. Chuẩn bị dataset

### Cách 1: Sinh Q/A tự động từ PDF (LLM-assisted)

Bỏ file pdf tài liệu vào folder evaluation/data

```bash
uv run python -m evaluation.cli generate-qa \
  --pdf-path evaluation/data/doc.pdf \
  --num-questions 3 \
  --output evaluation/data/qa_dataset.json
```
Thay "\" bằng "`" nếu dùng window

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

### Cách 3: Dùng research dataset (chạy trực tiếp không cần chuẩn bị data)

Các dataset có sẵn (tự tải từ HuggingFace):

| Dataset | Flag | Mô tả |
|---------|------|--------|
| `hotpotqa` | `--dataset hotpotqa` | Multi-hop reasoning (có distractor contexts) |
| `popqa` | `--dataset popqa` | Factual retrieval |
| `asqa` | `--dataset asqa` | Long-form ambiguous QA |
| `pubhealth` | `--dataset pubhealth` | Fact verification |
| `nq` | `--dataset nq` | Real search queries (Google) |

Dùng được với mọi command — không cần `--dataset-path`:

```bash
# Chỉ benchmark retrieval
uv run python -m evaluation.cli retrieval --dataset hotpotqa --num-samples 15

# Full pipeline
uv run python -m evaluation.cli full --dataset hotpotqa --num-samples 15

# Retrieval trước, generation sau
uv run python -m evaluation.cli retrieval --dataset popqa --num-samples 20
uv run python -m evaluation.cli generation --from evaluation/results/popqa_5k_rerankTrue/
```

## 2. Chạy evaluation

### 2.1. Retrieval-only (`retrieval`)

Chỉ chạy bước retrieval — không gọi Gemini API, không cần `GEMINI_API_KEY`.
Useful khi muốn benchmark retrieval (tuning `chunk_size`, `top_k`, reranking) mà không tốn API quota.

```bash
uv run python -m evaluation.cli retrieval \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa_dataset.json \
  --num-samples 10
```

Output: thư mục `evaluation/results/{name}/` chứa:
- `retrieval.json` — retrieval artifact (samples + retrieved contexts + scores + retrieval metrics)

Retrieval metrics được in ra terminal: `recall`, `rr` (MRR), `context_relevance`.

**Các flag:**

| Flag | Default | Mô tả |
|------|---------|--------|
| `--dataset` | `pdf_qa` | Loại dataset |
| `--dataset-path` | (bắt buộc cho pdf_qa) | Đường dẫn file/thư mục JSON |
| `--num-samples` | tất cả | Giới hạn số samples |
| `--chunk-size` | 1600 | Kích thước chunk (chars) |
| `--chunk-overlap` | 250 | Overlap giữa chunks |
| `--top-k` | 5 | Số chunks truy xuất |
| `--rerank / --no-rerank` | `--rerank` | Bật/tắt reranker |
| `--rerank-candidate-k` | 20 | Số ứng viên trước rerank |
| `--output-dir` | `evaluation/results` | Thư mục lưu kết quả |
| `--name` | auto-generated | Tên experiment |

### 2.2. Generation-only (`generation`)

Dùng retrieval artifact đã lưu, chỉ chạy bước generation.
Useful khi muốn thử nhiều prompt/model Gemini khác nhau mà không re-run retrieval.

```bash
uv run python -m evaluation.cli generation \
  --from evaluation/results/pdf_qa_5k_rerankTrue/
```

`--from` nhận đường dẫn tới:
- Thư mục experiment (tự tìm `retrieval.json` bên trong)
- File `retrieval.json` trực tiếp

**Override model Gemini:**

```bash
uv run python -m evaluation.cli generation \
  --from evaluation/results/pdf_qa_5k_rerankTrue/ \
  --gemini-model gemini-2.0-flash
```

| Flag | Mô tả |
|------|--------|
| `--from` | (bắt buộc) Path tới retrieval artifact hoặc experiment directory |
| `--name` | Override tên experiment |
| `--gemini-model` | Override model Gemini |
| `--ragas` | Dùng RAGAS metrics (cần API) |

Output: trong cùng thư mục experiment:
- `report.json` — full report (config + all metrics + per-sample results)
- `samples.csv` — per-sample Q/A và latency

**Cảnh báo config mismatch:** Nếu generation config khác retrieval config ở các field liên quan đến retrieval (`chunk_size`, `top_k`, ...), CLI sẽ in warning nhưng vẫn chạy.

### 2.3. Full pipeline (`full`)

Chạy toàn bộ pipeline end-to-end: retrieval → generation → metrics.
Lưu cả retrieval artifact (để reuse sau) lẫn report cuối cùng.

```bash
uv run python -m evaluation.cli full \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa_dataset.json \
  --num-samples 10
```

Output: thư mục `evaluation/results/{name}/` chứa:
- `retrieval.json` — retrieval artifact (reusable cho generation sau)
- `report.json` — full report
- `samples.csv` — per-sample CSV

Các flag giống phần [2.1 Retrieval-only](#21-retrieval-only-retrieval), thêm:

| Flag | Default | Mô tả |
|------|---------|--------|
| `--ragas` | off | Dùng RAGAS metrics (cần API) |

> **Note:** `evaluation run` là alias backward-compatible cho `evaluation full`, nhận cùng flag.

### 2.4. So sánh configs (`compare`)

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

### 2.5. Xuất báo cáo từ kết quả đã lưu (`report`)

```bash
# Bảng terminal
uv run python -m evaluation.cli report --results-dir evaluation/results

# CSV
uv run python -m evaluation.cli report --results-dir evaluation/results --format csv

# LaTeX
uv run python -m evaluation.cli report --results-dir evaluation/results --format latex
```

Tự động scan cả flat files (`*_report.json`) lẫn per-experiment directories (`*/report.json`).

## 3. Đọc kết quả

### 3.1. Metrics

Pipeline đo 3 nhóm metrics:

#### Retrieval Metrics (đo chất lượng truy xuất)

| Metric | Ý nghĩa | Cách đọc |
|--------|---------|----------|
| `recall` | % ground-truth contexts tìm thấy trong top-K | 1.0 = tìm đúng hết. 0.0 = không tìm thấy gì. Target > 0.7 |
| `rr` | Mean Reciprocal Rank — vị trí trung bình của kết quả đúng đầu tiên | 1.0 = luôn ở vị trí 1. 0.5 = trung bình ở vị trí 2. Target > 0.6 |
| `context_relevance` | Reranker score trung bình cho retrieved contexts | Cao = contexts liên quan đến query. Dùng để so sánh giữa configs |

#### Generation Metrics (đo chất lượng câu trả lời)

| Metric | Ý nghĩa | Cách đọc |
|--------|---------|----------|
| `faithfulness_nli` | % câu trong answer được entail bởi contexts (NLI model) | 1.0 = hoàn toàn grounded. 0.0 = hallucination hoàn toàn. Target > 0.7 |
| `answer_similarity` | Cosine similarity giữa question và answer embeddings | 1.0 = rất liên quan. Target > 0.5 |
| `rouge_l` | ROUGE-L F1 giữa answer và ground truth | 1.0 = match hoàn hảo. Thường 0.3-0.6 cho generative answers |
| `token_f1` | Token-level F1 giữa answer và ground truth | Tương tự ROUGE-L nhưng bag-of-words. Thường cao hơn ROUGE-L |
| `citation_coverage` | % citations trong answer là hợp lệ | 1.0 = mọi [n] đều hợp lệ. 0.0 = không dùng citation |

#### Latency Metrics (chỉ với `--ragas`)

| Metric | Ý nghĩa |
|--------|---------|
| `retrieval_latency_p50` | Median retrieval time (ms) |
| `retrieval_latency_p95` | 95th percentile retrieval time (ms) |
| `generation_latency_p50` | Median generation time (ms) |
| `generation_latency_p95` | 95th percentile generation time (ms) |

### 3.2. Output files

```
evaluation/results/
  # Per-experiment directory (staged workflow)
  pdf_qa_5k_rerankTrue/
    retrieval.json          # Retrieval artifact (reusable)
    report.json             # Full report
    samples.csv             # Per-sample Q/A

  # Legacy flat files (compare command)
  comparison.csv            # Comparison table
  comparison.tex            # LaTeX table
```

**`report.json`** chứa:
```json
{
  "config": { "name": "...", "chunk_size": 1600, ... },
  "aggregate_metrics": {
    "recall": 0.85,
    "rr": 0.72,
    "faithfulness_nli": 0.78,
    "answer_similarity": 0.65,
    "rouge_l": 0.42,
    "token_f1": 0.55,
    "citation_coverage": 0.80,
    "context_relevance": 0.73
  },
  "results": [ ... ],
  "stages_completed": ["retrieval", "generation"],
  "duration_seconds": 45.2
}
```

**`retrieval.json`** chứa:
```json
{
  "config": { "name": "...", "chunk_size": 1600, ... },
  "samples": [ { "question": "...", "ground_truth_answer": "...", ... } ],
  "retrieval_results": [ { "retrieved_contexts": [...], "retrieved_scores": [...], "latency_ms": 12.5 } ],
  "retrieval_metrics": { "recall": 0.85, "rr": 0.72, "context_relevance": 0.73 },
  "duration_seconds": 15.3
}
```

**`samples.csv`** chứa mỗi hàng là 1 sample:
```
question, ground_truth_answer, generated_answer, retrieval_latency_ms, generation_latency_ms, num_retrieved, citations_used
```

## 4. Ví dụ end-to-end

### Workflow 1: Full pipeline (đơn giản nhất)

```bash
cd backend

# Bước 1: Sinh Q/A từ PDF
uv run python -m evaluation.cli generate-qa \
  --pdf-path evaluation/data/doc.pdf \
  --num-questions 3 \
  --output evaluation/data/qa.json

# Bước 2: Chạy full evaluation
uv run python -m evaluation.cli full \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa.json \
  --num-samples 10

# Bước 3: Xuất LaTeX cho paper
uv run python -m evaluation.cli report \
  --results-dir evaluation/results \
  --format latex
```

### Workflow 2: Staged (tiết kiệm thời gian khi ablation)

```bash
cd backend

# Bước 1: Chạy retrieval 1 lần
uv run python -m evaluation.cli retrieval \
  --dataset pdf_qa \
  --dataset-path evaluation/data/qa.json \
  --num-samples 50

# Bước 2a: Generate với model mặc định
uv run python -m evaluation.cli generation \
  --from evaluation/results/pdf_qa_5k_rerankTrue/

# Bước 2b: Generate lại với model khác (không re-run retrieval!)
uv run python -m evaluation.cli generation \
  --from evaluation/results/pdf_qa_5k_rerankTrue/ \
  --gemini-model gemini-2.0-flash

# So sánh kết quả
uv run python -m evaluation.cli report --results-dir evaluation/results
```

### Workflow 3: So sánh configs

```bash
cd backend

uv run python -m evaluation.cli compare \
  --config-file evaluation/configs/default_comparison.yaml \
  --dataset-path evaluation/data/qa.json \
  --num-samples 20
```

## 5. Environment variables

| Variable | Mô tả | Bắt buộc |
|----------|--------|----------|
| `GEMINI_API_KEY` | API key cho Gemini (generation + Q/A sinh) | Cho generation và generate-qa. **Không cần cho retrieval-only** |
| `GEMINI_MODEL` | Model Gemini (default: `gemini-2.5-flash`) | Không |
| `EMBEDDING_MODEL_NAME` | Embedding model (default: `intfloat/e5-small-v2`) | Không |
| `RERANK_MODEL_NAME` | Reranker model (default: `BAAI/bge-reranker-base`) | Không |
| `NLI_MODEL_NAME` | NLI model (default: `cross-encoder/nli-deberta-v3-small`) | Không |

## 6. Chạy tests

```bash
uv run pytest tests/evaluation/ -v
```

## 7. CLI reference

```
evaluation generate-qa   Sinh Q/A pairs từ PDF bằng Gemini
evaluation retrieval     Chạy retrieval only, lưu artifact
evaluation generation    Chạy generation từ retrieval artifact có sẵn
evaluation full          Chạy full pipeline (retrieval + generation)
evaluation run           Alias cho full (backward compatible)
evaluation compare       So sánh nhiều configs từ file YAML
evaluation report        Xuất báo cáo từ kết quả đã lưu
```

## 8. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `GEMINI_API_KEY is required` | Chỉ cần cho `generation`, `full`, `generate-qa`. Command `retrieval` và `report` không cần API key |
| Lần đầu chạy rất chậm | Models tự tải lần đầu: e5-small-v2 (~130MB), bge-reranker-base (~1.1GB), nli-deberta (~300MB). Các lần sau dùng cache |
| `OutOfMemoryError` | Giảm `--num-samples`, hoặc dùng `--no-rerank` để bỏ reranker (~1.1GB VRAM) |
| `RateLimitExhausted` | Gemini API bị rate limit. Chờ vài phút rồi chạy lại, hoặc dùng staged workflow: `retrieval` trước, `generation` sau |
| `ModuleNotFoundError: ragas` | RAGAS là optional. Cài: `pip install ragas langchain-google-genai`. Không cần nếu không dùng `--ragas` |
| Kết quả khác nhau giữa các lần chạy | Embedding models là deterministic, nhưng Gemini generation có randomness. Dùng staged workflow để fix retrieval results |
