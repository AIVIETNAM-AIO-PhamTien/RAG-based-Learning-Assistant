# Evaluation Module Map

## Core Files

| File | Tác dụng | Exports chính |
|------|---------|---------------|
| `schemas.py` | Pydantic data models cho toàn bộ pipeline | `EvalSample`, `EvalResult`, `ExperimentConfig`, `ExperimentReport` |
| `config.py` | Settings từ `.env` (API keys, models, params) | `EvalSettings`, `get_eval_settings()` |
| `pipeline.py` | In-memory retrieval + Gemini generation | `InMemoryRetriever`, `EvalPipeline` |
| `metrics.py` | Tính metrics per-sample + aggregate (local + RAGAS opt-in) | `compute_sample_metrics()`, `compute_all_metrics()` |
| `rate_limiter.py` | Retry + rate limiting cho Gemini API (429/503) | `RateLimitedClient`, `get_rate_limited_client()` |
| `experiment.py` | Orchestrate: load dataset → run pipeline → compute metrics → report | `ExperimentRunner` |
| `export.py` | Xuất kết quả: JSON, CSV, LaTeX, terminal table | `export_json()`, `export_csv()`, `export_comparison_csv()` |
| `cli.py` | CLI entry point: `generate-qa`, `run`, `compare`, `report` | `main()` |

## Dataset Loaders (`datasets/`)

| File | Dataset | HuggingFace ID | Mô tả |
|------|---------|----------------|--------|
| `base.py` | — | — | Protocol `DatasetLoader` + factory `get_dataset_loader()` |
| `pdf_qa.py` | pdf_qa | local JSON | Load Q/A từ file JSON (dataset chính) |
| `qa_generator.py` | — | — | Sinh Q/A từ PDF bằng Gemini |
| `hotpotqa.py` | hotpotqa | `hotpotqa/hotpot_qa` | Multi-hop reasoning |
| `popqa.py` | popqa | `akariasai/PopQA` | Factual retrieval |
| `asqa.py` | asqa | `din0s/asqa` | Long-form ambiguous QA |
| `pubhealth.py` | pubhealth | `OpenMed/PubHealth-Processed` | Fact verification |
| `natural_questions.py` | nq | `google-research-datasets/natural_questions` | Real search queries |

## Dependency Flow

```
schemas.py ← config.py
     ↓            ↓
pipeline.py ← rate_limiter.py
     ↓
experiment.py ← metrics.py
     ↓             ↓
  cli.py ← export.py
     ↓
datasets/base.py → loaders
```
