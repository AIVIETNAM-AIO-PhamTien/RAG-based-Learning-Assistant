# Evaluation Module Map

## Core Files

| File | Tác dụng | Exports chính |
|------|---------|---------------|
| `schemas.py` | Pydantic data models cho toàn bộ pipeline | `EvalSample`, `EvalResult`, `ExperimentConfig`, `ExperimentReport`, `RetrievalArtifact`, `GenerationArtifact` |
| `config.py` | Settings từ `.env` (API keys, models, params) | `EvalSettings`, `get_eval_settings()` |
| `pipeline.py` | In-memory retrieval + Gemini generation | `InMemoryRetriever`, `EvalPipeline` |
| `metrics.py` | Tính metrics per-sample + aggregate (local + RAGAS opt-in) | `compute_retrieval_metrics()`, `compute_generation_metrics()`, `compute_sample_metrics()`, `compute_all_metrics()` |
| `rate_limiter.py` | Retry + rate limiting cho Gemini API (429/503) | `RateLimitedClient`, `get_rate_limited_client()` |
| `experiment.py` | Orchestrate: load dataset → run pipeline → compute metrics → report | `ExperimentRunner` |
| `artifacts.py` | Save/load/validate intermediate artifacts giữa stages | `save_artifact()`, `load_retrieval_artifact()`, `resolve_retrieval_artifact()`, `validate_config_compatibility()` |
| `export.py` | Xuất kết quả: JSON, CSV, LaTeX, terminal table | `export_json()`, `export_csv()`, `export_comparison_csv()` |
| `cli.py` | CLI entry point: `generate-qa`, `retrieval`, `generation`, `full`, `run`, `compare`, `report` | `main()` |

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

## Dependency Flow (imports)

```
               schemas.py        config.py
                |    |              |
                v    |              v
           pipeline.py  <---- rate_limiter.py
                |
                v
 metrics.py --> experiment.py
                |
                v
artifacts.py --> cli.py <-- export.py
                |
                v
          datasets/base.py --> [loaders]
```

## Data Flow (runtime)

```
Dataset Loader
      |
      v
EvalSample[]
      |
      v
EvalPipeline.retrieve_batch()  ──>  RetrievalResult[]  ──>  retrieval.json (artifact)
      |                                     |
      v                                     v
EvalPipeline.generate_batch()  ──>  GenerationResult[]
      |
      v
compute_*_metrics()  ──>  metric_scores per sample
      |
      v
ExperimentReport  ──>  export (JSON / CSV / LaTeX / terminal table)
```

## Thêm metric mới

1. Thêm computation vào `metrics.py`:
   - Retrieval metric → thêm key vào `compute_retrieval_metrics()`
   - Generation metric → thêm key vào `compute_generation_metrics()`
2. Cập nhật `_RET_PREFIXES` hoặc `_GEN_PREFIXES` trong `experiment.py` để `_group_metrics()` phân loại đúng
3. Thêm unit test trong `tests/evaluation/test_metrics.py`
4. Cập nhật `expected_keys` trong `test_compute_sample_metrics_returns_all_keys`

## Thêm dataset mới

1. Tạo `datasets/my_dataset.py` implement protocol `DatasetLoader` từ `datasets/base.py`:
   ```python
   class MyDatasetLoader:
       def load(self, *, num_samples: int | None = None) -> list[EvalSample]:
           ...
   ```
2. Đăng ký trong `datasets/base.py` → hàm `get_dataset_loader()`:
   ```python
   "my_dataset": lambda **kw: MyDatasetLoader(**kw),
   ```
3. Thêm test trong `tests/evaluation/test_datasets.py`
4. Cập nhật bảng dataset trong `README.md`
