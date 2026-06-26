from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable

import numpy as np
from google.genai import types

from app.rag.chunker import PageText, chunk_pages
from app.rag.embedder import get_embedder
from app.rag.prompts import SYSTEM_PROMPT, build_prompt
from app.rag.reranker import get_reranker
from app.schemas.chat import Citation
from evaluation.config import get_eval_settings
from evaluation.rate_limiter import RateLimitExhausted, get_rate_limited_client
from evaluation.schemas import (
    EvalResult,
    EvalSample,
    ExperimentConfig,
    GenerationResult,
    RetrievalResult,
)


class InMemoryRetriever:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._texts: list[str] = []

    @property
    def size(self) -> int:
        return len(self._texts)

    def index(self, texts: list[str]) -> None:
        if not texts:
            self._embeddings = None
            self._texts = []
            return

        embedder = get_embedder()
        vectors = embedder.embed_texts(texts)
        self._embeddings = np.array(vectors, dtype=np.float32)
        self._texts = list(texts)

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float, str]]:
        if self._embeddings is None or len(self._texts) == 0:
            return []

        embedder = get_embedder()
        query_vec = np.array(embedder.embed_query(query), dtype=np.float32)
        scores = self._embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]), self._texts[idx]) for idx in top_indices
        ]

    def retrieve_with_rerank(
        self, query: str, top_k: int, candidate_k: int
    ) -> list[tuple[int, float, str]]:
        candidates = self.retrieve(query, candidate_k)
        if not candidates:
            return []

        reranker = get_reranker()
        candidate_texts = [text for _, _, text in candidates]
        ranking = reranker.rerank(query, candidate_texts, top_k)

        return [
            (candidates[orig_idx][0], score, candidates[orig_idx][2])
            for orig_idx, score in ranking
        ]


class EvalPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.retriever = InMemoryRetriever()
        self._chunk_texts: list[str] = []
        self._chunk_to_context: dict[int, int] = {}

    def warmup(self) -> None:
        """Load models with a dummy call so first eval sample latency is clean."""
        logger = logging.getLogger(__name__)
        logger.info("Warming up models...")
        start = time.perf_counter()
        embedder = get_embedder()
        embedder.embed_query("warmup")
        if self.config.rerank_enabled:
            reranker = get_reranker()
            reranker.rerank("warmup", ["warmup document"], 1)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Warmup complete in %.0fms", elapsed)

    def prepare_corpus(
        self,
        contexts: list[str],
        distractors: list[str] | None = None,
    ) -> list[str]:
        all_texts = list(contexts)
        if distractors:
            all_texts.extend(distractors)

        pages = [PageText(page=i, text=t) for i, t in enumerate(all_texts)]
        chunks = chunk_pages(pages, self.config.chunk_size, self.config.chunk_overlap)

        self._chunk_texts = [c.text for c in chunks]

        self._chunk_to_context = {}
        for chunk_idx, chunk in enumerate(chunks):
            self._chunk_to_context[chunk_idx] = chunk.page

        self.retriever.index(self._chunk_texts)
        return self._chunk_texts

    def retrieve(self, query: str) -> RetrievalResult:
        start = time.perf_counter()

        if self.config.rerank_enabled:
            results = self.retriever.retrieve_with_rerank(
                query, self.config.top_k, self.config.rerank_candidate_k
            )
        else:
            results = self.retriever.retrieve(query, self.config.top_k)

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            retrieved_contexts=[text for _, _, text in results],
            retrieved_scores=[score for _, score, _ in results],
            latency_ms=latency,
            requested_k=self.config.top_k,
            effective_k=len(results),
        )

    def generate(self, query: str, retrieved_contexts: list[str]) -> GenerationResult:
        settings = get_eval_settings()
        if not settings.gemini_api_key:
            return GenerationResult(generated_answer="[SKIPPED: no API key]")

        citations = [
            Citation(
                index=i + 1,
                chunk_id="00000000-0000-0000-0000-000000000000",
                doc_id="00000000-0000-0000-0000-000000000000",
                doc_name="eval_doc",
                page=0,
                text=ctx,
                snippet=ctx[:500],
            )
            for i, ctx in enumerate(retrieved_contexts)
        ]

        prompt = build_prompt(query, citations)
        client = get_rate_limited_client()
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        start = time.perf_counter()
        try:
            response = client.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
                config=config,
            )
        except RateLimitExhausted:
            logging.getLogger(__name__).warning(
                "Rate limit exhausted for query: %.50s...", query
            )
            return GenerationResult(generated_answer="[ERROR: rate limited]")
        latency = (time.perf_counter() - start) * 1000

        answer = response.text or ""
        used = [int(m) for m in re.findall(r"\[(\d+)]", answer)]

        return GenerationResult(
            generated_answer=answer,
            latency_ms=latency,
            citations_used=used,
        )

    def retrieve_sample(self, sample: EvalSample) -> RetrievalResult:
        """Run corpus preparation and retrieval for a single sample."""
        distractors = sample.metadata.get("distractor_contexts", [])
        self.prepare_corpus(sample.ground_truth_contexts, distractors)
        return self.retrieve(sample.question)

    def generate_sample(
        self, query: str, retrieval: RetrievalResult
    ) -> GenerationResult:
        """Run generation for a single sample given retrieval results."""
        return self.generate(query, retrieval.retrieved_contexts)

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        retrieval = self.retrieve_sample(sample)
        generation = self.generate_sample(sample.question, retrieval)

        return EvalResult(
            sample=sample,
            retrieval=retrieval,
            generation=generation,
        )

    def retrieve_batch(
        self, samples: list[EvalSample], show_progress: bool = True
    ) -> list[RetrievalResult]:
        """Run retrieval for all samples without generation."""
        results: list[RetrievalResult] = []
        iterator: Iterable[EvalSample] = samples

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(samples, desc="Retrieving", unit="sample")
            except ImportError:
                pass

        for sample in iterator:
            result = self.retrieve_sample(sample)
            results.append(result)

        return results

    def generate_batch(
        self,
        samples: list[EvalSample],
        retrieval_results: list[RetrievalResult],
        show_progress: bool = True,
    ) -> list[GenerationResult]:
        """Run generation for all samples using existing retrieval results."""
        results: list[GenerationResult] = []
        pairs = zip(samples, retrieval_results, strict=True)

        if show_progress:
            try:
                from tqdm import tqdm

                pairs = tqdm(
                    list(pairs), desc="Generating", unit="sample"
                )
            except ImportError:
                pass

        for sample, retrieval in pairs:
            result = self.generate_sample(sample.question, retrieval)
            results.append(result)

        return results

    def evaluate_batch(
        self, samples: list[EvalSample], show_progress: bool = True
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        iterator: Iterable[EvalSample] = samples

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(samples, desc="Evaluating", unit="sample")
            except ImportError:
                pass

        for sample in iterator:
            result = self.evaluate_sample(sample)
            results.append(result)

        return results
