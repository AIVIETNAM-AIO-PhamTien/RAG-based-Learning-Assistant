from __future__ import annotations

import json
import logging
from pathlib import Path

from google.genai import types

from evaluation.config import get_eval_settings
from evaluation.rate_limiter import RateLimitExhausted, get_rate_limited_client
from evaluation.schemas import EvalSample

logger = logging.getLogger(__name__)

QA_GENERATION_PROMPT = """You are an expert at creating evaluation questions from educational text.
Given the following text chunk from a PDF document, generate {num_questions} question-answer pairs.

Requirements:
- Mix question types: factual recall, conceptual understanding, and reasoning
- Each answer must be directly answerable from the given text
- Provide the exact text passage that contains the answer as "context"
- Output valid JSON array

Text chunk (page {page}):
---
{text}
---

Output format (JSON array):
[
  {{
    "question": "...",
    "answer": "...",
    "context": "exact passage from the text that supports the answer",
    "difficulty": "easy|medium|hard"
  }}
]

Return ONLY the JSON array, no other text."""


class QAGenerator:
    def __init__(self) -> None:
        settings = get_eval_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Q/A generation")
        self._client = get_rate_limited_client()
        self.model = settings.gemini_model

    def generate_from_chunks(
        self,
        chunks: list[dict],
        num_questions_per_chunk: int = 3,
    ) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for chunk in chunks:
            text = chunk["text"]
            page = chunk.get("page", 0)
            source_pdf = chunk.get("source_pdf", "unknown")

            prompt = QA_GENERATION_PROMPT.format(
                num_questions=num_questions_per_chunk, page=page, text=text
            )
            try:
                response = self._client.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
            except RateLimitExhausted:
                logger.warning("Rate limit exhausted, skipping chunk (page %d)", page)
                continue

            if not response.text:
                logger.warning("Empty response for chunk (page %d), skipping", page)
                continue

            try:
                qa_pairs = json.loads(response.text)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed JSON for chunk (page %d), skipping", page)
                continue

            if not isinstance(qa_pairs, list):
                continue

            for i, qa in enumerate(qa_pairs):
                question = qa.get("question")
                answer = qa.get("answer")
                if not question or not answer:
                    continue
                context = qa.get("context", text)
                samples.append(
                    EvalSample(
                        question=question,
                        ground_truth_answer=answer,
                        ground_truth_contexts=[context],
                        metadata={
                            "source": "llm_generated",
                            "source_pdf": source_pdf,
                            "page": page,
                            "difficulty": qa.get("difficulty", "medium"),
                            "chunk_index": i,
                        },
                    )
                )
        return samples

    def generate_from_pdf(
        self,
        pdf_path: str | Path,
        num_questions_per_page: int = 3,
        chunk_size: int = 1600,
        chunk_overlap: int = 250,
    ) -> list[EvalSample]:
        from app.rag.chunker import PageText, chunk_pages
        from app.rag.parser import parse_pdf_text

        pages, _ = parse_pdf_text(str(pdf_path))
        page_texts = [PageText(page=p.page, text=p.text) for p in pages]
        text_chunks = chunk_pages(page_texts, chunk_size, chunk_overlap)

        chunks = [
            {
                "text": tc.text,
                "page": tc.page,
                "source_pdf": Path(pdf_path).name,
            }
            for tc in text_chunks
        ]
        return self.generate_from_chunks(chunks, num_questions_per_page)

    def save(self, samples: list[EvalSample], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "question": s.question,
                "answer": s.ground_truth_answer,
                "contexts": s.ground_truth_contexts,
                "source_pdf": s.metadata.get("source_pdf", ""),
                "page": s.metadata.get("page"),
                "difficulty": s.metadata.get("difficulty", "medium"),
            }
            for s in samples
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
