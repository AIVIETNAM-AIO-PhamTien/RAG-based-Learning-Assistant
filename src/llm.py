from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import get_settings
from src.schemas import SourceChunk


class LearningLLM:
    def __init__(self):
        self.settings = get_settings()
        prompt_dir = Path(__file__).parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(prompt_dir),
            autoescape=select_autoescape(default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def answer(self, question: str, sources: list[SourceChunk]) -> str:
        if not sources:
            return "Mình chưa tìm thấy thông tin liên quan trong tài liệu đã index. Hãy thêm PDF vào thư mục data và chạy lại indexing."

        if self.settings.openai_api_key:
            return self._answer_with_openai(question, sources)
        return self._fallback_answer(question, sources)

    def _answer_with_openai(self, question: str, sources: list[SourceChunk]) -> str:
        from openai import OpenAI

        template = self.env.get_template("answer.jinja2")
        prompt = template.render(question=question, sources=sources)
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý học tập RAG. Trả lời bằng tiếng Việt, có căn cứ từ tài liệu."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _fallback_answer(self, question: str, sources: list[SourceChunk]) -> str:
        lines = [
            f"Câu hỏi: {question}",
            "",
            "Mình tìm thấy các đoạn liên quan sau trong tài liệu:",
        ]
        for index, source in enumerate(sources, start=1):
            snippet = source.text.replace("\n", " ")
            if len(snippet) > 650:
                snippet = snippet[:650].rsplit(" ", 1)[0] + "..."
            lines.append(f"{index}. [{source.source}, trang {source.page}] {snippet}")
        lines.append("")
        lines.append("Để sinh câu trả lời tự nhiên hơn, hãy thêm OPENAI_API_KEY vào file .env.")
        return "\n".join(lines)
