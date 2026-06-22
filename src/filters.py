import re


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_useful_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= 40
