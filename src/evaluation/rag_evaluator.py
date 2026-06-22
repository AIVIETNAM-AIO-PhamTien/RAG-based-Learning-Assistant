import csv
from pathlib import Path

from src.learning import answer_question


def run_evaluation(csv_path: Path = Path("src/evaluation/benchmark_rag.csv")) -> list[dict[str, str | bool]]:
    results: list[dict[str, str | bool]] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            response = answer_question(row["question"])
            keyword = row["expected_keyword"].lower()
            passed = keyword in response.answer.lower()
            results.append({"question": row["question"], "expected_keyword": keyword, "passed": passed})
    return results


if __name__ == "__main__":
    for item in run_evaluation():
        print(item)
