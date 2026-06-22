from src.evaluation.rag_evaluator import run_evaluation


if __name__ == "__main__":
    results = run_evaluation()
    passed = sum(1 for item in results if item["passed"])
    print(f"Passed {passed}/{len(results)}")
