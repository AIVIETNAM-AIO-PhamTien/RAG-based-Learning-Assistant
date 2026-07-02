import pytest

from app.rag.metrics import citation_coverage


def test_citation_coverage_single_brackets():
    assert citation_coverage("Answer [1] and [2].", {1, 2, 3}) == pytest.approx(1.0)


def test_citation_coverage_combined_bracket():
    # Model sometimes writes [1, 2] instead of [1][2] despite the prompt
    # asking for separate brackets — the parser must still count both.
    assert citation_coverage("Answer [1, 2].", {1, 2, 3}) == pytest.approx(1.0)


def test_citation_coverage_combined_bracket_no_spaces():
    assert citation_coverage("Answer [1,2,3].", {1, 2, 3}) == pytest.approx(1.0)


def test_citation_coverage_mixed_single_and_combined():
    assert citation_coverage("Answer [1] and also [2, 3].", {1, 2, 3}) == pytest.approx(1.0)


def test_citation_coverage_partial_validity():
    # 2 used, only 1 is a valid index -> 1/2.
    assert citation_coverage("Answer [1, 5].", {1, 2, 3}) == pytest.approx(0.5)


def test_citation_coverage_no_citations():
    assert citation_coverage("Answer with no citations.", {1, 2, 3}) == 0.0
