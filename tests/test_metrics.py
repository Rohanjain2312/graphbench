"""Tests for graphbench.benchmark.metrics.

These tests are fully implemented in Phase 1 since metrics.py is complete.
All other test files are stubs pending their respective phase implementations.
"""

from graphbench.benchmark.metrics import exact_match, normalize_answer, token_f1


class TestNormalizeAnswer:
    """Tests for the canonical normalize_answer() function."""

    def test_lowercases(self) -> None:
        assert normalize_answer("Albert Einstein") == "albert einstein"

    def test_removes_punctuation(self) -> None:
        assert normalize_answer("hello, world!") == "hello world"

    def test_removes_articles(self) -> None:
        assert normalize_answer("the United States") == "united states"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_empty_string(self) -> None:
        assert normalize_answer("") == ""

    def test_combined(self) -> None:
        assert normalize_answer("The  Quick Brown Fox.") == "quick brown fox"


class TestExactMatch:
    """Tests for exact_match()."""

    def test_perfect_match(self) -> None:
        assert exact_match("Ulm", "Ulm") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match("ulm", "Ulm") == 1.0

    def test_mismatch(self) -> None:
        assert exact_match("Berlin", "Ulm") == 0.0

    def test_article_ignored(self) -> None:
        assert exact_match("the United States", "United States") == 1.0


class TestTokenF1:
    """Tests for token_f1()."""

    def test_perfect_match(self) -> None:
        assert token_f1("Albert Einstein", "Albert Einstein") == 1.0

    def test_partial_overlap(self) -> None:
        score = token_f1("Albert Einstein physicist", "Albert Einstein")
        assert 0.0 < score < 1.0

    def test_no_overlap(self) -> None:
        assert token_f1("Berlin", "Paris") == 0.0

    def test_both_empty(self) -> None:
        assert token_f1("", "") == 1.0

    def test_one_empty(self) -> None:
        assert token_f1("something", "") == 0.0
        assert token_f1("", "something") == 0.0
