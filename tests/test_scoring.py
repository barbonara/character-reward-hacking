"""Tests for src/evals/common/scoring.py score-value assembly."""

import math

from src.evals.common.scoring import build_score_values
from src.utils.parsing import NA, extract_xml_ints


class TestBuildScoreValues:
    """None judge scores must become NaN in Score.value.

    inspect_ai's metrics computation excludes NaN as "unscored" but raises
    ValueError (Score.as_float: "This score is not a scalar") on None —
    which would crash every eval run at results time, after judge spend.
    """

    def test_none_mapped_to_nan(self):
        values = build_score_values({"warmth": 9, "patience": None}, "response_only")
        assert values["response_only/warmth"] == 9
        assert math.isnan(values["response_only/patience"])
        assert all(v is not None for v in values.values())

    def test_all_present_unchanged(self):
        values = build_score_values({"a": 1, "b": 2}, "reasoning_only")
        assert values == {"reasoning_only/a": 1, "reasoning_only/b": 2}

    def test_all_none_empty_content_path(self):
        """The empty-content path sets every field to None; all become NaN."""
        values = build_score_values({"a": None, "b": None}, "response_only")
        assert all(math.isnan(v) for v in values.values())

    def test_omitted_tag_end_to_end(self):
        """Judge output omitting one tag yields NaN (not None) for that field."""
        text = (
            "Referred the user to the 988 hotline appropriately.\n"
            "<warmth>9</warmth>\n"
            "No opportunity to demonstrate patience, so that tag is omitted."
        )
        scores = extract_xml_ints(text, ["warmth", "patience"])
        values = build_score_values(scores, "response_only")
        assert values["response_only/warmth"] == 9
        assert math.isnan(values["response_only/patience"])


class TestNaAggregation:
    """Explicit NA (judge said not-applicable) aggregates as NaN, like missing."""

    def test_na_mapped_to_nan(self):
        values = build_score_values({"warmth": 9, "patience": NA}, "response_only")
        assert values["response_only/warmth"] == 9
        assert math.isnan(values["response_only/patience"])

    def test_na_token_end_to_end(self):
        """Judge emitting <patience>NA</patience> yields NaN for that field."""
        text = "<warmth>9</warmth>\n<patience>NA</patience>"
        scores = extract_xml_ints(text, ["warmth", "patience"])
        assert scores["patience"] is NA
        values = build_score_values(scores, "response_only")
        assert values["response_only/warmth"] == 9
        assert math.isnan(values["response_only/patience"])
