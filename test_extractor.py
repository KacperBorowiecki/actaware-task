"""Tests for the CO2 emissions extractor.

Three layers:
- TestUnitConversion / TestGrounding / TestParseSnippets / TestResume:
  deterministic, no LLM.
- TestE2E: end-to-end against expected_output.json (calls the LLM once per session).
"""
import json
from pathlib import Path

import pytest

import extractor
from extractor import (
    RawEntry,
    _normalize_unit,
    convert_to_metric_tons,
    is_grounded,
    is_plausible_year,
    parse_snippets,
    process_all,
    process_snippet,
)
from run import (
    PENDING,
    finalize_state,
    load_or_init_state,
    write_state_atomic,
)


HERE = Path(__file__).parent


class TestUnitConversion:
    def test_metric_tons_passthrough(self):
        assert convert_to_metric_tons(100.0, "metric tons") == 100.0

    def test_tonnes_passthrough(self):
        assert convert_to_metric_tons(100.0, "tonnes") == 100.0

    def test_tons_passthrough(self):
        assert convert_to_metric_tons(100.0, "tons") == 100.0

    def test_french_tonnes_metriques(self):
        assert convert_to_metric_tons(100.0, "tonnes métriques") == 100.0

    def test_kilotons_multiplied(self):
        assert convert_to_metric_tons(4.8, "kilotons") == 4800.0

    def test_tco2e_treated_as_metric_tons(self):
        assert convert_to_metric_tons(2000.0, "tCO2e") == 2000.0

    def test_megatons(self):
        assert convert_to_metric_tons(2.0, "megatons") == 2_000_000.0

    def test_unknown_unit_returns_none(self):
        assert convert_to_metric_tons(100.0, "lbs") is None

    def test_case_insensitive(self):
        assert convert_to_metric_tons(1.0, "TONNES") == 1.0

    def test_strips_of_co2_suffix(self):
        assert convert_to_metric_tons(8000.0, "tons of CO2") == 8000.0

    def test_strips_of_co2e_suffix(self):
        assert convert_to_metric_tons(8000.0, "metric tons of CO2e") == 8000.0

    def test_empty_unit_returns_none(self):
        assert convert_to_metric_tons(100.0, "") is None

    def test_numeric_unit_returns_none(self):
        assert convert_to_metric_tons(100.0, "12345") is None


class TestNormalizeUnit:
    def test_lowercase_and_collapse_whitespace(self):
        assert _normalize_unit("  Metric  Tons  ") == "metric tons"

    def test_strips_diacritics(self):
        assert _normalize_unit("tonnes métriques") == "tonnes metriques"

    def test_strips_punctuation(self):
        assert _normalize_unit("metric-tons") == "metric tons"

    def test_strips_trailing_co2(self):
        assert _normalize_unit("tons of CO2") == "tons"

    def test_strips_trailing_co2e(self):
        assert _normalize_unit("metric tons CO2e") == "metric tons"

    def test_does_not_strip_inline_co2(self):
        # tCO2e is itself the unit; the regex requires whitespace before co2/ghg
        assert _normalize_unit("tCO2e") == "tco2e"

    def test_empty_string(self):
        assert _normalize_unit("") == ""


class TestPlausibleYear:
    def test_recent_year(self):
        assert is_plausible_year(2024) is True

    def test_year_too_far_past(self):
        assert is_plausible_year(1899) is False

    def test_year_far_future_likely_value_confusion(self):
        assert is_plausible_year(12500) is False


class TestGrounding:
    def test_evidence_present_in_text(self):
        text = "Total emissions: 5,000 tons in 2023."
        assert is_grounded("5,000 tons", text) is True

    def test_evidence_absent_from_text(self):
        text = "Total emissions: 5,000 tons in 2023."
        assert is_grounded("10,000 tons", text) is False

    def test_evidence_whitespace_tolerant(self):
        text = "Total emissions:  5,000  tons in 2023."
        assert is_grounded("5,000 tons", text) is True

    def test_evidence_case_insensitive(self):
        text = "Total CO2 Emissions: 12,500 metric tons."
        assert is_grounded("total co2 emissions: 12,500 METRIC TONS", text) is True


class TestProcessSnippetMocked:
    """Exercise the full per-snippet pipeline without hitting the LLM."""

    def _patch_extract(self, monkeypatch, fake_entries: list[RawEntry]) -> None:
        monkeypatch.setattr(
            extractor,
            "extract_with_llm",
            lambda text, client, snippet_id="?": fake_entries,
        )

    def test_drops_ungrounded_keeps_valid(self, monkeypatch):
        self._patch_extract(monkeypatch, [
            RawEntry(value=999.0, unit="tons", year=2023, evidence="not present in text"),
            RawEntry(value=12500.0, unit="metric tons", year=2024, evidence="12,500 metric tons"),
        ])
        text = "Total: 12,500 metric tons in 2024."
        assert process_snippet(text, client=None, snippet_id="test") == [
            {"value": 12500.0, "year": 2024},
        ]

    def test_drops_unknown_unit(self, monkeypatch):
        self._patch_extract(monkeypatch, [
            RawEntry(value=8000.0, unit="lbs", year=2023, evidence="8000 lbs"),
        ])
        assert process_snippet("8000 lbs", client=None, snippet_id="test") == []

    def test_drops_implausible_year(self, monkeypatch):
        self._patch_extract(monkeypatch, [
            RawEntry(value=2024.0, unit="tons", year=12500, evidence="12500 tons"),
        ])
        assert process_snippet("12500 tons", client=None, snippet_id="test") == []

    def test_converts_kilotons(self, monkeypatch):
        self._patch_extract(monkeypatch, [
            RawEntry(value=4.8, unit="kilotons", year=2024, evidence="4.8 kilotons"),
        ])
        assert process_snippet("4.8 kilotons", client=None, snippet_id="test") == [
            {"value": 4800.0, "year": 2024},
        ]


class TestParseSnippets:
    def test_splits_on_delimiter(self):
        text = (
            "=== snippet_1 ===\n"
            "Hello world\n"
            "\n"
            "=== snippet_2 ===\n"
            "Foo bar\n"
        )
        result = parse_snippets(text)
        assert set(result.keys()) == {"snippet_1", "snippet_2"}
        assert "Hello world" in result["snippet_1"]
        assert "Foo bar" in result["snippet_2"]

    def test_parses_real_file(self):
        text = (HERE / "snippets.txt").read_text(encoding="utf-8")
        result = parse_snippets(text)
        assert set(result.keys()) == {f"snippet_{i}" for i in range(1, 8)}


class TestResume:
    def test_init_fresh_when_no_file(self, tmp_path):
        out = tmp_path / "output.json"
        state = load_or_init_state(out, ["snippet_1", "snippet_2"], resume=False)
        assert state == {"snippet_1": PENDING, "snippet_2": PENDING}

    def test_init_fresh_ignores_existing_file(self, tmp_path):
        out = tmp_path / "output.json"
        out.write_text('{"snippet_1": [{"value": 1.0, "year": 2024}]}', encoding="utf-8")
        state = load_or_init_state(out, ["snippet_1", "snippet_2"], resume=False)
        assert state == {"snippet_1": PENDING, "snippet_2": PENDING}

    def test_resume_loads_done_and_marks_missing_pending(self, tmp_path):
        out = tmp_path / "output.json"
        out.write_text(
            '{"snippet_1": [{"value": 100.0, "year": 2024}], "snippet_3": []}',
            encoding="utf-8",
        )
        state = load_or_init_state(
            out, ["snippet_1", "snippet_2", "snippet_3"], resume=True,
        )
        assert state["snippet_1"] == [{"value": 100.0, "year": 2024}]
        assert state["snippet_2"] is PENDING
        assert state["snippet_3"] == []

    def test_resume_with_no_existing_file_starts_fresh(self, tmp_path):
        out = tmp_path / "output.json"
        state = load_or_init_state(out, ["snippet_1"], resume=True)
        assert state == {"snippet_1": PENDING}

    def test_atomic_write_round_trip(self, tmp_path):
        out = tmp_path / "output.json"
        write_state_atomic(out, {"snippet_1": [{"value": 1.0, "year": 2024}]})
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == {"snippet_1": [{"value": 1.0, "year": 2024}]}
        assert not (tmp_path / "output.json.tmp").exists()

    def test_finalize_replaces_pending_with_empty_list(self):
        state = {"snippet_1": [{"value": 1.0, "year": 2024}], "snippet_2": PENDING}
        assert finalize_state(state) == {
            "snippet_1": [{"value": 1.0, "year": 2024}],
            "snippet_2": [],
        }


@pytest.fixture(scope="session")
def expected_output():
    return json.loads((HERE / "expected_output.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def actual_output():
    """Run the full pipeline once for the whole session (LLM is the expensive part)."""
    text = (HERE / "snippets.txt").read_text(encoding="utf-8")
    snippets = parse_snippets(text)
    return process_all(snippets)


class TestE2E:
    @pytest.mark.parametrize("snippet_id", [f"snippet_{i}" for i in range(1, 8)])
    def test_matches_expected(self, actual_output, expected_output, snippet_id):
        assert actual_output[snippet_id] == expected_output[snippet_id], (
            f"\n{snippet_id} mismatch:\n"
            f"  actual:   {actual_output[snippet_id]}\n"
            f"  expected: {expected_output[snippet_id]}"
        )
