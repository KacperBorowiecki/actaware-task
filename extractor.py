"""CO2 emissions extractor.

Pipeline per snippet:
    LLM (structured output) -> grounding check -> unit normalization -> conversion
The LLM extracts raw value/unit/year/evidence verbatim from the text;
the code handles grounding, unit normalization, and arithmetic conversion.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()
logger = logging.getLogger(__name__)


MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


# Whitelist of unit -> conversion factor (to metric tons).
# Keys are post-normalization (see _normalize_unit); the LLM returns the unit verbatim
# from the text and we normalize before lookup. Adding a new unit is a one-line change.
UNIT_FACTORS: dict[str, float] = {
    # metric ton family (factor 1.0)
    "metric tons": 1.0,
    "metric ton": 1.0,
    "tons": 1.0,
    "ton": 1.0,
    "tonnes": 1.0,
    "tonne": 1.0,
    "tonnes metriques": 1.0,
    "tonne metrique": 1.0,
    "t": 1.0,
    "tco2e": 1.0,
    "tco2": 1.0,
    # kiloton family (factor 1000)
    "kilotons": 1000.0,
    "kiloton": 1000.0,
    "kt": 1000.0,
    "ktco2e": 1000.0,
    # megaton family (factor 1_000_000)
    "megatons": 1_000_000.0,
    "megaton": 1_000_000.0,
    "mt": 1_000_000.0,
    "mtco2e": 1_000_000.0,
}


class RawEntry(BaseModel):
    """One emissions value as returned by the LLM (pre-validation, pre-conversion)."""
    value: float = Field(description="Numeric value, parsed as a float.")
    unit: str = Field(description="Unit of the value, verbatim as it appears in the text.")
    year: int = Field(description="Reporting year, e.g. 2024.")
    evidence: str = Field(
        description="Verbatim substring from the snippet that contains this value."
    )


class ExtractionResult(BaseModel):
    entries: list[RawEntry]


EXTRACTION_PROMPT = """You are extracting CO2 emissions data from an excerpt of a corporate sustainability report.

For each CO2 emissions value EXPLICITLY stated for a specific reporting year, return:
- value: the number parsed as a float. Handle European format ("1.234,56" -> 1234.56) and US format ("12,500" -> 12500.0).
- unit: the unit of measurement EXACTLY as it appears in the text (e.g. "metric tons", "kilotons", "tonnes metriques", "tCO2e"). Do NOT translate it. Return only the unit itself: strip any trailing qualifier such as "of CO2", "of CO2e", or "of GHG" (e.g. "500 tons of CO2" -> unit is "tons").
- year: the reporting year as an integer.
- evidence: the EXACT verbatim substring from the snippet that contains this value (will be checked for a literal match).

DEFINITIONS (per the GHG Protocol):
- "CO2 emissions" covers Scope 1 (direct), Scope 2 (purchased energy), Scope 3 (value chain), and Total emissions. All are valid values to extract when explicitly stated for a specific year.

STRICT RULES:
1. Extract only values that are EXPLICITLY stated as numbers. Do NOT calculate or infer values from percentages, year-over-year changes, or comparisons (e.g., "reduced by 15% vs 2023" does NOT yield a 2024 value).
2. If the snippet contains a note/footnote/qualifier defining which value should be considered the company's reported annual emissions, follow that definition and return ONLY the matching value.
3. If the same emissions metric is reported for multiple years (e.g. a multi-year table), return one entry per year.
4. Do NOT extract non-CO2 metrics (water, waste, energy share, renewable share, etc.).
5. If no CO2 emissions value is explicitly tied to a year, return an empty array.

Snippet:
---
{snippet_text}
---
"""


def parse_snippets(text: str) -> dict[str, str]:
    """Split snippets.txt into a dict of {snippet_id: body}."""
    pattern = re.compile(r"^===\s*(snippet_\d+)\s*===\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    snippets: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        snippets[m.group(1)] = text[start:end].strip()
    return snippets


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_grounded(evidence: str, snippet_text: str) -> bool:
    """Check that the LLM-provided evidence is actually present in the snippet.

    Whitespace- and case-tolerant: the LLM occasionally adjusts spacing or casing.
    """
    return _normalize_whitespace(evidence).lower() in _normalize_whitespace(snippet_text).lower()


def _normalize_unit(unit: str) -> str:
    """Strip diacritics, punctuation, lowercase, collapse whitespace.

    Also strips trailing qualifiers like "of CO2" / "of CO2e" / "of GHG"
    so "tons of CO2" normalizes to "tons" (safety net for the LLM).
    """
    s = unicodedata.normalize("NFKD", unit).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+(of\s+)?(co2e?|ghg)$", "", s).strip()
    return s


def convert_to_metric_tons(value: float, unit: str) -> float | None:
    """Convert `value` from `unit` to metric tons. Returns None for unknown units."""
    factor = UNIT_FACTORS.get(_normalize_unit(unit))
    if factor is None:
        return None
    return value * factor


def is_plausible_year(year: int) -> bool:
    """Sanity check on extracted year. Catches LLM confusing value with year."""
    return 1900 <= year <= datetime.now().year


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to .env.")
    return genai.Client(api_key=api_key)


def extract_with_llm(
    snippet_text: str,
    client: genai.Client,
    snippet_id: str = "?",
) -> list[RawEntry]:
    """Call Gemini with structured output. Returns raw (unvalidated) entries."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=EXTRACTION_PROMPT.format(snippet_text=snippet_text),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ExtractionResult,
        ),
    )
    parsed: ExtractionResult = response.parsed
    if parsed is None:
        logger.warning(
            "[%s] LLM returned no parseable result (response.parsed is None)",
            snippet_id,
        )
        return []
    return parsed.entries


def process_snippet(
    snippet_text: str,
    client: genai.Client,
    snippet_id: str = "?",
) -> list[dict]:
    """Full pipeline for a single snippet: extract -> ground -> convert -> serialize.

    `snippet_id` is used only for log context.
    """
    raw_entries = extract_with_llm(snippet_text, client, snippet_id=snippet_id)

    final: list[dict] = []
    for entry in raw_entries:
        if not is_grounded(entry.evidence, snippet_text):
            logger.warning(
                "[%s] dropping ungrounded entry: evidence=%r not found in snippet",
                snippet_id, entry.evidence,
            )
            continue
        if not is_plausible_year(entry.year):
            logger.warning(
                "[%s] dropping entry with implausible year: year=%s value=%s evidence=%r",
                snippet_id, entry.year, entry.value, entry.evidence,
            )
            continue
        converted = convert_to_metric_tons(entry.value, entry.unit)
        if converted is None:
            logger.warning(
                "[%s] dropping entry with unknown unit: unit=%r value=%s year=%s evidence=%r",
                snippet_id, entry.unit, entry.value, entry.year, entry.evidence,
            )
            continue
        final.append({"value": converted, "year": entry.year})
    return final


def process_all(snippets: dict[str, str]) -> dict[str, list[dict]]:
    """Process every snippet and return the final {snippet_id: [entries]} map.

    A failure on one snippet (e.g. transient LLM error) is logged and yields `[]`
    for that snippet, so the rest of the batch still completes.
    """
    client = get_client()
    out: dict[str, list[dict]] = {}
    for sid, text in snippets.items():
        try:
            out[sid] = process_snippet(text, client, snippet_id=sid)
        except Exception:
            logger.exception("[%s] extraction failed — returning empty list", sid)
            out[sid] = []
    return out
