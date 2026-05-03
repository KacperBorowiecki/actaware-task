# Writeup

## Approach

TDD-lite: golden `expected_output.json` locked before implementation, then 7 E2E tests plus deterministic unit tests (conversion, grounding, parsing). Pipeline per snippet:

```
LLM (Gemini, structured output) -> grounding check -> unit whitelist -> arithmetic conversion
```

LLM handles what's hard for code: language/context (FR), US vs EU number formats, snippet-local overrides of "annual". Code owns what must be deterministic: unit math, evidence validation, serialization. Structured output via Pydantic `response_schema` — no string parsing. The prompt uses GHG Protocol terminology (Scope 1/2/3, Total) — industry standard, not test-set tuning. **Grounding**: the LLM must return an `evidence` quote; code rejects any entry whose evidence is not literally present in the snippet (whitespace-tolerant).

Everything coded by Claude Code with Opus 4.7

I didn't consider any other approaches.

## Assumptions

- `value` is always `float` (snippet 4 forces `1234.56`, so int/float mixing would be inconsistent).
- `tCO2e` ≡ metric tons (ESG industry consensus). **Breaks for** methane- or N2O-heavy industries (agriculture, livestock, mining): tCO2e includes other GHGs converted to CO2-equivalent, so the *pure*-CO2 share is substantially lower than the reported number.
- "Annual CO2 emissions" = a single year's value. Multi-year tables produce one entry per year; we do **not** sum across years. **Breaks on** multi-year aggregates ("1.5 Mt over 2020-2024") — currently dropped, no year to attach.
- Output schema has no `unit` field, so "expressed in metric tons" is the output spec, not an input filter — kilotons and tCO2e are converted.
- `Mt` interpreted as megatons (SI convention). **Breaks if** a report uses "MT" colloquially to mean "Metric Tons". A sanity-check on suspiciously large converted values would catch this in production.
- If there is no mention which Scope is equal to annual — we return all of them

## Edge cases

**Handled:** 
- US/EU number formats (snippet 4) delegated to LLM; 
- conversion of known units (kilotons → tons)
- multi-year tables (snippet 7) emit one entry per year
- snippet-local "annual" overrides respected (snippet 6: note defines "annual = Scope 1")
- marketing-only text → `[]` (snippet 2)
- implicit values skipped (snippet 5: "15% reduction in 2024" — extract, not calculate).
- Unknown units (lbs, short tons): `convert_to_metric_tons` returns `None`, entry dropped + WARNING — better to drop than introduce garbage. It didn't appear in the dataset but it so obvious that it's needed.

## Scaling to 100k documents

I'd use async/concurrent requests to speed up processing and test cheaper or local LLMs to reduce cost.

**What breaks first:**
1. **Sequential latency** — running snippets one-by-one means hours/days for 100k. Switch to concurrent requests.
2. **Rate limits + cost** — Gemini API has request-per-minute caps, and per-call cost adds up. Add backoff/retry; consider batch mode or local LLMs.
3. **Quality validation** — manual coverage at n=7 doesn't scale. Add another LLM as a judge, extend the golden dataset to ~1000 entries, use confidence thresholds to flag low-certainty cases for human review.
4. **Storage** — currently we rewrite the entire output file after every snippet (O(n²) total IO). At 1M+ documents, switch to append-only or per-shard files, SQLite, or parquet.
5. **Data variations not in this dataset:**
   - Scope 1/2/3 reported without a Total or override → currently emits 3 entries per year. Production fix: hierarchy `Total explicit → Total; elif S1+S2+S3 → sum; else → emit per scope` (requires `scope` field in schema).
   - Multi-year aggregates ("1.5 Mt over 2020-2024") — no single year to attach.
   - Negative emissions / carbon offsets / sequestration — offsets ≠ emissions, would need separate handling.

## Time spent

~73 min for code. ~40 min for WRITEUP.md. With another hour: extend the test dataset + test on small local llms - to find something fast and cheap. Or try to understand the ESG market better
