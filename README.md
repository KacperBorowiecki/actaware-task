# Actaware CO2 Extraction Task

Extracts annual CO2 emissions (in metric tons) from corporate sustainability snippets
using Gemini with structured output, evidence grounding, and a deterministic unit-conversion layer.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux/macOS
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

## Run

```bash
python run.py                  # fresh run -> output.json
python run.py --resume         # skip snippets already processed in output.json
python run.py --verbose        # DEBUG logging
python run.py --help           # all flags
```

Logs are written to `extractor.log` (and stdout). Each completed snippet is persisted
to `output.json` immediately (atomic write), so a crashed run can resume with `--resume`.

## Tests

```bash
pytest -v                      # full suite (E2E hits the LLM once per session)
pytest -v -k "not TestE2E"     # offline only (no API key required)
```

## Files

| File | Role |
|---|---|
| `extractor.py` | LLM call, grounding check, unit normalization & conversion |
| `run.py` | CLI entry point, state file, resume logic, logging |
| `test_extractor.py` | pytest: unit + grounding + parse + resume + 7 E2E |
| `expected_output.json` | golden dataset (ground truth for E2E tests) |
| `output.json` | extraction result |
| `WRITEUP.md` | approach, assumptions, edge cases, scaling discussion |
