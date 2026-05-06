# Actaware CO2 Extraction Task

Extracts annual CO2 emissions (in metric tons) from corporate sustainability snippets
using an LLM with structured output, evidence grounding, and a deterministic
unit-conversion layer. Two backends are supported out of the box: **Gemini**
(cloud) and **Ollama** (local).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux/macOS
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set GEMINI_API_KEY (for Gemini) and/or OLLAMA_MODEL (for Ollama).
```

## Choosing the LLM backend

Set `LLM_CLIENT` in `.env` (or as a shell env var):

```bash
LLM_CLIENT=gemini    # default; uses GEMINI_API_KEY + GEMINI_MODEL
LLM_CLIENT=ollama    # uses local Ollama at OLLAMA_HOST with OLLAMA_MODEL
```

For Ollama: install Ollama locally (https://ollama.com/), then pull the model:

```bash
ollama pull gemma4:e2b
```

Each client has an `ALLOWED_MODELS` whitelist — passing a model outside it
raises `ValueError` with the allowed list. Extend the set in
`extractor.py` (`GeminiClient.ALLOWED_MODELS` / `OllamaClient.ALLOWED_MODELS`)
when you verify a new model works. Set to `None` to accept any model.

Add a new backend by implementing the `LLMClient` protocol in `extractor.py`
and adding a branch to `get_llm_client()`.

## Run

```bash
python run.py                                         # fresh run -> output.json (uses LLM_CLIENT from env)
python run.py --resume                                # skip snippets already processed
python run.py --client gemini                         # override env, use Gemini
python run.py --client ollama --model gemma3:12b      # override both backend and model
python run.py --verbose                               # DEBUG logging
python run.py --help                                  # all flags
```

`--client` and `--model` override the corresponding env vars (`LLM_CLIENT`,
`GEMINI_MODEL` / `OLLAMA_MODEL`) — handy for quick model comparisons without
editing `.env`. Model must be in the chosen client's `ALLOWED_MODELS`.

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
