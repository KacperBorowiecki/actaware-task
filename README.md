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

## Benchmarks & Docker

Beyond the core extraction task, this repo includes a benchmarking suite that
explores Ollama concurrency strategies, GPU sharing, mixed embedder+LLM
workloads, and Docker deployment. Full writeups in [docs/](docs/).

Quick commands:

```bash
python benchmark.py                          # per-model speed comparison (sequential)
python benchmark_parallel.py                 # 4 concurrency strategies × parallelism levels
python benchmark_mixed.py                    # mixed workload: embeddings + generation concurrent
python verbose_run.py test_prompt.txt        # ollama-run-style verbose timing of a single prompt
```

Docker deployment (Ollama runs in container, benchmark from host):

```bash
docker compose --profile single up -d        # 1 Ollama instance on :11434
docker compose --profile dual up -d          # 2 instances on :11434 + :11435 (shared GPU)
docker compose down
```

Analysis writeups:
- [docs/BENCHMARK_SUMMARY.md](docs/BENCHMARK_SUMMARY.md) — 1-pager for sharing, 5 key findings
- [docs/OLLAMA_CONCURRENCY_NOTES.md](docs/OLLAMA_CONCURRENCY_NOTES.md) — full log of NP=4/16/24/32/64 rounds
- [docs/DOCKER.md](docs/DOCKER.md) — Docker quick-start
- [docs/LLAMACPP_INT_MAX_BUG.md](docs/LLAMACPP_INT_MAX_BUG.md) — upstream bug repro at high NP × ctx
- [docs/HF_BF16_NOTES.md](docs/HF_BF16_NOTES.md) — HuggingFace bf16 backend notes

Raw per-request metrics and aggregate summaries live in [results/](results/).

## Files

| File | Role |
|---|---|
| `extractor.py` | LLM call, grounding check, unit normalization & conversion |
| `run.py` | CLI entry point, state file, resume logic, logging |
| `test_extractor.py` | pytest: unit + grounding + parse + resume + 7 E2E |
| `expected_output.json` | golden dataset (ground truth for E2E tests) |
| `output.json` | extraction result |
| `WRITEUP.md` | approach, assumptions, edge cases, scaling discussion |
| `benchmark.py` | per-model speed comparison (Gemini + Ollama, sequential) |
| `benchmark_parallel.py` | 4-approach concurrency benchmark (single/dual instance × seq/parallel) |
| `benchmark_mixed.py` | mixed workload — embedder + LLM co-existence on one Ollama |
| `verbose_run.py` | helper mimicking `ollama run --verbose` (incl. JSON-schema mode) |
| `docker-compose.yml` | Ollama orchestration, profiles: `single` / `dual` |
| `docs/` | all analysis writeups (benchmark summary, concurrency notes, bugs, Docker) |
| `results/` | per-request JSONL + aggregate summary JSON from every benchmark run |
