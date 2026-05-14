# Uruchomienie benchmarka przez Docker

Zdockeryzowana jest **tylko Ollama**. Benchmark (`benchmark_parallel.py`) działa z hosta i hituje kontenery przez HTTP.

## Wymagania

- Docker Desktop ≥28 (Windows/Mac) lub Docker Engine + nvidia-container-toolkit (Linux)
- NVIDIA GPU + sterowniki
- Python venv na hoscie z `pip install -r requirements.txt`

## Test 1: GPU passthrough działa

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Musi pokazać twoją kartę. Jeśli błąd — `nvidia-container-toolkit` nie jest aktywny.

## Konfiguracja

W `.env` (na hoscie, ten sam co dla benchmarka):

```
OLLAMA_NUM_PARALLEL=24
OLLAMA_CONTEXT_LENGTH=4096
OLLAMA_MODEL=gemma4:e2b
```

Defaulty są w `docker-compose.yml`, ale `.env` przesłania.

## Profile single (rekomendowany)

```powershell
docker compose --profile single up -d
docker compose logs -f model-puller    # poczekaj aż pull się skończy
docker compose ps                      # ollama-1: healthy

# Smoke test:
curl http://localhost:11434/api/version

# Benchmark z hosta:
.\.venv\Scripts\python.exe benchmark_parallel.py --approaches 2 --requests 30
```

## Profile dual (dla porównań app3/app4)

Dla dual instance default zostaje `NUM_PARALLEL=16` żeby zmieścić 2 instancje w 32 GB VRAM. Możesz override w `.env`.

```powershell
docker compose --profile dual up -d
docker compose ps                      # ollama-1: healthy, ollama-2: healthy

# Weryfikuj że oba widzą model:
docker compose exec ollama-1 ollama list
docker compose exec ollama-2 ollama list

# Weryfikuj że oba siedzą na GPU 0 (nie po jednym na każdej karcie):
nvidia-smi
# W sekcji Processes powinieneś zobaczyć DWA procesy ollama_llama_server na GPU 0

# Benchmark wszystkich 4 approaches:
.\.venv\Scripts\python.exe benchmark_parallel.py
```

## Zatrzymanie

```powershell
docker compose --profile single down   # albo --profile dual
```

Volume `ollama-models` zostaje (modele nie znikają). Żeby wyczyścić też modele:

```powershell
docker compose down -v
```

## Troubleshooting

**`Error: port is already in use`** — masz natywną Ollamę na hoscie. Zatrzymaj ją:
```powershell
# Znajdź proces:
netstat -ano | findstr ":11434"
# PID z ostatniej kolumny, np. 15996:
taskkill /PID 15996 /F
# Lub przez Windows: zamknij "Ollama" w tray, albo:
Get-Process ollama* | Stop-Process -Force
```

**OOM podczas `app3`/`app4`** — VRAM ciasno. Zmień w `.env`:
```
OLLAMA_NUM_PARALLEL=12
# albo:
OLLAMA_CONTEXT_LENGTH=2048
```
Potem `docker compose --profile dual restart`.

**`No GPU found` w kontenerze** — sprawdź czy Docker Desktop ma włączone WSL2 backend i NVIDIA GPU support. Test 1 powyżej musi przejść jako pierwszy.

**model-puller zawiesza się** — pewnie pierwszy pull jest długi (gemma4:e2b to ~2-3 GB). `docker compose logs -f model-puller` pokaże progress.

## Spodziewane wyniki (na RTX 5090)

Z `OLLAMA_NUM_PARALLEL=24`, ctx=4096, single profile:

| Approach | Throughput | p50 |
|---|---|---|
| app1 sequential | ~0.36 req/s | ~2.7 s |
| **app2 NP=24** | **~1.1 req/s** | ~34 s |

Z dual (NP=16 default w compose):

| Approach | Throughput | p50 |
|---|---|---|
| app3 dual seq | ~0.44 req/s | ~4.6 s |
| app4 dual + batch | ~1.1 req/s | ~41 s |

Pełny opis eksperymentów w [BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md) i [OLLAMA_CONCURRENCY_NOTES.md](OLLAMA_CONCURRENCY_NOTES.md).
