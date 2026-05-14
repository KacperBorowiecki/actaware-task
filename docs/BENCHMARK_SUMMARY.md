# Ollama throughput tuning — co naprawdę działa na 1 GPU

Krótkie podsumowanie z eksperymentów które robiłem na pojedynczej karcie RTX 5090 (32 GB)
z modelem `gemma4:e2b` (Q4_K_M, ~2-3 GB wagi). Workload: ekstrakcja danych z krótkich snippetów
(prompt ~900 tokenów input, ~90 tokenów output JSON), structured output przez Pydantic schema.

## Co testowałem

4 strategie obsługi requestów + skalowanie `OLLAMA_NUM_PARALLEL`:

| Strategia | Co to | Po co testować |
|---|---|---|
| **A1 single sequential** | 1 instancja, 1 request na raz | baseline latency |
| **A2 single + batching** | 1 instancja, klient wysyła wiele równolegle | server-side continuous batching |
| **A3 dual sequential** | 2 instancje Ollamy na różnych portach, każda sequential | czy dwa procesy "wyssą" lepiej GPU |
| **A4 dual + batching** | 2 instancje × batching | maksymalny throughput |

`NUM_PARALLEL` testowany: 4, 16, 24, 64.

## Wyniki

100 requestów per konfiguracja, ten sam prompt, te same snippety, RTX 5090, single GPU (`CUDA_VISIBLE_DEVICES=0`).

| Strategia | NP | Throughput | p50 latency | p99 | VRAM peak | Czas dla 1M req |
|---|---|---|---|---|---|---|
| A1 sequential | – | 0.367 req/s | 2.7 s | 3.5 s | 17.5 GB | 32 dni |
| A2 single | 4 | 0.460 | 16.7 s | 24.4 s | 13.5 GB | 25 dni |
| A2 single | 16 | 0.943 | 29.3 s | 45.9 s | 17.5 GB | 12 dni |
| **A2 single** | **24** | **1.112** | **34.2 s** | **57.3 s** | **20.5 GB** | **10.4 dni** ⭐ |
| A2 single | 64 | 1.135 | 57.8 s | 79.9 s | 18.0 GB¹ | 10.2 dni |
| A3 dual | 4 | 0.441 | 4.6 s | 5.9 s | 22.3 GB | 26 dni |
| A4 dual | 16 | 1.105 | 41.4 s | 68.4 s | 30.7 GB | 10.5 dni |

¹ Dla NP=64 musiałem ustawić `OLLAMA_CONTEXT_LENGTH=4096` żeby uniknąć crashu w llama.cpp (osobny bug, INT_MAX assert na dużych KV cache tensorach).

## 5 rzeczy które mnie zaskoczyły

### 1. "Dwa procesy na 1 GPU" to mit przy małym modelu

Na początku miałem hipotezę że dwa procesy Ollamy na jednym GPU dadzą lepsze throughput niż jeden z większym batchem (osobne CUDA contexts, lepsze wysycenie kerneli). Dane to obalają:

- A4 dual NP=16: **1.105 req/s, p50 41s**
- A2 single NP=24: **1.112 req/s, p50 34s**

Single instance z większym batchem **wygrywa o włos throughputem ALE ma 17% lepszą latency** przy połowie infrastruktury. Cały plan z dwoma instancjami okazał się niepotrzebny.

### 2. Throughput plateau przy NP=16-24

Per-slot efficiency:
- NP 4→16: 40 mreq/s na slot
- NP 16→24: 21 mreq/s na slot (×½)
- NP 24→64: 0.58 mreq/s na slot (×36 mniej)

Powyżej NP=24 GPU dla tego modelu jest po prostu wysycony — dorzucanie slotów dodaje już tylko queue depth, nie compute.

### 3. Latency rośnie liniowo z batch size

Continuous batching = klasyczny trade-off:
- NP=4: każdy request 16.7s
- NP=16: 29.3s (1.75×)
- NP=24: 34.2s (2.05×)
- NP=64: 57.8s (3.5×)

Per-token forward pass GPU = identyczny 300 tok/s wszędzie. Ale każdy slot dzieli GPU, więc wall-time per request rośnie. Dla interaktywnych zastosowań **batching to pułapka**, dla offline batch jobs — niezbędny.

### 4. Ollama auto-cachuje prefix między requestami

Testowałem prompt 2× pod rząd przez `ollama run --verbose`:

```
Run 1: prompt_eval 45ms (19k tok/s) — cold
Run 2: prompt_eval 5ms (168k tok/s) — cache hit
```

W daemon `ollama serve` jest persistent KV cache między requestami. Identyczny prefix = skip prompt eval. Działa cross-invocation, bez `cache_id` API jak w Gemini. To znacznie zmniejsza koszt długich system promptów.

### 5. NUM_PARALLEL kosztuje VRAM, ale context_length kosztuje **dużo bardziej**

VRAM dla różnych konfiguracji:
- NP=4, ctx=32K default: 13.5 GB
- NP=16, ctx=32K: 17.5 GB (+340 MB/slot)
- NP=24, ctx=32K: 20.5 GB
- NP=64, ctx=**4K**: **18.0 GB** ← mniej niż NP=24

64 sloty z krótkim kontekstem mieszczą się w mniej VRAM niż 24 sloty z domyślnym (32K). Bo VRAM dla KV cache to `NP × ctx × per_token_size`. Domyślny ctx Ollamy auto-skaluje się do wolnego VRAM (np. 32K dla mojej karty), co marnotrawi pamięć dla większości realnych workloadów.

**Praktyczna rada:** ustaw `OLLAMA_CONTEXT_LENGTH` na sensowną wartość (2× max długość twojego promptu) zamiast pozwalać Ollamie wybierać.

## Werdykt

Dla mojego workloadu (krótkie snippety, structured JSON output, 1 GPU):

| Cel | Wybór | Why |
|---|---|---|
| Interactive UI (user czeka) | A1 sequential | 2.7s p50, deterministic |
| Mieszany (decent throughput + OK latency) | A3 dual sequential | 4.6s p50, 1.21× throughput |
| Offline batch (1M+ snippetów) | **A2 NP=24** | 1.11 req/s, ~10 dni dla 1M, jedna instancja |

Lokalnie 1M snippetów to **~10 dni** + ~$45 elektryczność. Gemini 3.1 Flash Lite + prompt cache: **~4 godziny, $183**. Total cost of ownership Gemini wygrywa dramatycznie jeśli ten batch zrobiłbyś więcej niż raz.

## Bonus: Docker porównanie

Sprawdziłem czy te same wyniki da się odtworzyć w Dockerze (Ollama w 2 kontenerach na tej samej karcie, benchmark z hosta). Setup w [DOCKER.md](DOCKER.md), compose w [docker-compose.yml](../docker-compose.yml).

Apples-to-apples na **NP=16, PW=32**:

| Approach | Native | Docker | Diff |
|---|---|---|---|
| A1 sequential | 0.367 req/s | 0.355 | -3% |
| A2 single+batching | 0.943 | 0.828 | **-12%** |
| A3 dual sequential | 0.440 | 0.419 | -5% |
| A4 dual+batching | 1.105 | 1.052 | -5% |

Correctness: 100% w obu.

**Docker tax: ~3-12%** (mediana ~5%) dla Ollamy z GPU passthrough na Windows Docker Desktop (WSL2). Konsystentne z prognozą "2-5% overhead WSL2 + lekka warstwa networkowa".

Inne obserwacje z dockerowego runu:
- **VRAM niższy o 5-10 GB** vs native — bo brakuje Ollama Desktop tray app i innych Windows GPU consumerów w baseline'ie
- **Dwie instancje na jednym GPU** dzielą zasoby tak samo jak natywne procesy (zweryfikowane przez nvidia-smi: GPU 0 zajęty, GPU 1 idle, mimo że karta ma 2 GPU dostępne)
- Wszystkie wnioski (sweet spot przy NP=24, plateau, ortogonalność JSON schema) **zachowują walor** dla setupów dockerowych

## Setup do reprodukcji

### Native

```powershell
# Terminal 1 — Ollama
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_PARALLEL = "24"
$env:OLLAMA_CONTEXT_LENGTH = "4096"   # zamiast default 32768
$env:OLLAMA_MAX_LOADED_MODELS = "1"
ollama serve

# Terminal 2 — benchmark
python benchmark_parallel.py --approaches 2 --parallel-workers 48
```

### Docker

```powershell
docker compose --profile single up -d
python benchmark_parallel.py --approaches 2 --parallel-workers 48
docker compose --profile single down
```

Dla porównania z dual: `--profile dual` i `python benchmark_parallel.py` (wszystkie 4 approaches).

Pełne dane + per-request metrics + analiza jest w [OLLAMA_CONCURRENCY_NOTES.md](OLLAMA_CONCURRENCY_NOTES.md).
Bug llama.cpp z NP=32 + default ctx: [LLAMACPP_INT_MAX_BUG.md](LLAMACPP_INT_MAX_BUG.md).
