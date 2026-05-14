# Ollama Concurrency Benchmark — Notatki

Notatki z eksperymentów nad uruchamianiem wielu requestów do Ollamy na jednym GPU
dla workloadu ekstrakcji CO2 (krótkie snippety + structured output JSON).

## Kontekst

**Pytanie wyjściowe:** Jak najlepiej uruchomić dwa modele/instancje LLM na jednej
karcie graficznej żeby współdzieliły zasoby i maksymalizowały throughput?

**Workload:** Ekstrakcja danych z [snippets.txt](../snippets.txt) (9 unikalnych, cyklowane do 100).
Model `gemma4:e2b`, structured output przez Pydantic schema, deterministyczny prompt.

**Hardware:** RTX 5090 (32 GB VRAM), AMD Ryzen 9 9950X (16C/32T), 62 GB RAM.

**Narzędzia:** [benchmark_parallel.py](../benchmark_parallel.py) (nowy), [extractor.py](../extractor.py) (zrefaktorowane: `OllamaClient` i `get_llm_client` przyjmują `host=` kwarg).

## 4 testowane konfiguracje

| # | ID | Instancje Ollamy | `NUM_PARALLEL` | Client threads | Strategia |
|---|---|---|---|---|---|
| 1 | `approach_1_single_seq` | 1 @ `:11434` | 4 | 1 | request po requestcie |
| 2 | `approach_2_single_parallel` | 1 @ `:11434` | 4-16 | 8-32 | `ThreadPoolExecutor`, wszystko async |
| 3 | `approach_3_dual_seq` | 2 @ `:11434` + `:11435` | 4-16 | 2 (1 per inst.) | każda instancja swoje 50 sekwencyjnie, instancje równolegle |
| 4 | `approach_4_dual_parallel` | 2 @ obu portach | 4-16 | 16-64 | round-robin po hostach, wszystko async |

Każdy run: 100 requestów per approach, NVML VRAM sampling 250 ms, correctness check vs [expected_output.json](../expected_output.json).

## Wyniki

### Round 1: pomyłka z 2 GPU (2026-05-14 ~06:22 UTC)

Pierwszy run szedł na dwóch GPU bo Ollama domyślnie widzi wszystkie karty (`CUDA_VISIBLE_DEVICES` nieustawione). Każda instancja chwyciła własną kartę → "dwa procesy na jednym GPU" nie zostało faktycznie przetestowane.

| Approach | Throughput | p50 latency | VRAM peak |
|---|---|---|---|
| app1 | 0.363 req/s | 2687 ms | 18 GB |
| app2 | 0.427 req/s | 18414 ms | 18 GB |
| app3 | **0.665 req/s** | 2874 ms | **30.7 GB** ← suma po 2 GPU |
| app4 | **0.707 req/s** | 10906 ms | **31.4 GB** |

Wnioski **odrzucone**: skok app3/app4 (1.83×/1.95×) to **free 2-GPU effect**, nie zasługa concurrency strategy.

**Fix:** `$env:CUDA_VISIBLE_DEVICES = "0"` przed `ollama serve` w obu terminalach.

### Round 2: 1 GPU, NUM_PARALLEL=4 (2026-05-14 ~06:41 UTC)

Tu zaczęło się robić ciekawie.

| Approach | Throughput | p50 latency | p95 | VRAM baseline |
|---|---|---|---|---|
| app1 | 0.363 req/s | 2681 ms | 3516 ms | 13.4 GB |
| **app2** | **0.460 req/s** | **16731 ms** | **23490 ms** | 13.5 GB |
| **app3** | 0.441 req/s | **4593 ms** | **5647 ms** | 22.3 GB |
| **app4** | **0.510 req/s** | **28986 ms** ⚠️ | **42646 ms** ⚠️ | 22.2 GB |

Correctness: 100% w każdym approach (sanity check).

**Kluczowa obserwacja podczas runu:** `nvidia-smi dmon -s u` pokazał:
- app2 (NP=4, 8 client threads): **SM ~60-65%**, mem ~9%
- app3 (2 procesy sequential): **SM ~98%**, mem ~9%

→ Dwie instancje na jednym GPU **faktycznie wysycają kartę**, podczas gdy single-instance batching z NP=4 zostawia ~35% idle.

### Round 3: 1 GPU, NUM_PARALLEL=16 (2026-05-14 ~06:58 UTC)

Test czy zwiększenie batcha w single instance dorówna throughputowi 2-procesowemu.

| Approach | Throughput | p50 latency | p95 | VRAM peak | Wall (s) |
|---|---|---|---|---|---|
| app1 | 0.367 req/s | 2656 ms | 3500 ms | 17.5 GB | 272.8 |
| **app2 (NP=16)** | **0.943 req/s** | **29265 ms** | **39914 ms** | 17.5 GB | 106.0 |
| app3 (NP=16) | 0.440 req/s | 4562 ms | 5677 ms | 30.6 GB | 227.1 |
| **app4 (NP=16)** | **1.105 req/s** | **41439 ms** | **64358 ms** | 30.7 GB | 90.5 |

Correctness: 100% w każdym approach.

**WOW.** app2 podwoiło throughput (+105%), app4 też (+117%). Prognoza 0.55-0.65 była drastycznie zaniżona — `NUM_PARALLEL=4` zostawiało ogromny zapas na karcie.

**Najważniejsze:** app2 NP=16 (0.94) **prawie dogania** app4 NP=16 (1.10). Różnica 15%. To **odrzuca** wcześniejszą tezę "dwa procesy są fundamentalnie lepsze". Faktyczna teza: **brakowało batcha, nie procesów**.

VRAM:
- app2 NP=16: 17.5 GB (baseline 13.5 GB → +4 GB = 12 dodatkowych slotów × **~340 MB/slot**)
- app3/app4 NP=16: 30.7 GB peak (2 instancje × NP=16). Headroom 1.3 GB na 32 GB. Mieści się, ale ciasno.

### Round 3 — Analiza prompt eval (czy Ollama cachuje prefix?)

**TL;DR: TAK, Ollama auto-cachuje cross-request prefix. Wcześniejsze "podejrzanie szybkie" prompt eval timings to artefakt cache reuse.**

#### Direct test poza benchmarkiem

Użytkownik puścił identyczny prompt 2× przez `ollama run gemma4:e2b --verbose`:

| | Run 1 (zimny) | Run 2 (cache hit) | Stosunek |
|---|---|---|---|
| prompt_eval_count | 877 tokens | 877 tokens | — |
| prompt_eval_duration | **45.4 ms** | **5.2 ms** | **8.7×** |
| prompt_eval_rate | 19,333 tok/s | 167,840 tok/s | 8.7× |
| eval_rate (generation) | 290.79 tok/s | 299.05 tok/s | stabilne |

5.2 ms to overhead (cache lookup + KV prep), nie compute. Wszystkie 877 tokenów reusowane z poprzedniego runu (w obrębie tego samego `ollama serve` daemonem).

#### Jak to się ma do benchmarków

W naszym workloadzie:
- ~800 tokenów statycznych instrukcji (EXTRACTION_PROMPT) — identyczne w każdym requestcie → **cache hit**
- ~100 tokenów zmiennego snippet'u → **fresh compute**

Wyliczenie dla app1 (10 ms prompt eval):
```
100 fresh tokens × 1/19,333 tok/s ≈ 5 ms compute
+ ~5 ms cache lookup overhead
= ~10 ms total                     ✓ idealnie tłumaczy obserwowane
```

Cache jest aktywny w obrębie `keep_alive` (default 5 min od ostatniego użycia). Daemon `ollama serve` trzyma KV cache slotów między oddzielnymi requestami klienta.

#### Werdykt

| Pytanie | Odpowiedź |
|---|---|
| Czy Ollama cachuje prefix automatycznie? | **TAK** |
| Czy trzeba explicit API jak w Gemini? | **NIE**, jest implicit |
| Działa cross-request (różne klienty)? | **TAK**, w obrębie `keep_alive` |
| Działa cross-slot przy continuous batching? | Prawdopodobnie tak, llama.cpp ma fingerprint matching |

→ **Implikacja:** propozycje "variant 1 (dłuższy snippet)" i "variant 2 (dłuższy prefix + cache)" są **już zrealizowane** w naszym workloadzie. Dłuższy prefix nie pomoże bo i tak jest cached.

### Round 3 — JSON Schema: praktycznie darmowa

**Wcześniejsze wersje tej sekcji niepoprawnie twierdziły że schema spowalnia generację 8-9×.** Empiryczny test (`verbose_run.py` na tym samym prompcie, single request, z i bez schema) to obala:

| | Bez schema | Z schema |
|---|---|---|
| eval_rate (forward pass) | 296 tok/s | 297 tok/s |
| total wall time (bez load) | ~2380 ms | ~2350 ms |
| output tokens | 638 | 62 |

**Total wall time per request jest praktycznie identyczny.** Schema nie spowalnia.

#### Mechanizm który się kompensuje

Schema dodaje overhead per token (grammar mask + state update na CPU, ~34 ms/token wg dekompozycji `total - eval - pe - load`). ALE jednocześnie wymusza compact output — model pisze 10× mniej tokenów. W rezultacie:

- **Bez schema:** dużo tokenów × tanio per token (~3.7 ms/tok wall) = ~2.4 s
- **Ze schema:** mało tokenów × drogo per token (~40 ms/tok wall) = ~2.4 s

Te dwa efekty się kasują dla typowych snippetów.

#### Implikacja

Schema jest **praktycznie darmowa** dla naszego workloadu, ALE daje:
- ✅ Structured output bez post-processingu
- ✅ Walidację przez Pydantic za jednym zamachem
- ✅ Determinizm formatu (bez schema model mógłby czasem zwrócić markdown z embedded JSON, czasem czysty JSON, czasem coś dziwnego)

**Wniosek:** zostawiamy `format=ExtractionResult.model_json_schema()`. Nie ma wymiernego speedupu z usunięcia, są wymierne benefity z zostawienia.

#### Co naprawdę determinuje throughput

Patrząc na nasze 4 approaches, jedyny faktyczny driver wall-time throughputu to **batchowanie** (continuous batching server-side). Schema nic do tego nie wnosi:

| | eval_rate per request |
|---|---|
| app1 sequential | 298 tok/s |
| app2 NP=4 (4 in batch) | 98 tok/s |
| app2 NP=16 (16 in batch) | 67 tok/s |
| app4 NP=16 dual | 44 tok/s |

Per-request eval_rate spada bo GPU compute jest dzielony między slotami w batchu. To czysty trade-off **per-request latency** vs **aggregate throughput** (więcej slotów = więcej tokenów na sekundę łącznie, ale każdy slot pracuje wolniej).

### Round 4: 1 GPU, NUM_PARALLEL=24 (2026-05-14 ~07:59 UTC)

Test diminishing returns w skalowaniu NUM_PARALLEL. Tylko app2 (single instance) — `--approaches 2`.

| Approach | Throughput | p50 latency | p95 | p99 | VRAM peak | Wall (s) |
|---|---|---|---|---|---|---|
| **app2 (NP=24)** | **1.112 req/s** | **34.2 s** | **50.4 s** | **57.3 s** | 20.5 GB | 90 |

Correctness: 100%, 48 client threads (`--parallel-workers 48`).

#### Diminishing returns potwierdzone

Per-slot scaling efficiency:

| Skok | Dodane sloty | Przyrost throughput | Efficiency per slot |
|---|---|---|---|
| NP 4 → 16 | +12 | +0.483 req/s | **40 mreq/s/slot** |
| NP 16 → 24 | +8 | +0.169 req/s | **21 mreq/s/slot** (½ poprzedniego) |

Każdy kolejny slot daje połowę mniej. Plateau blisko.

#### app2 NP=24 wyprzedza app4 NP=16 — single instance wygrywa

| Konfiguracja | Throughput | p50 latency | Instancje |
|---|---|---|---|
| **app2 NP=24** | **1.112 req/s** | **34.2 s** | **1** |
| app4 NP=16 (dual) | 1.105 req/s | 41.4 s | 2 |

**Pojedyncza instancja z NP=24 ma lepszy throughput I lepszą latency niż dwa procesy z NP=16.** Plus operacyjnie prostszy (1 proces, mniej VRAM, brak round-robin między hostami).

#### Próba NP=32 — fail (zobacz [LLAMACPP_INT_MAX_BUG.md](LLAMACPP_INT_MAX_BUG.md))

Próbowaliśmy też NP=32. Crash z `GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX)` w llama.cpp CUDA backend. To NIE jest brak VRAM (27 GB wolnego) — to bug w llama.cpp z dużymi KV cache tensorami. Workaround: `OLLAMA_CONTEXT_LENGTH=4096`. Bug udokumentowany osobno.

### Round 5: 1 GPU, NUM_PARALLEL=64, OLLAMA_CONTEXT_LENGTH=4096 (2026-05-14 ~08:04 UTC)

Test poza plateau — sprawdzić czy NP=64 jeszcze coś wyciska.

| Approach | Throughput | p50 | p95 | p99 | VRAM | Wall (s) |
|---|---|---|---|---|---|---|
| **app2 (NP=64, ctx=4K)** | **1.135 req/s** | **57.8 s** | **78.2 s** | **79.9 s** | **18.0 GB** | 88 |

Correctness 100%, 128 client threads (`--parallel-workers 128`).

#### Plateau formalnie potwierdzone

Per-slot scaling efficiency w całym łańcuchu:

| Skok | Sloty | Δ throughput | Per-slot efficiency |
|---|---|---|---|
| NP 4 → 16 | +12 | +0.483 | **40 mreq/s/slot** |
| NP 16 → 24 | +8 | +0.169 | **21 mreq/s/slot** (½) |
| NP 24 → 64 | +40 | **+0.023** | **0.58 mreq/s/slot** (~36× drop) |

Każdy dodatkowy slot ponad ~24 daje **marginalnie nic**. GPU faktycznie wysycony dla tego workloadu (mały model + structured output) — więcej slotów nie ma czego batchować równolegle, tylko dodaje queue depth.

#### Latency eksploduje

| Metric | NP=24 | NP=64 | Zmiana |
|---|---|---|---|
| p50 | 34.2 s | 57.8 s | **+69%** |
| p95 | 50.4 s | 78.2 s | +55% |
| p99 | 57.3 s | 79.9 s | +39% |

To czysty queue wait — każdy request siedzi w kolejce za 63 innymi w batchu. Forward pass per token (eval_rate) zostaje taki sam, ale wall time per request jest pełen "czekania na swoją turę".

#### VRAM observation: context > NUM_PARALLEL

| Config | NUM_PARALLEL | num_ctx | VRAM baseline |
|---|---|---|---|
| Round 4 | 24 | 32768 (default) | 20.5 GB |
| Round 5 | 64 | **4096** | **18.0 GB** |

**64 sloty × 4K kontekstu zajmują mniej VRAM niż 24 sloty × 32K kontekstu.** Per-position KV jest stały — total KV memory skaluje się `NP × num_ctx`. Dla typowych workloadów warto agresywnie zmniejszyć `num_ctx` (większość snippet'ów ma <2K tokenów), nie martwić się o NP.

#### Werdykt: app2 NP=24 zostaje optimum

Marginalny zysk throughputu (+2%) NP=64 nie usprawiedliwia 70% wzrostu latency. Czas dla 1M req:
- NP=24: 10.4 dni
- NP=64: 10.2 dni — oszczędność **5 godzin** kosztem dwukrotnie gorszej latency


## Aktualne rekomendacje per use case (po Round 4)

| Cel | Wybór | Throughput | p50 | p99 | Czas dla 1M req |
|---|---|---|---|---|---|
| **Min latency** (interactive) | `app1` | 0.367 | 2.7 s | 3.5 s | 32 dni |
| **Sweet spot** (low latency + decent throughput) | `app3` | 0.440 | 4.6 s | 5.9 s | 26 dni |
| **Max throughput, simple deployment** | **`app2 NP=24`** | **1.112** | 34 s | 57 s | **10.4 dni** |
| ~~Dual instance~~ | ~~app4 NP=16~~ | 1.105 | 41 s | 68 s | 10.5 dni — **OBSOLETE** |
| **1M+ offline, time matters** | **Gemini 3.1 Flash Lite + cache** | — | — | — | **~4 h, $183** |

### Zmiana vs poprzedniej wersji

**Dual-instance konfiguracje (app3/app4) zostają zdetronizowane.** Round 4 pokazał że `app2 NP=24` ma lepszy throughput I niższe latency niż `app4 NP=16` przy połowie infrastruktury (1 instancja zamiast 2). Wcześniejsza teza "dwa procesy są fundamentalnie lepsze" — formally rejected. Wystarczy zwiększyć NUM_PARALLEL w single instance.

Próba NP=32 nie powiodła się przez bug w llama.cpp ([LLAMACPP_INT_MAX_BUG.md](LLAMACPP_INT_MAX_BUG.md)) — z workaroundem `OLLAMA_CONTEXT_LENGTH=4096` mogłoby działać i prawdopodobnie dałoby kolejne ~10-15% throughputu (już mocno diminishing returns).

## Kluczowe wnioski

### 1. Dwa procesy > batching w jednym procesie (dla GPU saturation)

Single-instance continuous batching (`NUM_PARALLEL=4`) wysyca GPU tylko do ~60-65% SM utilization. Dwa osobne procesy z `NUM_PARALLEL=1` każdy wysycają GPU do ~98%.

**Mechanizm:** Każdy proces ma własny CUDA context. NVIDIA driver multipleksuje kernele z obu kontekstów — gdy proces A robi CPU work (grammar check, sampling), proces B karmi GPU. W jednym procesie wszystkie sekwencje w batchu synchronizują się przy każdym kroku → mikro-gaps między forward passes.

**Implikacja:** Dwa procesy są strategią dla **wysokiej latency-efektywności** (małe per-request latency + dobre wysycenie GPU), w przeciwieństwie do dużego batcha który daje throughput **kosztem latency**.

### 2. Continuous batching: throughput vs latency tradeoff

| Konfiguracja | Throughput vs app1 | Latency vs app1 |
|---|---|---|
| app2 NP=4 (single + batch) | +27% | **+524%** (6.2× gorsza) |
| app3 dual sequential | +21% | +71% (akceptowalne) |
| app4 NP=4 (dual + batch) | +40% | **+981%** (11× gorsza) |

Dla *interaktywnych* zastosowań (każdy snippet z osobna, user czeka) — **batching jest pułapką**. Dla *offline batch* gdzie liczy się tylko total time — batching pomaga, ale margin nad dwoma procesami sequential jest niewielki.

### 3. Workload nie jest ani memory ani compute bound

`nvidia-smi dmon` dla app2 przy NP=4:
- SM utilization: 60-65% (room to grow)
- Memory utilization: **9%** (zupełnie nie memory-bound)
- CPU: 8% na 32 logical cores

Bottleneckiem **nie** są kanoniczne kandydaci. Faktyczne źródła:
- Kernel launch overhead (driver) — tiny model, dużo małych kerneli
- Synchronization między sekwencjami w batchu
- Grammar check po każdym tokenie (constrained decoding na CPU)
- Sampling step

Dla `gemma4:e2b` na RTX 5090: **GPU jest po prostu za szybki dla tego modelu**. Większy model lepiej wysycałby compute.

### 4. NUM_PARALLEL kosztuje VRAM liniowo

```
NP=4  → 13.5 GB baseline (model + 4 KV slots + system)
NP=16 → 17.5 GB baseline (model + 16 KV slots + system)
       +4 GB / +12 slots = ~340 MB per slot
```

KV cache jest alokowany przy starcie Ollamy niezależnie od aktualnej liczby in-flight requestów.

**Implikacja:** Dla większego modelu (np. `gemma3:12b` ~10 GB) `NUM_PARALLEL=16` może w ogóle nie wejść w VRAM. Trzeba dobierać NP do model size.

### 5. Async client nie pomógłby

`ThreadPoolExecutor` z `ollama.Client` (sync) i `asyncio` z `ollama.AsyncClient` mają **tę samą efektywną współbieżność** dla naszego workloadu, bo:
- GIL zwalnia się podczas HTTP I/O (99%+ czasu wątku)
- Localhost HTTP ma µs latency — żadnych zysków z HTTP/2 multiplex
- 8-16 concurrent requestów to skala gdzie wątki kosztują tyle samo co async

Klient **nie jest** wąskim gardłem. Logi pokazują że requesty są kompletowane grupami po 4 (= server NUM_PARALLEL), z minimalnym odstępem — czyli klient utrzymuje wszystkie sloty serwera zapełnione.

### 6. Task Manager kłamie dla CUDA workloadu

Windows Task Manager "Procesor graficzny" 3D% mierzy DXGI engine, nie compute. Niedoszacowuje CUDA workload o ~30%. Realny pomiar: `nvidia-smi dmon -s u -d 1` (kolumny `sm` i `mem`).

## Koszt: local vs Gemini API dla 1M requestów

Z benchmarku: **~913 prompt tokens + ~90 completion tokens per request** (87% promptu to statyczne instrukcje — idealny case dla prompt caching).

Local Ollama (app4, NP=4):
- Czas: **23 dni** non-stop na RTX 5090
- Koszt: ~$45 elektryczność (550W × 23 dni × $0.15/kWh)
- GPU zablokowane na 23 dni

Gemini API (1M req z prompt caching):
| Model | Total cost | Czas (4000 RPM) |
|---|---|---|
| Gemini 3.1 Flash-Lite | **$183** | ~4 godziny |
| Gemini 2.5 Flash | $283 | ~4 godziny |
| Gemini 3 Flash Preview | $367 | ~4 godziny |

**Wniosek:** Local jest tańszy w $$$ (~$138 różnicy), ale Gemini wygrywa total cost of ownership: 4h vs 23 dni, brak monopolizacji GPU, retry per request zamiast restart całego batcha po crashu, możesz w tym czasie używać karty do czego innego.

## Rekomendacje per use case (dla `gemma4:e2b` na 1 × RTX 5090)

| Cel | Wybór | Liczby |
|---|---|---|
| **Min latency** (interactive user) | `app1` — sequential | 0.36 req/s, p50=2.7s, p99=3.5s |
| **Best latency/throughput tradeoff** | `app3` — 2 procesy sequential | 0.44 req/s, p50=4.6s, p99=5.8s |
| **Max throughput** (offline batch) | `app2 NP=16` lub `app4` | 0.46-0.51+ req/s, p50=17-29s |
| **1M+ requestów offline** | **Gemini API z cache** | ~4h, $183 (Flash Lite) |

## Otwarte / pending experiments

- [ ] **Pełne wyniki NUM_PARALLEL=16 runu** — czy app2 dogoni app4? Czy app3/app4 zmieszczą się w VRAM przy 2 instancjach × 16 slotów?
- [ ] **Async client comparison** — empiryczna weryfikacja że to nie zmienia wyników (prognoza: ±2% szum)
- [ ] **Bez structured output** — usunąć `format=ExtractionResult.model_json_schema()` w [extractor.py:287](../extractor.py#L287), zmierzyć ile kosztuje grammar enforcement
- [ ] **Większy model** (`gemma3:12b`) — czy lepiej wysyca GPU? Czy paradoksalnie ma wyższy aggregate throughput?
- [ ] **Gemini API real test** — odpalić `benchmark.py --model gemini:gemini-3.1-flash-lite-preview` na 9 snippetach × N reps żeby zweryfikować correctness + real latency

## Repo artifacts

- [benchmark_parallel.py](../benchmark_parallel.py) — runner 4 konfiguracji
- [extractor.py](../extractor.py) — refactored z `host=` kwarg dla `OllamaClient`
- [summary_parallel.json](../results/summary_parallel.json) — wyniki Round 2 (1 GPU, NP=4)
- `summary_np16.json` — wyniki Round 3 (1 GPU, NP=16, in progress)
- [benchmark.py](../benchmark.py) — istniejący benchmark per-model (nie modifyowany)

## Setup do reprodukcji

```powershell
# Terminal 1 (host_a)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL = "4"        # lub 16 dla Round 3
$env:OLLAMA_MAX_LOADED_MODELS = "1"
ollama serve

# Terminal 2 (host_b) — tylko dla app3/app4
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_NUM_PARALLEL = "4"
$env:OLLAMA_MODELS = "$env:USERPROFILE\.ollama\models"
ollama serve

# Run
.\.venv\Scripts\python.exe benchmark_parallel.py
```
