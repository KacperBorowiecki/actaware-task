# Bug: llama.cpp CUDA cpy assert na dużym KV cache

Repro empirycznie znaleziony podczas testowania `OLLAMA_NUM_PARALLEL=32` na RTX 5090 z modelem `gemma4:e2b`. Może być wart raportowania (i/lub fixa) upstream w llama.cpp.

## Symptom

Ollama crashuje runnera (model fails to load) z:

```
C:\a\ollama\ollama\ml\backend\ggml\ggml\src\ggml-cuda\cpy.cu:396:
GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX) failed
```

Po stronie klienta HTTP 500:
```
ollama._types.ResponseError: model failed to load, this may be due to resource limitations
or an internal error (status code: 500)
```

VRAM **NIE** jest wyczerpana w momencie crashu (27.3 GiB free na 32 GB karcie).

## Repro

**Hardware:** RTX 5090 (32 GiB VRAM), Windows 11
**Ollama:** v0.23.3 (z bundled llama.cpp / ggml)
**Model:** gemma4:e2b (Q4_K_M, "gemma4" architecture per logi)

**Setup który wywołuje crash:**
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_PARALLEL = "32"
# OLLAMA_CONTEXT_LENGTH unset → defaultuje do 32768 (vram-based)
ollama serve
```

Pierwsze zapytanie do `/api/chat` powoduje crash runnera.

## Mechanizm

Z logów Ollamy w momencie crashu:
```
vram-based default context: 32768
Parallel:32  KvSize:1048576    ← 32 sloty × 32768 ctx
BatchSize:512  FlashAttention:Enabled
```

`KvSize=1048576` to **łączna liczba pozycji KV cache** (sloty × ctx). Każda pozycja zajmuje:
- key + value
- × num_layers (36 dla gemma4:e2b)
- × hidden_dim (różne per warstwa)

Wynik: pojedynczy tensor (np. key cache jednej warstwy) przekracza **INT_MAX = 2,147,483,647 bytes (~2 GiB)**, na który assert w [ggml-cuda/cpy.cu:396](https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/cpy.cu) reaguje.

## Workaround (działa)

Ograniczyć context length tak, żeby `Parallel × ctx × per_position_size` nie przekraczał ~2 GiB per tensor:

```powershell
$env:OLLAMA_CONTEXT_LENGTH = "4096"     # zamiast default 32768
$env:OLLAMA_NUM_PARALLEL = "32"
ollama serve
```

Dla naszego workloadu (snippety ~1K tokenów) `ctx=4096` jest aż nadto.

Alternatywnie ograniczyć NUM_PARALLEL. NP=24 z default ctx też działa (poniżej granicy assert).

## Reproduce w izolacji (proposed)

Hipoteza do weryfikacji:
1. `NP × ctx ≈ 1,048,576` → crash (niezależnie jak rozdzielone, np. NP=64 × ctx=16384 powinno też pęknąć)
2. `NP × ctx ≈ 524,288` → OK (NP=32 × ctx=16384 albo NP=16 × ctx=32768)
3. Próg może zależeć od `n_layers` i `hidden_dim` modelu — dla większego modelu próg byłby niższy

Warto by zrobić skrypt który skanuje przestrzeń (NP, ctx) i znajduje dokładny próg empiryczny dla różnych modeli.

## Możliwe rozwiązania upstream

**Krótkoterminowe (defensive check):**
- W llama.cpp przy alokacji KV cache, jeśli `total_kv_bytes > INT_MAX`, zwrócić sensowny error message zamiast crashować w assert. User dostaje "KV cache too large, reduce NP or context", a nie nieczytelny GGML_ASSERT.

**Średnioterminowe (Ollama-side guard):**
- Ollama już ma `vram-based default context` — można dodać dodatkowe ograniczenie żeby `NP × default_ctx × per_position_size < INT_MAX`. Lepiej wybrać konserwatywny default niż crashować.

**Długoterminowe (llama.cpp ggml refactor):**
- ggml zaakceptuje tensory >2 GB w cpy kernel — wymaga przejścia z `int` na `int64_t` w indexach offset/size w niektórych miejscach. Ryzyko regresji wydajności.
- Albo: dzielić duży KV cache na kilka mniejszych tensorów per warstwa (logical view, fizyczne tensory <2GB). Inwazyjne.

## Linki do śledzenia

- llama.cpp issues: https://github.com/ggerganov/llama.cpp/issues — search "INT_MAX cpy" or "KV cache size"
- Ollama issues: https://github.com/ollama/ollama/issues — search "INT_MAX" or "model failed to load NUM_PARALLEL"

## Status

- [ ] Sprawdzić czy issue już zgłoszone w llama.cpp / ollama
- [ ] Reproduce w izolacji bez Ollamy (raw llama.cpp `llama-bench` lub `llama-cli`)
- [ ] Jeśli nowe — otworzyć issue z minimalnym repro
- [ ] Ewentualnie PR z defensive check (option 1 powyżej, najtańszy)

## Powiązane notatki

- [OLLAMA_CONCURRENCY_NOTES.md](OLLAMA_CONCURRENCY_NOTES.md) — kontekst w jakim natknęliśmy się na ten bug (testy NUM_PARALLEL na RTX 5090)
