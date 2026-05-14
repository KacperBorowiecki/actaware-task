# Gemma 4 BF16 (HF) + PLE offload vs Ollama Q4 — Notatka z sesji

**Data:** 2026-05-12
**Projekt:** Actaware CO2 extraction task
**Sprzęt:** 2× RTX 5090 (32 GB każda), CUDA 12.8, Windows 11
**Modele:** `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `gemma4:e2b`, `gemma4:e4b` (Ollama Q4_K_M)

## Cel
Sprawdzić czy pełen model BF16 z HuggingFace daje lepszą jakość niż skwantyzowana wersja Q4_K_M w Ollamie. Dodatkowo: zmieścić go w VRAM na RTX 5090 przy użyciu Per-Layer Embeddings (PLE) offload.

Stary pipeline (Ollama + Gemini) zostawiony bez zmian. Dodany nowy plik `run_hf.py` jako standalone benchmark, reużywa `process_snippet`, `parse_snippets`, `expected_output.json`.

---

## Główne odkrycia

### 1. Outlines constraint shape musi pasować do "naturalnego" outputu modelu

**Problem:** `output_type=ExtractionResult` (Pydantic `{entries: list[RawEntry]}`) dawał **0/9** — model wypluwał `{"entries": []}` dla każdego snippeta.

**Powód:** gemma-4 niewymuszony naturalnie generuje **płaską listę** `[{"value": ..., "unit": ...}]`. Constrained decoding wymusza pierwszy token `{"entries":[`, czyli ścieżkę inną niż model preferuje. Confidence się rozsypuje i greedy wybiera `]` zaraz po `[` bo to "tańszy" wybór niż otwarcie `{`.

**Fix:** Constraint na typ "płaska lista", przez `RootModel`:

```python
class EntryList(RootModel[list[RawEntry]]):
    pass

# w outlines:
result = model(prompt, output_type=EntryList, ...)
entries = EntryList.model_validate_json(result).root
```

Po fixie: **7/9** (E2B). Multi-row tabele (s7, s9) nadal urywane.

### 2. Close-bias na multi-row tabelach (outlines + greedy)

Po wygenerowaniu pierwszego validnego entry model staje przed wyborem: `,` (kontynuuj) vs `]` (zamknij). Greedy + regex constraint → `]` częściej wygrywa.

- **E2B 7/9** — zawsze urywał s7 (tabela 3-letnia: 2022/2023/2024) i s9 (3 scope-y) do pierwszego wpisu
- **E4B 9/9** — większy model ma silniejszy prior na "lista nie skończona", trafia wszystkie

Innymi słowy: **rozmiar modelu zwycięża nad mechanizmem constraint** — E4B przebija problem siłą.

### 3. HF rekomendowane sampling params SZKODZĄ pod constrained decoding

HF docs: "Use temperature=1.0, top_p=0.95, top_k=64 for all use cases."

Dla unconstrained generation: pewnie OK. Pod outlines:
- greedy → 7/9 (E2B)
- z HF sampling params → 6/9 (E2B)

Sampling combined with regex constraint produces worse outputs because the model is more often pushed off its preferred path. Dla ekstrakcji z constraint: **greedy lepszy**.

### 4. AutoProcessor vs AutoTokenizer (Gemma 4)

Dla text-only **funkcjonalnie identyczne** — sprawdzone empirycznie:
- `Gemma4Processor.tokenizer` to dokładnie ten sam obiekt co `AutoTokenizer.from_pretrained(...)` zwraca (klasa `GemmaTokenizer`)
- `processor.apply_chat_template(...)` == `tokenizer.apply_chat_template(...)` (same string, same token IDs)

`AutoProcessor` jest "official" wg HF docs, ale wymaga `torchvision` (multimodal video processor). Dla samego tekstu różnica = 0 — tylko semantyczna zgodność z docs.

### 5. Per-Layer Embeddings (PLE) offload — esencja sesji

**Reference:** [Reddit thread: Per-Layer Embeddings explainer](https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/)

Gemma 4 E2B/E4B mają ogromną macierz PLE:
- `model.language_model.embed_tokens_per_layer` = `(vocab=262144, all_layers_combined=8960)`
- E2B: 2.35B params = **4.7 GB BF16**
- E4B: ~2.81B params = **5.25 GB BF16**
- Klasa: `Gemma4TextScaledWordEmbedding(nn.Embedding)` — robi `super().forward(input_ids) * embed_scale` gdzie `embed_scale=16.0`

PLE to **statyczna lookup table** — żaden matmul, brak cross-row interactions. Może siedzieć na CPU; per-token gather to ~17 KB transfer, per prompt ~16 MB.

#### 5a. ❌ Co NIE działa: `device_map` przez accelerate

```python
device_map = {"model.language_model.embed_tokens_per_layer": "cpu", "": "cuda:0"}
```

VRAM allocated spada z 9.7 → 5.3 GB ✓ ALE wall **3× wolniej** (119s vs 41s).

**Powód:** accelerate kopiuje CAŁĄ macierz (4.7 GB) z CPU na GPU na każdy forward, potem zwalnia. Per 100 tokenów generation: ~500 GB ruchu po PCIe.

#### 5b. ✓ Co działa: custom `CPULookupEmbedding`

```python
class CPULookupEmbedding(nn.Module):
    def __init__(self, original: nn.Embedding, target_device: torch.device):
        super().__init__()
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        self.target_device = target_device
        self.embed_scale = float(getattr(original, "scalar_embed_scale", 1.0))
        weight = original.weight.detach().to("cpu").contiguous()
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.detach().cpu()                      # sync!
        out_cpu = nn.functional.embedding(ids_cpu, self.weight, self.padding_idx)
        out_gpu = out_cpu.to(self.target_device)                # sync!
        return out_gpu * self.embed_scale                       # critical!

    def _apply(self, fn, recurse=True):
        # .to(device) by przeciągnęło weight buffer na GPU. Blokujemy.
        saved = self._buffers.pop("weight", None)
        try:    return super()._apply(fn, recurse)
        finally:
            if saved is not None: self._buffers["weight"] = saved
```

#### 5c. Krytyczne gotchas (każdy z nich łamał setup po cichu)

1. **`embed_scale = 16.0` MUSI być pomnożone** w forward.
   Bez: outputy skompresowane 16× → model emit nonsense (`"8,,000"`, `"bbScopeESESUC..."`, double commas, character repetition).
   To NIE jest oczywiste z samej nazwy modułu — trzeba zaglądnąć do `transformers/models/gemma4/modeling_gemma4.py`:
   ```python
   class Gemma4TextScaledWordEmbedding(nn.Embedding):
       def forward(self, input_ids):
           return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
   ```

2. **Synchronous transfers**. `non_blocking=True` na non-pinned destination → race condition, GPU czyta z pamięci zanim PCIe transfer się skończy. Outputy "prawie poprawne" ale skorumpowane (np. `year=22032` zamiast `2023`). Po prostu `.to(device)` bez `non_blocking`.

3. **Override `_apply`** — bez tego `hf_model.to("cuda")` przeciąga weight buffer naszego modułu z powrotem na GPU.

4. **Buffer (`register_buffer`), nie Parameter** — semantically lookup table is not a trainable parameter.

#### 5d. Dwie strategie offloadingu

| | v1: GPU first → swap | v2: CPU first → move-non-PLE |
|---|---|---|
| Load | from_pretrained(device_map="cuda:0"), potem swap PLE | from_pretrained(no device_map), swap PLE, hf_model.to("cuda:0") |
| Allocated | 5.26 GB ✓ | 5.27 GB ✓ |
| **Reserved** | ~9.85 GB ⚠ | **~5.76 GB ✓** |
| Load time | 6 s | 10.5 s (CPU→GPU transfer 5.3 GB) |
| nvidia-smi widzi | ~9.85 GB | ~5.76 GB |

**v2 jest lepsze** — PLE nigdy nie wchodzi na GPU, więc PyTorch caching allocator nie ma czego trzymać w cache (po freed PLE alokator nie oddaje pamięci CUDA z powrotem).

---

## Finalna tabela porównawcza (9-snippet CO2 extraction)

| Setup | Correctness | Wall (9 snip) | Avg/snip | VRAM loaded | Disk |
|---|---|---|---|---|---|
| **Ollama gemma4:e2b Q4_K_M** (`num_ctx=131072`) | **9/9** | **29 s** | **2.80 s** | 9.0 GB | **7.2 GB** |
| **Ollama gemma4:e2b Q4_K_M** (`num_ctx=5000`) | 9/9 (extrapolated) | ~29 s | ~2.8 s | **7.9 GB** | **7.2 GB** |
| HF gemma-4-E2B-it BF16 + PLE offload | 7/9 | 26 s | 2.93 s | **5.76 GB** | 10.3 GB |
| Ollama gemma4:e4b Q4_K_M (`num_ctx=131072`) | 9/9 | 47 s | 3.64 s | 13 GB | 9.6 GB |
| Ollama gemma4:e4b Q4_K_M (`num_ctx=5000`) | 9/9 (extrapolated) | ~47 s | ~3.6 s | **10 GB** | 9.6 GB |
| **HF gemma-4-E4B-it BF16 + PLE offload** | **9/9** | 40 s | 4.48 s | 10.34 GB | 16.0 GB |

### VRAM breakdown — wpływ kontekstu na Ollama VRAM

Domyślnie Ollama alokuje KV cache z góry dla pełnego `context length` modelu (`ollama show <model>` → `context length 131072`). Sprawdzone empirycznie z `num_ctx=5000`:

| model | num_ctx=131072 | num_ctx=5000 | różnica = KV cache 131K |
|---|---|---|---|
| gemma4:e2b | 9.0 GB | **7.9 GB** | ~1.1 GB |
| gemma4:e4b | 13.0 GB | **10.0 GB** | ~3.0 GB |

Większy narzut dla e4b bo ma `embedding_length=2560` (vs 1536 dla e2b) → grubszy KV per token.

**Ustawienie `num_ctx` w Ollamie:**

```bash
# Per-request via REST API
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:e2b",
  "prompt": "...",
  "options": {"num_ctx": 5000},
  "keep_alive": "60s"
}'

# W naszym extractor.py (OllamaClient.extract): dodać `options={"num_ctx": 5000, "temperature": 0.0}`
# w wywołaniu self._client.chat(...)
```

**Same wagi Q4 (bez KV):**
- gemma4:e2b: ~3.6 GB (z 7.2 GB na dysku, czyli RAM = disk × 0.5 z powodu mmap/dedup buffers? Wymaga sprawdzenia.)
- gemma4:e4b: ~4.8 GB

**Wniosek:** dla naszego zadania (~1000-token prompts) ustawienie `num_ctx=5000` w Ollamie zwalnia 1-3 GB VRAM bez wpływu na correctness. Wciąż jednak HF E2B + PLE offload (5.76 GB) jest niżej niż Ollama e2b@5000ctx (7.9 GB) — czyli BF16 z PLE offload **ma niższy footprint VRAM niż Q4 w Ollamie** dla tego rozmiaru modelu, mimo 4× wyższej precyzji.

### Najważniejsze wnioski

1. **Q4_K_M nie pogarsza dokładności na tym zadaniu** — tam gdzie HF BF16 trafia, Ollama Q4 zwraca **identyczne** liczby i lata. Architektura modelu (E2B vs E4B) ma większy wpływ na correctness niż kwantyzacja.

2. **HF E4B BF16 + PLE offload mieści się w 10.34 GB VRAM** — to **mniej niż Ollama gemma4:e4b Q4 w 13 GB** (mimo precyzji 4×). Dzięki temu że Ollama włącza pełen 131K KV cache.

3. **Dla tego konkretnego zadania Ollama gemma4:e2b Q4 jest sweet-spotem.** Najszybsze, najmniejszy disk footprint, 9/9. Idź do BF16 tylko gdy:
   - Masz powód podejrzewać że Q4 pogarsza wyniki (tu nie pogarsza)
   - Potrzebujesz dokładnie tej samej numerycznej precyzji co model author (research)
   - Chcesz pełną kontrolę nad samplowaniem/intermediate states

---

## Pliki które się zmieniły w tej sesji

- **`run_hf.py`** — nowy, standalone benchmark z PLE offload
- **`requirements.txt`** — dodane `torch>=2.7`, `torchvision>=0.22`, `transformers>=5.0`, `accelerate>=1.0`, `outlines>=1.2` (z komentarzem o cu128 wheels)
- **`extractor.py`** — dodany `gemma4:e4b` do `OllamaClient.ALLOWED_MODELS`

## Co spróbować dalej

- [ ] **Ollama z `num_ctx 2048`** — fair-fight VRAM comparison
- [ ] **llama.cpp direct** z `-ot "per_layer_token_embd\.weight=CPU"` — Ollama nie eksponuje tej opcji oficjalnie, ale binarka llama-server tak. Można podpiąć nasz OllamaClient pod ten endpoint (kompatybilne API).
- [ ] **Prompt hardening na multi-row** — eksplicitne "list ALL years/scopes, never truncate after the first entry" w EXTRACTION_PROMPT (osobno dla HF wrapper, bez ruszania głównego)
- [ ] **`pin_memory()` + `torch.cuda.synchronize()` po transferze** — czy z proper sync barrier non_blocking byłby szybszy bez race conditions?
- [ ] **`lm-format-enforcer`** zamiast `outlines` — alternative constraint library, może ma inne biases
- [ ] **Beam search z outlines** — `num_beams=3` mogłoby pomóc na multi-row close-bias?
- [ ] **`gemma-4-E2B`** (base, bez `-it`) z few-shot examples w prompcie — może base model bez instruct-fine-tune lepiej radzi sobie z continuation pod constraint?

## Komendy które warto pamiętać

```bash
# Quick run HF (E2B z PLE offload)
.venv/Scripts/python.exe run_hf.py --offload-ple

# E4B z PLE offload
.venv/Scripts/python.exe run_hf.py --model google/gemma-4-E4B-it --offload-ple

# Ollama health check
ollama list                          # disk sizes
ollama ps                            # currently loaded models + VRAM
ollama show gemma4:e2b               # architecture, context, default params

# HF cache inspection
python -c "from huggingface_hub import scan_cache_dir; \
    [print(f'{r.repo_id}: {r.size_on_disk_str}') for r in scan_cache_dir().repos]"

# VRAM debug w trakcie sesji Python:
import torch
torch.cuda.memory_allocated(0) / 1024**2   # MB w tensorach
torch.cuda.memory_reserved(0) / 1024**2    # MB w PyTorch pool
torch.cuda.empty_cache()                   # zwolnij reserved (czasem)
```

## Referencje

- HF model card: <https://huggingface.co/google/gemma-4-E2B>
- PLE explainer (Reddit, -p-e-w-): <https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/>
- `Gemma4TextScaledWordEmbedding` source: `transformers/models/gemma4/modeling_gemma4.py`
- llama.cpp tensor override: `-ot "<regex>=<device>"`
- outlines docs: <https://dottxt-ai.github.io/outlines/>
