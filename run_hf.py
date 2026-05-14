"""Standalone benchmark: full BF16 google/gemma-4-E2B-it via HuggingFace transformers
with `outlines` constrained decoding for guaranteed JSON-schema output.

Compares quality (match vs expected_output.json) and speed (per-snippet wall time)
against the Ollama Q4_K_M run. Does NOT touch the main pipeline / .env / extractor
client factory — it just reuses the post-LLM machinery (grounding, unit conversion,
expected output) from `extractor.py` / `run.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from pydantic import RootModel, ValidationError
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor

import outlines

from extractor import (
    EXTRACTION_PROMPT,
    RawEntry,
    parse_snippets,
    process_snippet,
)


class EntryList(RootModel[list[RawEntry]]):
    """Outlines-friendly schema: a flat JSON array of entries (matches gemma-4's
    natural output shape). The `ExtractionResult` wrapper {"entries": [...]}
    works for Ollama but causes gemma-4 to degenerate to `{"entries":[]}` under
    constrained decoding — it gets pushed off its preferred path. Flat list
    keeps the model on-distribution; we wrap manually after parsing."""


HERE = Path(__file__).parent
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
logger = logging.getLogger("run_hf")


def _vram_snapshot(device: str) -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) on the given CUDA device."""
    idx = int(device.split(":")[1]) if ":" in device else 0
    return (
        torch.cuda.memory_allocated(idx) / 1024**2,
        torch.cuda.memory_reserved(idx) / 1024**2,
    )


class CPULookupEmbedding(nn.Module):
    """Drop-in replacement for `embed_tokens_per_layer` that keeps the full weight
    on pinned CPU memory and only transfers the gathered rows to GPU per forward.

    `accelerate`'s default cpu-offload (via device_map={...: 'cpu'}) materializes
    the WHOLE weight on GPU every forward, then frees it — for a 4.7 GB embedding
    that's ~500 GB of PCIe traffic across a 100-token generation. PLE is a static
    lookup (no matmul, no cross-row interactions), so we can gather rows on CPU
    and transfer only seq_len * embed_dim * 2 bytes (~16 MB / prompt) instead.

    Preserves the `Gemma4TextScaledWordEmbedding` behavior of multiplying the
    embedding output by `embed_scale` (=16.0 for gemma-4-E2B). Without this
    scaling the per-layer signal is ~16x too small and the model emits gibberish.
    """

    def __init__(self, original: nn.Embedding, target_device: torch.device) -> None:
        super().__init__()
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        self.target_device = target_device
        self.embed_scale = float(getattr(original, "scalar_embed_scale", 1.0))
        weight = original.weight.detach().to("cpu").contiguous()
        if target_device.type == "cuda":
            weight = weight.pin_memory()
        # Buffer (not Parameter): no autograd, but persists in state_dict.
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Synchronous transfers: non_blocking on a non-pinned destination tensor
        # can race with the embedding read on GPU, producing garbled outputs.
        ids_cpu = input_ids.detach().cpu()
        out_cpu = nn.functional.embedding(ids_cpu, self.weight, self.padding_idx)
        out_gpu = out_cpu.to(self.target_device)
        return out_gpu * self.embed_scale

    def _apply(self, fn, recurse: bool = True):
        # `.to(device)` / `.cuda()` / `.cpu()` call _apply on every buffer/param.
        # We must keep our `weight` buffer on CPU regardless — that's the whole
        # point of this module. Temporarily detach the buffer, let the parent
        # apply `fn` to everything else, then reattach.
        saved_weight = self._buffers.pop("weight", None)
        try:
            return super()._apply(fn, recurse)
        finally:
            if saved_weight is not None:
                self._buffers["weight"] = saved_weight


def _get_submodule(root: nn.Module, dotted: str) -> nn.Module:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _set_submodule(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


class HFClient:
    """Implements the LLMClient protocol against a local HF transformers model
    with `outlines` for schema-constrained decoding."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        offload_ple: bool = False,
    ) -> None:
        # Follows the HF model card example: AutoProcessor is the official entry point
        # for Gemma 4 (multimodal). For text-only inference processor.tokenizer is
        # functionally identical to AutoTokenizer.from_pretrained(...) — same class
        # (GemmaTokenizer), same chat template, same token IDs.
        logger.info("Loading processor %s", model_id)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._device = device

        vram_before = _vram_snapshot(device)
        logger.info(
            "VRAM before load on %s: allocated=%.1f MB, reserved=%.1f MB",
            device, vram_before[0], vram_before[1],
        )

        t0 = time.perf_counter()
        if offload_ple:
            # Load entirely on CPU first so the PLE weight never touches VRAM.
            # PyTorch's caching allocator otherwise holds onto the freed slot:
            # reserved memory stays at the load-time peak (~9.8 GB) even after we
            # del + empty_cache. Loading on CPU and only moving non-PLE modules
            # to GPU keeps both allocated AND reserved at ~5.3 GB.
            logger.info("Loading model %s (dtype=%s) on CPU for PLE offload", model_id, dtype)
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            hf_model.eval()

            ple_path = "model.language_model.embed_tokens_per_layer"
            original_ple = _get_submodule(hf_model, ple_path)
            if not isinstance(original_ple, nn.Embedding):
                raise RuntimeError(
                    f"Expected nn.Embedding at {ple_path}, got {type(original_ple).__name__}"
                )
            ple_size_gb = original_ple.weight.element_size() * original_ple.weight.numel() / 1024**3
            logger.info(
                "PLE offload: replacing %s (%.2f GB BF16) with CPULookupEmbedding",
                ple_path, ple_size_gb,
            )
            _set_submodule(hf_model, ple_path, CPULookupEmbedding(original_ple, torch.device(device)))
            del original_ple

            logger.info("Moving non-PLE modules to %s", device)
            hf_model = hf_model.to(device)
        else:
            logger.info("Loading model %s (dtype=%s, device=%s)", model_id, dtype, device)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device,
            )
            hf_model.eval()
        self.load_seconds = time.perf_counter() - t0
        logger.info("Model loaded in %.2fs", self.load_seconds)

        vram_after = _vram_snapshot(device)
        self.vram_before_mb = vram_before
        self.vram_after_load_mb = vram_after
        logger.info(
            "VRAM after load on %s: allocated=%.1f MB, reserved=%.1f MB (delta allocated: +%.1f MB)",
            device, vram_after[0], vram_after[1], vram_after[0] - vram_before[0],
        )
        # Spot-check: report which device each parameter actually sits on (CPU offload?)
        device_counts: dict[str, int] = {}
        for p in hf_model.parameters():
            d = str(p.device)
            device_counts[d] = device_counts.get(d, 0) + p.numel()
        total = sum(device_counts.values())
        for d, n in sorted(device_counts.items()):
            logger.info("  params on %s: %.2fB (%.1f%%)", d, n / 1e9, 100 * n / total)

        # outlines wraps the model+tokenizer; passing processor.tokenizer (not the full
        # processor) keeps outlines on its text-only Transformers path rather than the
        # multimodal TransformersMultiModal path which expects image/audio inputs.
        self._model = outlines.from_transformers(hf_model, self._processor.tokenizer)
        self._model_id = model_id

    def _build_prompt(self, snippet_text: str) -> str:
        # System message + user message, exactly as in the HF docs example.
        # enable_thinking=False: skip the model's "thinking" preamble so it emits JSON directly.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(snippet_text=snippet_text)},
        ]
        return self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def extract(self, snippet_text: str, snippet_id: str = "?") -> list[RawEntry]:
        prompt = self._build_prompt(snippet_text)
        # output_type=EntryList constrains decoding to a flat JSON array.
        # Sampling params from the Gemma 4 model card "Best Practices" — applied across
        # all use cases including structured extraction. Greedy decoding caused the
        # model to truncate multi-row tables after the first entry (close-bias under
        # constrained decoding); sampling lets the model stay on its trained distribution.
        raw_json = self._model(
            prompt,
            output_type=EntryList,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        try:
            return EntryList.model_validate_json(raw_json).root
        except ValidationError as exc:
            logger.warning(
                "[%s] HF returned unparseable JSON despite constraint: %s (raw: %r)",
                snippet_id, exc, raw_json[:200],
            )
            return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction with HF transformers + outlines.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id (default: %(default)s).")
    parser.add_argument("--snippets", default="snippets.txt")
    parser.add_argument("--output", default="output_hf.json")
    parser.add_argument("--expected", default="expected_output.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload-ple", action="store_true",
                        help="Offload Gemma 4's embed_tokens_per_layer (PLE lookup, ~4.7 GB BF16) to CPU.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Sampling is non-deterministic; fix the seed so the run reproduces.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    snippets_path = HERE / args.snippets
    output_path = HERE / args.output
    expected_path = HERE / args.expected

    snippets = parse_snippets(snippets_path.read_text(encoding="utf-8"))
    logger.info("Parsed %d snippets", len(snippets))

    client = HFClient(args.model, device=args.device, offload_ple=args.offload_ple)

    # One warmup call so the first measured snippet excludes CUDA-graph / autotune cost.
    logger.info("Warming up generation kernel...")
    t_warm = time.perf_counter()
    _ = client.extract("Warmup. Total CO2 emissions in 2024: 1 metric ton.", snippet_id="_warmup")
    warmup_seconds = time.perf_counter() - t_warm
    vram_after_warmup = _vram_snapshot(args.device)
    logger.info(
        "Warmup done in %.2fs. VRAM after warmup: allocated=%.1f MB, reserved=%.1f MB",
        warmup_seconds, vram_after_warmup[0], vram_after_warmup[1],
    )

    # Timed run
    results: dict[str, list[dict]] = {}
    per_snippet: dict[str, float] = {}
    t_total = time.perf_counter()
    for sid, text in snippets.items():
        t0 = time.perf_counter()
        try:
            results[sid] = process_snippet(text, client, snippet_id=sid)
        except Exception:
            logger.exception("[%s] failed", sid)
            results[sid] = []
        per_snippet[sid] = time.perf_counter() - t0
        logger.info("%s done in %.2fs (%d entries)", sid, per_snippet[sid], len(results[sid]))
    total_wall = time.perf_counter() - t_total

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", output_path.name)

    # Diff vs expected
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for sid in sorted(expected):
        if results.get(sid) != expected[sid]:
            mismatches.append(sid)
            logger.warning(
                "MISMATCH %s: expected=%s actual=%s",
                sid, expected[sid], results.get(sid),
            )

    vram_final = _vram_snapshot(args.device)

    # Summary
    print()
    print("=" * 72)
    print(f"Model:           {args.model}")
    print(f"Model load:      {client.load_seconds:.2f}s")
    print(f"Warmup call:     {warmup_seconds:.2f}s")
    print(f"Total wall (9 snippets, warm): {total_wall:.2f}s")
    print(f"Mean per snippet (warm):       {total_wall / len(snippets):.2f}s")
    print()
    print("VRAM usage (MB allocated / reserved):")
    print(f"  before load:    {client.vram_before_mb[0]:7.1f} / {client.vram_before_mb[1]:7.1f}")
    print(f"  after load:     {client.vram_after_load_mb[0]:7.1f} / {client.vram_after_load_mb[1]:7.1f}   "
          f"(+{client.vram_after_load_mb[0] - client.vram_before_mb[0]:.1f} MB allocated)")
    print(f"  after warmup:   {vram_after_warmup[0]:7.1f} / {vram_after_warmup[1]:7.1f}")
    print(f"  after 9 snips:  {vram_final[0]:7.1f} / {vram_final[1]:7.1f}")
    print()
    print("Per-snippet times:")
    for sid in sorted(per_snippet):
        match = "OK" if results.get(sid) == expected.get(sid) else "MISS"
        print(f"  {sid}: {per_snippet[sid]:6.2f}s   [{match}]")
    print()
    print(f"Correctness: {len(snippets) - len(mismatches)}/{len(snippets)} match expected_output.json")
    if mismatches:
        print(f"Mismatches: {mismatches}")
    print("=" * 72)


if __name__ == "__main__":
    main()
