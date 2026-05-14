"""Speed benchmark for multiple LLM models (Ollama + Gemini).

Hardcoded model list (override with --model client:model_name to run just one),
configurable repeats. Records per-run metrics to JSONL and produces a
summary.json with per-(client, model) aggregates.

VRAM measurement uses NVML directly (nvidia-ml-py). For each Ollama model we
run a single warmup pass first (loaded to GPU but not counted in summary), then
sample total GPU VRAM via NVML and store both the absolute used and the delta
versus a baseline sampled before any model was loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from extractor import OLLAMA_HOST, get_llm_client, parse_snippets, process_snippet

logger = logging.getLogger(__name__)


def _unload_all_ollama(host: str) -> int:
    """Force Ollama to unload every currently-loaded model (keep_alive=0).

    Returns the number of models unloaded. Used to ensure VRAM deltas are
    isolated per-model rather than cumulative across the benchmark run.
    """
    try:
        import ollama
        c = ollama.Client(host=host)
        ps = c.ps()
        loaded = [m.get("model") or m.get("name") for m in ps.get("models", [])]
        loaded = [m for m in loaded if m]
        if not loaded:
            return 0
        logger.info("Unloading %d Ollama model(s) for clean VRAM: %s",
                    len(loaded), loaded)
        for m in loaded:
            try:
                c.generate(model=m, prompt="", keep_alive=0)
            except Exception as exc:
                logger.warning("  failed to unload %s: %s", m, exc)
        time.sleep(0.5)  # let CUDA actually release VRAM
        return len(loaded)
    except Exception as exc:
        logger.warning("unload-all failed: %s", exc)
        return 0


MODELS: list[tuple[str, str]] = [
    ("gemini", "gemini-3.1-flash-lite-preview"),
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-pro"),
    ("ollama", "gemma4:e2b"),
    ("ollama", "gemma4:e4b"),
    ("ollama", "gemma3:12b"),
    ("ollama", "qwen3:30b-a3b-q8_0"),
]


class NVMLHelper:
    def __init__(self) -> None:
        self.ok = False
        self._pynvml = None
        self._handles: list = []
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._pynvml = pynvml
            self.ok = True
        except Exception as exc:
            print(
                f"[WARN] NVML unavailable, GPU VRAM fields will be null: {exc}",
                file=sys.stderr,
            )

    def used_mb(self) -> int | None:
        if not self.ok:
            return None
        try:
            total = sum(
                self._pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles
            )
            return round(total / 1024 / 1024)
        except Exception:
            return None

    def shutdown(self) -> None:
        if self.ok and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def _tps(tokens: int | None, duration_ns: int | None) -> float | None:
    if tokens is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(tokens / (duration_ns / 1e9), 2)


def _wall_tps(tokens: int | None, wall_ms: float) -> float | None:
    if tokens is None or wall_ms <= 0:
        return None
    return round(tokens / (wall_ms / 1000), 2)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_one(
    client,
    snippet_id: str,
    snippet_text: str,
    expected_entries: list[dict] | None,
    *,
    client_type: str,
    model: str,
    rep: int,
    warmup: bool,
    gpu_vram_used_mb: int | None,
    gpu_vram_delta_mb: int | None,
) -> dict:
    t0 = time.perf_counter()
    error: str | None = None
    entries_count = 0
    correct: bool | None = None
    try:
        actual = process_snippet(snippet_text, client, snippet_id=snippet_id)
        entries_count = len(actual)
        if expected_entries is not None:
            correct = (actual == expected_entries)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s/%s] extract failed", client_type, model)
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)

    m = getattr(client, "last_metrics", None) or {}
    prompt_tokens = m.get("prompt_tokens")
    completion_tokens = m.get("completion_tokens")
    if "total_tokens" in m:
        total_tokens = m.get("total_tokens")
    elif prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    else:
        total_tokens = None

    return {
        "timestamp": _utc_now_iso(),
        "client": client_type,
        "model": model,
        "snippet_id": snippet_id,
        "rep": rep,
        "warmup": warmup,
        "wall_duration_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "load_duration_ns": m.get("load_duration_ns"),
        "prompt_eval_duration_ns": m.get("prompt_eval_duration_ns"),
        "eval_duration_ns": m.get("eval_duration_ns"),
        "total_duration_ns": m.get("total_duration_ns"),
        "prompt_eval_tps": _tps(prompt_tokens, m.get("prompt_eval_duration_ns")),
        "eval_tps": _tps(completion_tokens, m.get("eval_duration_ns")),
        "wall_tps": _wall_tps(completion_tokens, wall_ms) if error is None else None,
        "gpu_vram_used_mb": gpu_vram_used_mb,
        "gpu_vram_delta_mb": gpu_vram_delta_mb,
        "entries_extracted": entries_count,
        "correct": correct,
        "error": error,
    }


def _stats(values: list) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def build_summary(
    records: list[dict],
    *,
    repeats: int,
    snippets_count: int,
    gpu_vram_baseline_used_mb: int | None,
    total_duration_ms: float,
) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = {}
    warmups: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for r in records:
        key = (r["client"], r["model"])
        if r["warmup"]:
            warmups[key] = r
            continue
        if key not in by_key:
            order.append(key)
        by_key.setdefault(key, []).append(r)

    models = []
    for key in order:
        client, model = key
        runs = by_key[key]
        ok_runs = [r for r in runs if r["error"] is None]
        cold_load_ms = None
        wm = warmups.get(key)
        if wm and wm.get("load_duration_ns"):
            cold_load_ms = round(wm["load_duration_ns"] / 1e6, 2)
        vram_used = next(
            (r["gpu_vram_used_mb"] for r in runs if r["gpu_vram_used_mb"] is not None),
            None,
        )
        vram_delta = next(
            (r["gpu_vram_delta_mb"] for r in runs if r["gpu_vram_delta_mb"] is not None),
            None,
        )
        per_model_total_ms = (
            round(sum(r["wall_duration_ms"] for r in ok_runs), 2)
            if ok_runs else 0.0
        )
        scored = [r for r in ok_runs if r.get("correct") is not None]
        n_correct = sum(1 for r in scored if r["correct"])
        accuracy_pct = (
            round(100 * n_correct / len(scored), 1) if scored else None
        )
        models.append({
            "client": client,
            "model": model,
            "n_runs": len(ok_runs),
            "n_errors": len(runs) - len(ok_runs),
            "n_correct": n_correct if scored else None,
            "n_scored": len(scored),
            "accuracy_pct": accuracy_pct,
            "total_duration_ms": per_model_total_ms,
            "cold_load_duration_ms": cold_load_ms,
            "gpu_vram_used_mb": vram_used,
            "gpu_vram_delta_mb": vram_delta,
            "wall_duration_ms": _stats([r["wall_duration_ms"] for r in ok_runs]),
            "prompt_tokens": _stats([r["prompt_tokens"] for r in ok_runs]),
            "completion_tokens": _stats([r["completion_tokens"] for r in ok_runs]),
            "eval_tps": _stats([r["eval_tps"] for r in ok_runs]),
            "prompt_eval_tps": _stats([r["prompt_eval_tps"] for r in ok_runs]),
            "wall_tps": _stats([r["wall_tps"] for r in ok_runs]),
        })

    def _sort_key(m: dict) -> tuple[float, float]:
        acc = m.get("accuracy_pct")
        wtps = m.get("wall_tps")
        return (
            -(acc if acc is not None else -1.0),
            -(wtps["median"] if wtps else 0.0),
        )

    models.sort(key=_sort_key)

    return {
        "generated_at": _utc_now_iso(),
        "total_duration_ms": total_duration_ms,
        "repeats": repeats,
        "snippets_count": snippets_count,
        "gpu_vram_baseline_used_mb": gpu_vram_baseline_used_mb,
        "models": models,
    }


def _parse_model_flag(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--model must be 'client:model_name', got {s!r}"
        )
    client, _, model = s.partition(":")
    return client.strip().lower(), model.strip()


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    p = argparse.ArgumentParser(
        description="Speed benchmark for LLM clients (Ollama + Gemini)"
    )
    p.add_argument("--repeats", type=int, default=1,
                   help="Repetitions per (snippet, model). Default: 1")
    p.add_argument("--out", default="benchmarks.jsonl",
                   help="JSONL output path. Default: benchmarks.jsonl")
    p.add_argument("--summary", default="summary.json",
                   help="Summary JSON path. Default: summary.json")
    p.add_argument("--snippets", default="snippets.txt",
                   help="Snippets file. Default: snippets.txt")
    p.add_argument("--expected", default="expected_output.json",
                   help="Expected outputs JSON for accuracy scoring. "
                        "Default: expected_output.json. Pass empty string to disable.")
    p.add_argument("--model", type=_parse_model_flag, default=None,
                   help="Run only this one model, e.g. 'ollama:gemma4:e2b' "
                        "or 'gemini:gemini-2.5-flash'. Without this flag the "
                        "hardcoded MODELS list is used.")
    args = p.parse_args()

    snippets_path = Path(args.snippets)
    snippets = parse_snippets(snippets_path.read_text(encoding="utf-8"))
    snippet_items = list(snippets.items())
    if not snippet_items:
        print(f"No snippets found in {snippets_path}", file=sys.stderr)
        return 2

    expected: dict[str, list[dict]] = {}
    if args.expected:
        try:
            expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
            logger.info("Loaded expected outputs for %d snippets from %s",
                        len(expected), args.expected)
        except FileNotFoundError:
            logger.warning("Expected file %s not found — accuracy scoring disabled.",
                           args.expected)

    models = [args.model] if args.model else MODELS

    nvml = NVMLHelper()

    _unload_all_ollama(OLLAMA_HOST)
    baseline_mb = nvml.used_mb()
    if baseline_mb is not None:
        logger.info("GPU VRAM baseline (after unload): %d MB", baseline_mb)

    out_path = Path(args.out)
    summary_path = Path(args.summary)
    out_path.write_text("", encoding="utf-8")

    records: list[dict] = []
    out_fh = out_path.open("a", encoding="utf-8")

    t_bench_start = time.perf_counter()
    try:
        for client_type, model in models:
            if client_type == "ollama":
                _unload_all_ollama(OLLAMA_HOST)
            logger.info("=== %s : %s ===", client_type, model)
            try:
                client = get_llm_client(client_type, model)
            except Exception as exc:
                logger.error("Failed to construct client %s/%s: %s",
                             client_type, model, exc)
                err_rec = {
                    "timestamp": _utc_now_iso(),
                    "client": client_type,
                    "model": model,
                    "snippet_id": None,
                    "rep": 0,
                    "warmup": False,
                    "wall_duration_ms": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "load_duration_ns": None,
                    "prompt_eval_duration_ns": None,
                    "eval_duration_ns": None,
                    "total_duration_ns": None,
                    "prompt_eval_tps": None,
                    "eval_tps": None,
                    "wall_tps": None,
                    "gpu_vram_used_mb": None,
                    "gpu_vram_delta_mb": None,
                    "entries_extracted": 0,
                    "correct": None,
                    "error": f"client init failed: {type(exc).__name__}: {exc}",
                }
                out_fh.write(json.dumps(err_rec) + "\n")
                out_fh.flush()
                records.append(err_rec)
                continue

            gpu_vram_used = None
            gpu_vram_delta = None
            if client_type == "ollama":
                first_id, first_text = snippet_items[0]
                logger.info("warmup with %s ...", first_id)
                rec = run_one(
                    client, first_id, first_text, None,
                    client_type=client_type, model=model,
                    rep=0, warmup=True,
                    gpu_vram_used_mb=None, gpu_vram_delta_mb=None,
                )
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                records.append(rec)
                used = nvml.used_mb()
                if used is not None and baseline_mb is not None:
                    gpu_vram_used = used
                    gpu_vram_delta = used - baseline_mb
                    logger.info("VRAM after warmup: %d MB (delta %+d MB)",
                                used, gpu_vram_delta)

            for rep in range(args.repeats):
                for snippet_id, snippet_text in snippet_items:
                    logger.info("[rep %d] %s/%s :: %s",
                                rep, client_type, model, snippet_id)
                    rec = run_one(
                        client, snippet_id, snippet_text,
                        expected.get(snippet_id),
                        client_type=client_type, model=model,
                        rep=rep, warmup=False,
                        gpu_vram_used_mb=gpu_vram_used,
                        gpu_vram_delta_mb=gpu_vram_delta,
                    )
                    out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()
                    records.append(rec)
                    if rec["error"]:
                        logger.warning("  err: %s", rec["error"])
                    else:
                        correct_str = (
                            "OK" if rec["correct"] is True
                            else "MISS" if rec["correct"] is False
                            else "?"
                        )
                        logger.info(
                            "  [%s] wall=%sms prompt_tok=%s compl_tok=%s "
                            "eval_tps=%s wall_tps=%s",
                            correct_str,
                            rec["wall_duration_ms"], rec["prompt_tokens"],
                            rec["completion_tokens"], rec["eval_tps"],
                            rec["wall_tps"],
                        )
    finally:
        total_duration_ms = round((time.perf_counter() - t_bench_start) * 1000, 2)
        _unload_all_ollama(OLLAMA_HOST)
        out_fh.close()
        nvml.shutdown()

    summary = build_summary(
        records,
        repeats=args.repeats,
        snippets_count=len(snippet_items),
        gpu_vram_baseline_used_mb=baseline_mb,
        total_duration_ms=total_duration_ms,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. Wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
