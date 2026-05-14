"""Mixed-workload benchmark: embeddings + generation on one Ollama instance.

Tests how an embedding model (qwen3-embedding:4b) and a generation model
(gemma4:e2b) coexist on a single Ollama daemon. RAG-style real-world pattern.

Three modes (run independently):
    emb-only  : N_EMB embedding requests only
    gen-only  : N_GEN generation requests only (extraction pipeline)
    mixed     : N_EMB + N_GEN requests fired concurrently

Concurrency: single ThreadPoolExecutor with --workers threads, all requests
submitted at start. In `mixed`, embeddings will likely finish much faster than
generation (single forward pass vs. autoregressive), so the timeline starts
fully mixed then degenerates to gen-only tail.

Prerequisites: Ollama with `OLLAMA_MAX_LOADED_MODELS>=2` so both models can sit
in VRAM simultaneously. Both models pulled.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client as OllamaRawClient

from benchmark import NVMLHelper, _unload_all_ollama
from extractor import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    get_llm_client,
    parse_snippets,
    process_snippet,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:21434"
DEFAULT_GEN_MODEL = "gemma4:e2b"
DEFAULT_EMB_MODEL = "qwen3-embedding:4b"
DEFAULT_EMB_REQUESTS = 1000
DEFAULT_GEN_REQUESTS = 100
DEFAULT_WORKERS = 48

_local = threading.local()


def get_emb_client(host: str) -> OllamaRawClient:
    attr = "emb_client"
    client = getattr(_local, attr, None)
    if client is None:
        client = OllamaRawClient(host=host)
        setattr(_local, attr, client)
    return client


def get_gen_client(host: str, model: str):
    attr = f"gen_client_{model}".replace(":", "_").replace(".", "_")
    client = getattr(_local, attr, None)
    if client is None:
        client = get_llm_client("ollama", model, host=host)
        setattr(_local, attr, client)
    return client


class VRAMSampler:
    def __init__(self, nvml: NVMLHelper, interval_s: float = 0.25):
        self.nvml = nvml
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            v = self.nvml.used_mb()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return round(s[lo], 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def verify_alive(host: str, timeout_s: float = 3.0) -> None:
    url = host.rstrip("/") + "/api/version"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{url} returned {resp.status}")
        body = resp.read().decode("utf-8", errors="replace")
        logger.info("Ollama alive @ %s -> %s", host, body.strip())


def run_emb(*, request_id: int, text: str, host: str, model: str, mode: str, t_phase_start: float) -> dict:
    client = get_emb_client(host)
    start_offset_s = round(time.perf_counter() - t_phase_start, 3)
    t0 = time.perf_counter()
    error = None
    embedding_dim = None
    total_duration_ns = None
    load_duration_ns = None
    prompt_eval_count = None
    try:
        resp = client.embed(model=model, input=text)
        as_dict = dict(resp) if not isinstance(resp, dict) else resp
        embs = as_dict.get("embeddings") or []
        if embs:
            embedding_dim = len(embs[0])
        total_duration_ns = as_dict.get("total_duration")
        load_duration_ns = as_dict.get("load_duration")
        prompt_eval_count = as_dict.get("prompt_eval_count")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s/emb/%d] failed", mode, request_id)
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    end_offset_s = round(time.perf_counter() - t_phase_start, 3)
    return {
        "timestamp": _utc_now_iso(),
        "type": "emb",
        "mode": mode,
        "request_id": request_id,
        "model": model,
        "start_offset_s": start_offset_s,
        "end_offset_s": end_offset_s,
        "wall_duration_ms": wall_ms,
        "total_duration_ns": total_duration_ns,
        "load_duration_ns": load_duration_ns,
        "prompt_eval_count": prompt_eval_count,
        "embedding_dim": embedding_dim,
        "error": error,
    }


def run_gen(
    *,
    request_id: int,
    snippet_id: str,
    snippet_text: str,
    host: str,
    model: str,
    mode: str,
    expected: list[dict] | None,
    t_phase_start: float,
) -> dict:
    client = get_gen_client(host, model)
    start_offset_s = round(time.perf_counter() - t_phase_start, 3)
    t0 = time.perf_counter()
    error = None
    entries: list[dict] = []
    correct = None
    try:
        entries = process_snippet(snippet_text, client, snippet_id=snippet_id)
        if expected is not None:
            correct = entries == expected
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s/gen/%d] failed", mode, request_id)
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    end_offset_s = round(time.perf_counter() - t_phase_start, 3)
    m = getattr(client, "last_metrics", None) or {}
    return {
        "timestamp": _utc_now_iso(),
        "type": "gen",
        "mode": mode,
        "request_id": request_id,
        "snippet_id": snippet_id,
        "model": model,
        "start_offset_s": start_offset_s,
        "end_offset_s": end_offset_s,
        "wall_duration_ms": wall_ms,
        "prompt_tokens": m.get("prompt_tokens"),
        "completion_tokens": m.get("completion_tokens"),
        "eval_duration_ns": m.get("eval_duration_ns"),
        "prompt_eval_duration_ns": m.get("prompt_eval_duration_ns"),
        "total_duration_ns": m.get("total_duration_ns"),
        "entries_extracted": len(entries),
        "correct": correct,
        "error": error,
    }


def run_phase(
    *,
    mode: str,
    snippets: list[tuple[str, str]],
    expected: dict[str, list[dict]],
    n_emb: int,
    n_gen: int,
    host: str,
    emb_model: str,
    gen_model: str,
    workers: int,
    nvml: NVMLHelper,
) -> tuple[list[dict], dict]:
    """Run a single phase (emb-only / gen-only / mixed). Returns records + summary."""
    logger.info("=" * 70)
    logger.info("PHASE: %s | n_emb=%d n_gen=%d workers=%d", mode, n_emb, n_gen, workers)

    # Warmup BOTH models regardless of mode — we want identical VRAM footprint
    # (both models resident) across all phases so they're directly comparable.
    warmup_sid, warmup_text = snippets[0]
    logger.info("[%s] warmup emb on %s", mode, host)
    get_emb_client(host).embed(model=emb_model, input=warmup_text)
    logger.info("[%s] warmup gen on %s", mode, host)
    process_snippet(warmup_text, get_gen_client(host, gen_model), snippet_id=warmup_sid)

    vram_baseline = nvml.used_mb()
    logger.info("[%s] VRAM baseline (models loaded): %s MB", mode, vram_baseline)

    sampler = VRAMSampler(nvml)
    sampler.start()

    records: list[dict] = []
    t0 = time.perf_counter()

    # For mixed mode: TWO SEPARATE pools, started at the same instant, so gen and
    # emb actually run concurrently from t=0 (not gen-after-emb-queue-clears).
    # For pure modes: one pool suffices.
    try:
        if n_emb > 0 and n_gen > 0:
            # MIXED: 2 pools, parallel start
            with ThreadPoolExecutor(max_workers=max(1, workers // 2)) as emb_pool, \
                 ThreadPoolExecutor(max_workers=max(1, workers // 2)) as gen_pool:
                futures = []
                for i in range(n_emb):
                    sid, text = snippets[i % len(snippets)]
                    futures.append(emb_pool.submit(
                        run_emb, request_id=i, text=text, host=host,
                        model=emb_model, mode=mode, t_phase_start=t0,
                    ))
                for i in range(n_gen):
                    sid, text = snippets[i % len(snippets)]
                    futures.append(gen_pool.submit(
                        run_gen, request_id=i, snippet_id=sid,
                        snippet_text=text, host=host, model=gen_model,
                        mode=mode, expected=expected.get(sid),
                        t_phase_start=t0,
                    ))
                n_done = 0
                for f in as_completed(futures):
                    rec = f.result()
                    records.append(rec)
                    n_done += 1
                    if n_done % 50 == 0 or n_done == len(futures):
                        logger.info("[%s] %d/%d completed", mode, n_done, len(futures))
        else:
            # PURE: single pool
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = []
                for i in range(n_emb):
                    sid, text = snippets[i % len(snippets)]
                    futures.append(pool.submit(
                        run_emb, request_id=i, text=text, host=host,
                        model=emb_model, mode=mode, t_phase_start=t0,
                    ))
                for i in range(n_gen):
                    sid, text = snippets[i % len(snippets)]
                    futures.append(pool.submit(
                        run_gen, request_id=i, snippet_id=sid,
                        snippet_text=text, host=host, model=gen_model,
                        mode=mode, expected=expected.get(sid),
                        t_phase_start=t0,
                    ))
                n_done = 0
                for f in as_completed(futures):
                    rec = f.result()
                    records.append(rec)
                    n_done += 1
                    if n_done % 50 == 0 or n_done == len(futures):
                        logger.info("[%s] %d/%d completed", mode, n_done, len(futures))
    finally:
        wall_total = time.perf_counter() - t0
        sampler.stop()

    # Aggregate per type
    summary_by_type: dict[str, dict] = {}
    for t in ("emb", "gen"):
        subset = [r for r in records if r["type"] == t]
        if not subset:
            continue
        ok = [r for r in subset if r["error"] is None]
        latencies = [r["wall_duration_ms"] for r in ok]
        sub = {
            "n_requests": len(subset),
            "n_ok": len(ok),
            "n_errors": len(subset) - len(ok),
            "throughput_req_per_s": (
                round(len(ok) / wall_total, 3) if wall_total > 0 else None
            ),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1) if latencies else None,
                "median": round(statistics.median(latencies), 1) if latencies else None,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
                "min": round(min(latencies), 1) if latencies else None,
                "max": round(max(latencies), 1) if latencies else None,
            },
        }
        if t == "gen":
            scored = [r for r in ok if r.get("correct") is not None]
            n_correct = sum(1 for r in scored if r["correct"])
            sub["correctness_pct"] = (
                round(100 * n_correct / len(scored), 1) if scored else None
            )
            completion = sum(r.get("completion_tokens") or 0 for r in ok)
            sub["tokens_per_sec_completion_aggregate"] = (
                round(completion / wall_total, 1) if wall_total > 0 else None
            )
        elif t == "emb":
            tokens_in = sum(r.get("prompt_eval_count") or 0 for r in ok)
            sub["tokens_per_sec_input_aggregate"] = (
                round(tokens_in / wall_total, 1) if wall_total > 0 else None
            )
            dims = [r["embedding_dim"] for r in ok if r.get("embedding_dim")]
            sub["embedding_dim"] = dims[0] if dims else None
        summary_by_type[t] = sub

    vram_peak = max(sampler.samples) if sampler.samples else None
    vram_mean = round(statistics.mean(sampler.samples), 1) if sampler.samples else None

    summary = {
        "mode": mode,
        "wall_total_s": round(wall_total, 3),
        "vram_mb": {
            "baseline": vram_baseline,
            "peak": vram_peak,
            "mean": vram_mean,
            "delta_peak": (
                vram_peak - vram_baseline
                if vram_peak is not None and vram_baseline is not None
                else None
            ),
        },
        "by_type": summary_by_type,
    }
    return records, summary


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--emb-model", default=DEFAULT_EMB_MODEL)
    p.add_argument("--n-emb", type=int, default=DEFAULT_EMB_REQUESTS)
    p.add_argument("--n-gen", type=int, default=DEFAULT_GEN_REQUESTS)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--modes", default="emb-only,gen-only,mixed",
                   help="Comma-separated list. Default: emb-only,gen-only,mixed")
    p.add_argument("--snippets", default="snippets.txt")
    p.add_argument("--expected", default="expected_output.json")
    p.add_argument("--out", default="results/benchmarks_mixed.jsonl")
    p.add_argument("--summary", default="results/summary_mixed.json")
    p.add_argument("--cooldown-s", type=int, default=10)
    args = p.parse_args()

    snippets = parse_snippets(Path(args.snippets).read_text(encoding="utf-8"))
    snippet_items = list(snippets.items())
    expected: dict[str, list[dict]] = {}
    if args.expected:
        try:
            expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Expected file %s not found", args.expected)

    verify_alive(args.host)
    nvml = NVMLHelper()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"emb-only", "gen-only", "mixed"}
    if any(m not in valid for m in modes):
        print(f"Invalid mode in {modes}; valid: {valid}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.write_text("", encoding="utf-8")
    out_fh = out_path.open("a", encoding="utf-8")

    phase_summaries: list[dict] = []
    t_overall = time.perf_counter()
    try:
        for mode in modes:
            if mode == "emb-only":
                n_emb_run, n_gen_run = args.n_emb, 0
            elif mode == "gen-only":
                n_emb_run, n_gen_run = 0, args.n_gen
            else:  # mixed
                n_emb_run, n_gen_run = args.n_emb, args.n_gen

            records, summary = run_phase(
                mode=mode,
                snippets=snippet_items,
                expected=expected,
                n_emb=n_emb_run,
                n_gen=n_gen_run,
                host=args.host,
                emb_model=args.emb_model,
                gen_model=args.gen_model,
                workers=args.workers,
                nvml=nvml,
            )

            records.sort(key=lambda r: (r["type"], r["request_id"]))
            for rec in records:
                out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            phase_summaries.append(summary)

            logger.info(
                "[%s] DONE wall=%.2fs emb_throughput=%s gen_throughput=%s",
                mode,
                summary["wall_total_s"],
                summary["by_type"].get("emb", {}).get("throughput_req_per_s"),
                summary["by_type"].get("gen", {}).get("throughput_req_per_s"),
            )

            if args.cooldown_s > 0 and mode != modes[-1]:
                logger.info("cooldown %ds ...", args.cooldown_s)
                time.sleep(args.cooldown_s)
    finally:
        out_fh.close()
        try:
            _unload_all_ollama(args.host)
        except Exception:
            pass
        nvml.shutdown()

    summary_doc = {
        "generated_at": _utc_now_iso(),
        "host": args.host,
        "gen_model": args.gen_model,
        "emb_model": args.emb_model,
        "n_emb": args.n_emb,
        "n_gen": args.n_gen,
        "workers": args.workers,
        "total_duration_s": round(time.perf_counter() - t_overall, 3),
        "phases": phase_summaries,
    }
    Path(args.summary).write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", args.out, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
