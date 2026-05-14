r"""Concurrency benchmark for Ollama: 4 configurations, fixed model & snippets.

Compares 4 strategies for serving the same workload (N requests, single Ollama
model `gemma4:e2b`) to find the best throughput on a single GPU:

    approach_1_single_seq        1 instance @ host_a, sequential
    approach_2_single_parallel   1 instance @ host_a, NUM_PARALLEL>=4, async client
    approach_3_dual_seq          2 instances @ host_a + host_b, sequential per instance
    approach_4_dual_parallel     2 instances + async, round-robin across hosts

Prerequisites — start Ollama instances manually before running.
CRITICAL on multi-GPU machines: set CUDA_VISIBLE_DEVICES=0 in BOTH terminals
so both instances share the SAME GPU. Without it, Ollama spreads across all
visible GPUs and the test no longer measures shared-card contention.

    # Terminal 1 (host_a)
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:OLLAMA_HOST = "127.0.0.1:11434"
    $env:OLLAMA_NUM_PARALLEL = "4"
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    ollama serve

    # Terminal 2 (host_b) — only needed for approach 3 & 4
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_NUM_PARALLEL = "4"
    $env:OLLAMA_MODELS = "$env:USERPROFILE\.ollama\models"
    ollama serve

NUM_PARALLEL=4 is set for both instances throughout; sequential behavior in
approach 1 & 3 is enforced client-side. Without concurrent client requests the
server has nothing to batch, so the configuration is equivalent.
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

from benchmark import NVMLHelper, _unload_all_ollama
from extractor import get_llm_client, parse_snippets, process_snippet

logger = logging.getLogger(__name__)


APPROACH_1 = "approach_1_single_seq"
APPROACH_2 = "approach_2_single_parallel"
APPROACH_3 = "approach_3_dual_seq"
APPROACH_4 = "approach_4_dual_parallel"

DEFAULT_HOST_A = "http://127.0.0.1:11434"
DEFAULT_HOST_B = "http://127.0.0.1:11435"
DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_REQUESTS = 100
DEFAULT_PARALLEL_WORKERS = 8
DEFAULT_COOLDOWN_S = 5


# Thread-local cache: one OllamaClient per (host, model) per worker thread.
# Avoids sharing `client.last_metrics` between threads, which would race.
_local = threading.local()


def _local_attr(host: str, model: str) -> str:
    safe = (host + "_" + model).replace(":", "_").replace(".", "_").replace("/", "_")
    return f"_client_{safe}"


def get_thread_client(host: str, model: str):
    attr = _local_attr(host, model)
    client = getattr(_local, attr, None)
    if client is None:
        client = get_llm_client("ollama", model, host=host)
        setattr(_local, attr, client)
    return client


class VRAMSampler:
    """Background thread sampling NVML VRAM at fixed interval."""

    def __init__(self, nvml: NVMLHelper, interval_s: float = 0.25) -> None:
        self.nvml = nvml
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            v = self.nvml.used_mb()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
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


def verify_ollama_alive(host: str, timeout_s: float = 3.0) -> None:
    """Raise RuntimeError if host's /api/version is not reachable."""
    url = host.rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{url} returned status {resp.status}")
            body = resp.read().decode("utf-8", errors="replace")
            logger.info("Ollama alive @ %s -> %s", host, body.strip())
    except Exception as exc:
        raise RuntimeError(
            f"Ollama not reachable at {host} ({exc}). "
            f"Start it with: $env:OLLAMA_HOST=\"{host.replace('http://', '')}\"; ollama serve"
        ) from exc


def run_request(
    *,
    approach_id: str,
    request_id: int,
    snippet_id: str,
    snippet_text: str,
    host: str,
    model: str,
    expected: list[dict] | None,
) -> dict:
    """Single timed request. Returns one JSONL record dict."""
    client = get_thread_client(host, model)
    t0 = time.perf_counter()
    error: str | None = None
    entries: list[dict] = []
    correct: bool | None = None
    try:
        entries = process_snippet(snippet_text, client, snippet_id=snippet_id)
        if expected is not None:
            correct = entries == expected
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s/req_%d] request failed", approach_id, request_id)
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)

    m = getattr(client, "last_metrics", None) or {}
    return {
        "timestamp": _utc_now_iso(),
        "approach_id": approach_id,
        "request_id": request_id,
        "snippet_id": snippet_id,
        "host": host,
        "model": model,
        "wall_duration_ms": wall_ms,
        "prompt_tokens": m.get("prompt_tokens"),
        "completion_tokens": m.get("completion_tokens"),
        "load_duration_ns": m.get("load_duration_ns"),
        "prompt_eval_duration_ns": m.get("prompt_eval_duration_ns"),
        "eval_duration_ns": m.get("eval_duration_ns"),
        "total_duration_ns": m.get("total_duration_ns"),
        "entries_extracted": len(entries),
        "correct": correct,
        "error": error,
    }


def run_approach_1(
    requests: list[tuple[str, str]], host: str, model: str, expected: dict[str, list[dict]]
) -> list[dict]:
    """Sequential, single instance."""
    records: list[dict] = []
    for i, (sid, text) in enumerate(requests):
        rec = run_request(
            approach_id=APPROACH_1, request_id=i, snippet_id=sid,
            snippet_text=text, host=host, model=model,
            expected=expected.get(sid),
        )
        records.append(rec)
        logger.info("[%s] %d/%d wall=%sms tokens=%s",
                    APPROACH_1, i + 1, len(requests),
                    rec["wall_duration_ms"], rec["completion_tokens"])
    return records


def run_approach_2(
    requests: list[tuple[str, str]], host: str, model: str,
    expected: dict[str, list[dict]], max_workers: int,
) -> list[dict]:
    """Parallel client, single instance."""
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_request,
                approach_id=APPROACH_2, request_id=i, snippet_id=sid,
                snippet_text=text, host=host, model=model,
                expected=expected.get(sid),
            ): i
            for i, (sid, text) in enumerate(requests)
        }
        for f in as_completed(futures):
            rec = f.result()
            records.append(rec)
            logger.info("[%s] %d/%d completed wall=%sms",
                        APPROACH_2, len(records), len(requests), rec["wall_duration_ms"])
    records.sort(key=lambda r: r["request_id"])
    return records


def run_approach_3(
    requests: list[tuple[str, str]], hosts: list[str], model: str,
    expected: dict[str, list[dict]],
) -> list[dict]:
    """Dual instance, sequential per instance, instances run in parallel."""
    mid = len(requests) // 2
    batches = [
        (0, requests[:mid], hosts[0]),
        (mid, requests[mid:], hosts[1]),
    ]

    def run_batch(start_idx: int, batch: list[tuple[str, str]], host: str) -> list[dict]:
        out: list[dict] = []
        for offset, (sid, text) in enumerate(batch):
            rec = run_request(
                approach_id=APPROACH_3, request_id=start_idx + offset,
                snippet_id=sid, snippet_text=text, host=host, model=model,
                expected=expected.get(sid),
            )
            out.append(rec)
            logger.info("[%s @ %s] %d/%d wall=%sms",
                        APPROACH_3, host, offset + 1, len(batch),
                        rec["wall_duration_ms"])
        return out

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_batch, s, b, h) for s, b, h in batches]
        for f in as_completed(futures):
            records.extend(f.result())
    records.sort(key=lambda r: r["request_id"])
    return records


def run_approach_4(
    requests: list[tuple[str, str]], hosts: list[str], model: str,
    expected: dict[str, list[dict]], max_workers: int,
) -> list[dict]:
    """Dual instance, parallel client, round-robin host assignment."""
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, (sid, text) in enumerate(requests):
            host = hosts[i % 2]
            futures.append(pool.submit(
                run_request,
                approach_id=APPROACH_4, request_id=i, snippet_id=sid,
                snippet_text=text, host=host, model=model,
                expected=expected.get(sid),
            ))
        for f in as_completed(futures):
            rec = f.result()
            records.append(rec)
            logger.info("[%s] %d/%d completed host=%s wall=%sms",
                        APPROACH_4, len(records), len(requests),
                        rec["host"].rsplit(":", 1)[-1], rec["wall_duration_ms"])
    records.sort(key=lambda r: r["request_id"])
    return records


def summarize_approach(
    approach_id: str, records: list[dict], wall_total_s: float,
    vram_samples: list[int], vram_baseline: int | None,
) -> dict:
    ok = [r for r in records if r["error"] is None]
    errors = len(records) - len(ok)
    latencies = [r["wall_duration_ms"] for r in ok]
    completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in ok)

    scored = [r for r in ok if r.get("correct") is not None]
    n_correct = sum(1 for r in scored if r["correct"])

    vram_peak = max(vram_samples) if vram_samples else None
    vram_mean = round(statistics.mean(vram_samples), 1) if vram_samples else None
    vram_delta = (
        vram_peak - vram_baseline
        if vram_peak is not None and vram_baseline is not None
        else None
    )

    throughput = round(len(ok) / wall_total_s, 3) if wall_total_s > 0 else None
    tps_completion = round(completion_tokens / wall_total_s, 1) if wall_total_s > 0 else None
    tps_total = round((prompt_tokens + completion_tokens) / wall_total_s, 1) if wall_total_s > 0 else None

    return {
        "approach_id": approach_id,
        "n_requests": len(records),
        "n_ok": len(ok),
        "n_errors": errors,
        "wall_total_s": round(wall_total_s, 3),
        "throughput_req_per_s": throughput,
        "tokens_per_sec_completion_aggregate": tps_completion,
        "tokens_per_sec_total_aggregate": tps_total,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "vram_mb": {
            "baseline": vram_baseline,
            "peak": vram_peak,
            "mean": vram_mean,
            "delta_peak": vram_delta,
        },
        "n_correct": n_correct if scored else None,
        "n_scored": len(scored),
        "correctness_pct": (
            round(100 * n_correct / len(scored), 1) if scored else None
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model name. Default: {DEFAULT_MODEL}")
    p.add_argument("--requests", type=int, default=DEFAULT_REQUESTS,
                   help=f"Requests per approach. Default: {DEFAULT_REQUESTS}")
    p.add_argument("--host-a", default=DEFAULT_HOST_A,
                   help=f"First Ollama host URL. Default: {DEFAULT_HOST_A}")
    p.add_argument("--host-b", default=DEFAULT_HOST_B,
                   help=f"Second Ollama host URL (approach 3 & 4). Default: {DEFAULT_HOST_B}")
    p.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS,
                   help=f"Client thread pool size PER OLLAMA INSTANCE. "
                        f"Approach 2 uses this verbatim ({DEFAULT_PARALLEL_WORKERS}); "
                        f"approach 4 multiplies by num_hosts (=2 by default, so "
                        f"{DEFAULT_PARALLEL_WORKERS * 2}). Default: {DEFAULT_PARALLEL_WORKERS}")
    p.add_argument("--cooldown-s", type=int, default=DEFAULT_COOLDOWN_S,
                   help=f"Sleep between approaches (let VRAM settle). "
                        f"Default: {DEFAULT_COOLDOWN_S}")
    p.add_argument("--approaches", default="1,2,3,4",
                   help="Comma-separated approach IDs to run. Default: 1,2,3,4")
    p.add_argument("--snippets", default="snippets.txt",
                   help="Snippets file. Default: snippets.txt")
    p.add_argument("--expected", default="expected_output.json",
                   help="Expected outputs JSON for correctness scoring. "
                        "Default: expected_output.json. Empty to disable.")
    p.add_argument("--out", default="results/benchmarks_parallel.jsonl",
                   help="JSONL output path. Default: results/benchmarks_parallel.jsonl")
    p.add_argument("--summary", default="results/summary_parallel.json",
                   help="Summary JSON path. Default: results/summary_parallel.json")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()

    # Parse and validate approach selection
    try:
        approaches_to_run = sorted({int(x.strip()) for x in args.approaches.split(",") if x.strip()})
    except ValueError:
        print(f"--approaches must be comma-separated ints, got {args.approaches!r}",
              file=sys.stderr)
        return 2
    if not approaches_to_run or any(a not in (1, 2, 3, 4) for a in approaches_to_run):
        print(f"--approaches must be subset of 1,2,3,4 (got {approaches_to_run})",
              file=sys.stderr)
        return 2

    # Load snippets and cycle to N
    snippets_path = Path(args.snippets)
    snippets = parse_snippets(snippets_path.read_text(encoding="utf-8"))
    snippet_items = list(snippets.items())
    if not snippet_items:
        print(f"No snippets found in {snippets_path}", file=sys.stderr)
        return 2
    requests = [snippet_items[i % len(snippet_items)] for i in range(args.requests)]
    logger.info("Loaded %d unique snippets, cycled to %d requests per approach",
                len(snippet_items), len(requests))

    # Load expected outputs
    expected: dict[str, list[dict]] = {}
    if args.expected:
        try:
            expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
            logger.info("Loaded expected outputs for %d snippets", len(expected))
        except FileNotFoundError:
            logger.warning("Expected file %s not found — correctness disabled.",
                           args.expected)

    hosts = [args.host_a, args.host_b]

    # Build plan
    plan: list[tuple[str, str, list[str]]] = []
    if 1 in approaches_to_run:
        plan.append((APPROACH_1, "seq", [hosts[0]]))
    if 2 in approaches_to_run:
        plan.append((APPROACH_2, "parallel", [hosts[0]]))
    if 3 in approaches_to_run:
        plan.append((APPROACH_3, "dual_seq", hosts))
    if 4 in approaches_to_run:
        plan.append((APPROACH_4, "dual_parallel", hosts))

    # Verify reachability for every host we'll need
    needed_hosts: set[str] = set()
    for _, _, hs in plan:
        needed_hosts.update(hs)
    for h in sorted(needed_hosts):
        verify_ollama_alive(h)

    nvml = NVMLHelper()
    out_path = Path(args.out)
    summary_path = Path(args.summary)
    out_path.write_text("", encoding="utf-8")  # clear

    summaries: list[dict] = []
    t_overall = time.perf_counter()
    out_fh = out_path.open("a", encoding="utf-8")
    try:
        for approach_id, mode, hs in plan:
            logger.info("=" * 60)
            logger.info("STARTING %s on hosts %s", approach_id, hs)

            # Clean VRAM baseline: unload model from each host that's about to be used
            for h in hs:
                _unload_all_ollama(h)
            time.sleep(1)

            # Warmup each instance (loads model into VRAM, primes KV cache)
            warmup_sid, warmup_text = snippet_items[0]
            for h in hs:
                logger.info("[%s] warmup on %s with %s", approach_id, h, warmup_sid)
                try:
                    client = get_thread_client(h, args.model)
                    process_snippet(warmup_text, client, snippet_id=warmup_sid)
                except Exception:
                    logger.exception("warmup failed on %s", h)
                    return 3

            # Establish post-warmup VRAM baseline
            vram_baseline = nvml.used_mb()
            logger.info("[%s] VRAM baseline (model loaded): %s MB",
                        approach_id, vram_baseline)

            sampler = VRAMSampler(nvml)
            sampler.start()
            t0 = time.perf_counter()
            try:
                if mode == "seq":
                    records = run_approach_1(requests, hs[0], args.model, expected)
                elif mode == "parallel":
                    records = run_approach_2(
                        requests, hs[0], args.model, expected, args.parallel_workers,
                    )
                elif mode == "dual_seq":
                    records = run_approach_3(requests, hs, args.model, expected)
                elif mode == "dual_parallel":
                    records = run_approach_4(
                        requests, hs, args.model, expected,
                        args.parallel_workers * len(hs),
                    )
                else:
                    raise RuntimeError(f"Unknown mode {mode}")
            finally:
                wall_total = time.perf_counter() - t0
                sampler.stop()

            # Flush records to JSONL in request order
            for rec in records:
                out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()

            summary = summarize_approach(
                approach_id, records, wall_total, sampler.samples, vram_baseline,
            )
            summaries.append(summary)

            lat_p95 = summary["latency_ms"]["p95"]
            corr = summary["correctness_pct"]
            logger.info(
                "[%s] DONE wall=%.2fs throughput=%s req/s "
                "p50=%sms p95=%sms p99=%sms correct=%s%% errors=%d",
                approach_id, wall_total,
                summary["throughput_req_per_s"],
                summary["latency_ms"]["p50"], lat_p95,
                summary["latency_ms"]["p99"], corr, summary["n_errors"],
            )

            if args.cooldown_s > 0:
                logger.info("cooldown %ds ...", args.cooldown_s)
                time.sleep(args.cooldown_s)
    finally:
        out_fh.close()
        # Best-effort cleanup of every host we may have touched
        for h in needed_hosts:
            try:
                _unload_all_ollama(h)
            except Exception:
                pass
        nvml.shutdown()

    summary_doc = {
        "generated_at": _utc_now_iso(),
        "model": args.model,
        "requests_per_approach": args.requests,
        "snippets_unique": len(snippet_items),
        "parallel_workers_per_instance": args.parallel_workers,
        "client_threads_by_approach": {
            APPROACH_1: 1,
            APPROACH_2: args.parallel_workers,
            APPROACH_3: 2,
            APPROACH_4: args.parallel_workers * 2,
        },
        "hosts": {"host_a": hosts[0], "host_b": hosts[1]},
        "total_duration_s": round(time.perf_counter() - t_overall, 3),
        "approaches": summaries,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
