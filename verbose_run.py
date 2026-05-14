"""Mimic `ollama run --verbose` from Python, with optional --schema flag.

Usage:
    python verbose_run.py test_prompt.txt           # bez schema (jak ollama run --verbose)
    python verbose_run.py test_prompt.txt --schema  # z naszym ExtractionResult JSON schema
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ollama

from extractor import ExtractionResult


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt_file", help="Path to prompt text file")
    p.add_argument("--schema", action="store_true",
                   help="Use ExtractionResult JSON schema as format= constraint")
    p.add_argument("--model", default="gemma4:e2b")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    args = p.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    client = ollama.Client(host=args.host)

    kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0},
    }
    if args.schema:
        kwargs["format"] = ExtractionResult.model_json_schema()

    print(f"--- model: {args.model}  schema: {args.schema}  ---", file=sys.stderr)
    resp = client.chat(**kwargs)

    # Print model output to stdout (like ollama run does)
    print(resp["message"]["content"])

    # Print verbose metrics to stderr (matches ollama run --verbose format)
    total_s = resp["total_duration"] / 1e9
    load_s = (resp.get("load_duration") or 0) / 1e9
    pe_count = resp.get("prompt_eval_count", 0)
    pe_s = (resp.get("prompt_eval_duration") or 0) / 1e9
    eval_count = resp.get("eval_count", 0)
    eval_s = (resp.get("eval_duration") or 0) / 1e9
    pe_rate = pe_count / pe_s if pe_s > 0 else 0
    eval_rate = eval_count / eval_s if eval_s > 0 else 0

    print(file=sys.stderr)
    print(f"total duration:       {total_s:.4f}s", file=sys.stderr)
    print(f"load duration:        {load_s*1000:.4f}ms", file=sys.stderr)
    print(f"prompt eval count:    {pe_count} token(s)", file=sys.stderr)
    print(f"prompt eval duration: {pe_s*1000:.4f}ms", file=sys.stderr)
    print(f"prompt eval rate:     {pe_rate:.2f} tokens/s", file=sys.stderr)
    print(f"eval count:           {eval_count} token(s)", file=sys.stderr)
    print(f"eval duration:        {eval_s:.4f}s", file=sys.stderr)
    print(f"eval rate:            {eval_rate:.2f} tokens/s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
