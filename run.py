"""Entry point: read snippets.txt, run extraction, write output.json."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from extractor import get_client, parse_snippets, process_snippet


HERE = Path(__file__).parent
logger = logging.getLogger("run")

PENDING: list | None = None


def setup_logging(log_path: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def load_or_init_state(output_path: Path, snippet_ids: list[str], resume: bool) -> dict:
    """Return a {snippet_id: list | None} state dict.

    - resume=False: every id starts as PENDING (None) regardless of any existing file.
    - resume=True:  load existing file (if present); fill in any missing ids with PENDING.
    """
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return {sid: existing.get(sid, PENDING) for sid in snippet_ids}
    return {sid: PENDING for sid in snippet_ids}


def write_state_atomic(output_path: Path, state: dict) -> None:
    """Write the full state JSON via tmp + rename to avoid torn files on crash."""
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)


def finalize_state(state: dict) -> dict:
    """Replace any remaining PENDING entries with [] for the final on-disk format."""
    return {sid: ([] if v is PENDING else v) for sid, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CO2 emissions from snippets.")
    parser.add_argument("--snippets", default="snippets.txt", help="Input file.")
    parser.add_argument("--output", default="output.json", help="Output JSON (also state file during run).")
    parser.add_argument("--log", default="extractor.log", help="Log file path.")
    parser.add_argument("--resume", action="store_true", help="Skip snippets already processed in the existing output file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    snippets_path = HERE / args.snippets
    output_path = HERE / args.output
    log_path = HERE / args.log

    setup_logging(log_path, args.verbose)
    logger.info("Starting extraction (resume=%s)", args.resume)

    snippets = parse_snippets(snippets_path.read_text(encoding="utf-8"))
    logger.info("Parsed %d snippets from %s", len(snippets), snippets_path.name)

    state = load_or_init_state(output_path, list(snippets.keys()), args.resume)
    write_state_atomic(output_path, state)  # persist initial pending state

    pending_ids = [sid for sid, v in state.items() if v is PENDING]
    skipped = len(snippets) - len(pending_ids)
    logger.info("Processing %d snippets (skipping %d already done)", len(pending_ids), skipped)

    if pending_ids:
        client = get_client()
        for sid in pending_ids:
            logger.info("Processing %s", sid)
            try:
                entries = process_snippet(snippets[sid], client, snippet_id=sid)
            except Exception:
                logger.exception("Failed to process %s — leaving as pending (re-run with --resume to retry)", sid)
                continue
            state[sid] = entries
            write_state_atomic(output_path, state)
            logger.info("%s done: %d entries extracted", sid, len(entries))

    final = finalize_state(state)
    write_state_atomic(output_path, final)
    logger.info("Wrote final output to %s (%d snippets)", output_path.name, len(final))


if __name__ == "__main__":
    main()
