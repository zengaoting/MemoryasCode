from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def append_jsonl(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    _repair_truncated_jsonl_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _repair_truncated_jsonl_tail(path: Path) -> None:
    """Drop only an interrupted final record before a resumed append."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = stream.tell() - 1
        while position > 0:
            position -= 1
            stream.seek(position)
            if stream.read(1) == b"\n":
                stream.truncate(position + 1)
                logger.warning("Removed interrupted final JSONL record from %s", path)
                return
        stream.truncate(0)
        logger.warning("Removed interrupted only JSONL record from %s", path)


def read_jsonl(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", line_number, path)
    return rows


def write_jsonl(items: list[dict], path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def format_fact_line(fact: dict) -> str:
    """Format one fact exactly as it appears in the answer context."""
    duplicate_note = ""
    if fact.get("semantic_duplicate_count", 1) > 1:
        duplicate_note = (
            f" | semantic_duplicates={fact.get('semantic_duplicate_count')} "
            f"| source_ids={','.join(fact.get('source_fact_ids', []))}"
        )
    dialogue_ids = fact.get("dialogue_ids") or [
        fact.get("dia_id", fact.get("fact_id", "unknown"))
    ]
    return (
        f"- [{','.join(str(d) for d in dialogue_ids)} | "
        f"fact_id={fact.get('fact_id', 'unknown')} | "
        f"time={fact.get('normalized_time') or fact.get('time_text') or 'unknown'} | "
        f"speaker={fact.get('speaker') or 'unknown'}{duplicate_note}] "
        f"{fact.get('fact_text') or ''}"
    )


def estimate_text_tokens(text: str) -> int:
    """Return the repository's deterministic character-based token estimate."""
    return max(1, len(text) // 4) if text else 0
