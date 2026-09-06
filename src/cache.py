"""Shared JSON transcript-cache helpers.

Both the transcriber and merge_exports work with per-export transcript
caches (<export_dir>/_transcripts_cache.json). This module owns the one
implementation of how a cache is read, written, and re-keyed, so both tools
cannot drift apart again (a past bug: merge_exports parsed cache files with
different corrupt-JSON semantics and merged raw legacy keys, silently losing
transcripts).

Cache keys are canonical absolute media paths — the same strings parser's
_resolve_file() stores in Message.file — so a cache travels with its export
directory and survives moves and CWD changes.

Behavior choice (documented, shared by all callers): a missing or corrupt
cache file degrades to an empty cache with a warning on stderr, never an
exception. Transcription is loss-tolerant by design: a broken cache only
means some files are re-transcribed or show the "no transcript" placeholder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE_FILE_NAME = "_transcripts_cache.json"


def read_cache(cache_path: Path) -> dict[str, str]:
    """Read a cache file; empty dict when the file is missing or corrupt.

    A corrupt file is reported on stderr (so the user knows transcripts will
    be missing) but treated as empty — the caller can rebuild it on flush.
    """
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"⚠ Кэш расшифровок повреждён, начинаю с пустого: {cache_path} ({exc})",
            file=sys.stderr,
        )
        return {}
    return data


def write_cache(cache_path: Path, data: dict[str, str]) -> None:
    """Persist the cache as compact single-line JSON (utf-8)."""
    cache_path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def migrate_cache_keys(cache: dict[str, str], export_root: Path) -> dict[str, str]:
    """Re-key legacy cache entries to canonical absolute paths under export_root.

    Legacy keys were export-relative paths as composed from the CLI argument
    (e.g. "ChatExport_x/voice_messages/audio_1.ogg"), so they change with the
    CWD and break when the folder moves. Relative keys are matched by dropping
    leading path components until the remainder points to an existing file; the
    entry is then re-keyed to that file's canonical absolute path. Absolute
    keys are kept verbatim. Entries that cannot be matched to an existing file
    are dropped — they would be re-transcribed anyway. Pure function: the input
    dict is not modified.
    """
    root = Path(export_root).resolve()
    migrated: dict[str, str] = {}
    for key, text in cache.items():
        key_path = Path(key)
        if key_path.is_absolute():
            migrated[key] = text
            continue
        parts = key_path.parts
        for i in range(len(parts)):
            candidate = root.joinpath(*parts[i:])
            if candidate.exists():
                migrated[str(candidate.resolve())] = text
                break
    return migrated
