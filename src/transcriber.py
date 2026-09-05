"""Speech-to-text transcription using faster-whisper.

Supports GPU (CUDA) and CPU modes, with JSON caching to avoid re-transcribing.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import override

# ---------------------------------------------------------------------------
# Graceful import — only crash when we actually try to use the model
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore[import-untyped]
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE_SUFFIX = "_transcripts_cache.json"


def _read_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _migrate_cache_keys(cache: dict[str, str], export_root: Path) -> dict[str, str]:
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


def _write_cache(cache_path: Path, data: dict[str, str]) -> None:
    cache_path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main transcriber class
# ---------------------------------------------------------------------------
class Transcriber:
    """Loads whisper model once, transcribes many audio files.

    Usage:
        t = Transcriber(model_size="medium", device="cuda")
        text = t.transcribe("voice.ogg")
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "auto",
        language: str = "ru",
        beam_size: int = 5,
        cpu_threads: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        if not HAS_WHISPER:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self._language = language
        self._beam_size = beam_size

        cpu_t = cpu_threads or max(1, mp.cpu_count() // 2)
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_t,
            num_workers=2,
        )

        # Batched pipeline for extra throughput on GPU
        self._use_batched = device == "cuda"
        if self._use_batched:
            self._batched = BatchedInferencePipeline(model=self._model)

        # Cache
        self._cache_path: Path | None = None
        self._cache: dict[str, str] = {}
        if cache_dir:
            cp = Path(cache_dir) / _CACHE_SUFFIX
            self._cache_path = cp
            self._cache = _migrate_cache_keys(_read_cache(cp), Path(cache_dir).resolve())

    # ------------------------------------------------------------------
    def transcribe(self, filepath: str | Path) -> str:
        """Transcribe a single audio/video file. Uses cache if available.

        Only updates the in-memory cache; flush_cache() persists it to disk.
        """
        fp = str(filepath)
        if fp in self._cache:
            return self._cache[fp]

        if self._use_batched:
            segments, _info = self._batched.transcribe(
                fp, language=self._language, beam_size=self._beam_size, batch_size=16
            )
        else:
            segments, _info = self._model.transcribe(
                fp, language=self._language, beam_size=self._beam_size, vad_filter=True
            )

        text = " ".join(seg.text.strip() for seg in segments)

        self._cache[fp] = text
        return text

    # ------------------------------------------------------------------
    def flush_cache(self) -> None:
        """Force-write cache to disk."""
        if self._cache_path:
            _write_cache(self._cache_path, self._cache)
