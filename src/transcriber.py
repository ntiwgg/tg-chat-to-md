"""Speech-to-text transcription using faster-whisper.

Supports GPU (CUDA) and CPU modes, with JSON caching to avoid re-transcribing.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from .cache import CACHE_FILE_NAME, migrate_cache_keys, read_cache, write_cache

# ---------------------------------------------------------------------------
# Graceful import — only crash when we actually try to use the model
# ---------------------------------------------------------------------------
try:
    from faster_whisper import (  # type: ignore[import-untyped]
        BatchedInferencePipeline,
        WhisperModel,
    )
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


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
            cp = Path(cache_dir) / CACHE_FILE_NAME
            self._cache_path = cp
            self._cache = migrate_cache_keys(read_cache(cp), Path(cache_dir).resolve())

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
            write_cache(self._cache_path, self._cache)
