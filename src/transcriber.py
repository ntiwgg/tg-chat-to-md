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


def _write_cache(cache_path: Path, data: dict[str, str]) -> None:
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main transcriber class
# ---------------------------------------------------------------------------
class Transcriber:
    """Loads whisper model once, transcribes many audio files.

    Usage:
        t = Transcriber(model_size="medium", device="cuda")
        text = t.transcribe("voice.ogg")
        t.transcribe_all(["a.ogg", "b.mp4"])
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
            self._cache = _read_cache(cp)

    # ------------------------------------------------------------------
    def transcribe(self, filepath: str | Path) -> str:
        """Transcribe a single audio/video file. Uses cache if available."""
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
        if self._cache_path:
            _write_cache(self._cache_path, self._cache)

        return text

    # ------------------------------------------------------------------
    def transcribe_all(self, filepaths: list[str | Path]) -> dict[str, str]:
        """Transcribe a batch of files, using cache where possible."""
        results: dict[str, str] = {}
        for fp in filepaths:
            results[str(fp)] = self.transcribe(fp)
        return results

    # ------------------------------------------------------------------
    def flush_cache(self) -> None:
        """Force-write cache to disk."""
        if self._cache_path:
            _write_cache(self._cache_path, self._cache)


# ---------------------------------------------------------------------------
# CPU-only parallel pool (multiprocessing)
# ---------------------------------------------------------------------------
def _worker_transcribe(args: tuple[str, str, str, str, int, int]) -> tuple[str, str]:
    """Picklable worker for multiprocessing Pool."""
    filepath, model_size, device, compute_type, beam_size, language = args  # noqa: F821 — args are unpacked
    # Re-import inside worker (fresh process)
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        filepath, language=language, beam_size=beam_size, vad_filter=True
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return filepath, text


def transcribe_parallel_cpu(
    filepaths: list[str],
    model_size: str = "medium",
    language: str = "ru",
    beam_size: int = 5,
    num_workers: int | None = None,
) -> dict[str, str]:
    """Transcribe files in parallel using multiple CPU processes.

    Each worker loads its own model — high RAM usage but max throughput.
    """
    if not HAS_WHISPER:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

    workers = num_workers or max(1, mp.cpu_count() - 1)
    compute_type = "int8"
    device = "cpu"

    tasks = [(fp, model_size, device, compute_type, beam_size, language) for fp in filepaths]

    results: dict[str, str] = {}
    with mp.Pool(processes=workers) as pool:
        for filepath, text in pool.imap_unordered(_worker_transcribe, tasks):
            results[filepath] = text

    return results
