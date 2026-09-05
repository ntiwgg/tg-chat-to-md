"""Unit tests for the CLI surface and dead CPU-code removal.

Hermetic: no audio decoding, no model load, no network. Covers:
  - the real CLI parser accepts --device cpu;
  - the removed --workers flag and the never-existing --cpu flag are rejected
    with an argparse error (exit code 2), not silently accepted;
  - `telegram_to_md.py --help` exits 0 and advertises --device without
    mentioning --workers or --cpu;
  - src.transcriber keeps its public API (transcribe, flush_cache, HAS_WHISPER)
    and no longer defines the removed multiprocessing helpers.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import src.transcriber as transcriber_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "telegram_to_md.py"
GHOST_EXPORT = "ChatExport_ghost/"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the real CLI script in a subprocess from the repo root."""
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# argparse surface: real flags accepted, dead flags rejected
# ---------------------------------------------------------------------------
def test_device_cpu_parses_fine() -> None:
    proc = _run_cli(GHOST_EXPORT, "--device", "cpu")

    # Exit 1 (not argparse's 2): parse_args() succeeded, then main() failed on
    # the missing export dir — the first runtime check after parsing.
    assert proc.returncode == 1
    assert "директория не найдена" in proc.stderr


def test_workers_flag_is_rejected() -> None:
    proc = _run_cli(GHOST_EXPORT, "--workers", "8")

    assert proc.returncode == 2
    assert "unrecognized arguments: --workers 8" in proc.stderr


def test_cpu_flag_is_rejected() -> None:
    proc = _run_cli(GHOST_EXPORT, "--cpu")

    assert proc.returncode == 2
    assert "unrecognized arguments: --cpu" in proc.stderr


def test_help_exits_zero_and_describes_real_flags() -> None:
    proc = _run_cli("--help")

    assert proc.returncode == 0
    assert "--device" in proc.stdout
    assert "--workers" not in proc.stdout
    assert "--cpu" not in proc.stdout
    assert "--no-cache" in proc.stdout


# ---------------------------------------------------------------------------
# src.transcriber: dead multiprocessing API removed, public API intact
# ---------------------------------------------------------------------------
def test_transcriber_no_longer_has_removed_names() -> None:
    assert not hasattr(transcriber_mod, "transcribe_all")
    assert not hasattr(transcriber_mod, "transcribe_parallel_cpu")
    assert not hasattr(transcriber_mod, "_worker_transcribe")
    assert not hasattr(transcriber_mod.Transcriber, "transcribe_all")

    source = inspect.getsource(transcriber_mod)
    assert "transcribe_all" not in source
    assert "transcribe_parallel_cpu" not in source
    assert "_worker_transcribe" not in source
    assert "CPU-only parallel pool" not in source


def test_transcriber_public_api_stays_intact() -> None:
    assert hasattr(transcriber_mod, "HAS_WHISPER")
    assert isinstance(transcriber_mod.HAS_WHISPER, bool)
    assert callable(transcriber_mod.Transcriber.transcribe)
    assert callable(transcriber_mod.Transcriber.flush_cache)
