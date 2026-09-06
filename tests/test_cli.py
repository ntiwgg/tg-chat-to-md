"""Unit tests for the CLI surface and dead CPU-code removal.

Hermetic: no audio decoding, no model load, no network. Covers:
  - the real CLI parser accepts --device cpu;
  - the removed --workers flag and the never-existing --cpu flag are rejected
    with an argparse error (exit code 2), not silently accepted;
  - `telegram_to_md.py --help` exits 0 and advertises --device without
    mentioning --workers or --cpu;
  - src.transcriber keeps its public API (transcribe, flush_cache, HAS_WHISPER)
    and no longer defines the removed multiprocessing helpers;
  - --version exits 0 with a semver-ish string;
  - a missing result.json exits 1 with a friendly message, not a traceback;
  - a failing model load exits 1 with an actionable hint, not a traceback;
  - a per-file transcription failure is skipped with a warning and a summary
    line, and the run still produces Markdown.
"""

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def _make_voice_export(tmp_path: Path, n: int = 2) -> Path:
    """Synthetic export with n placeholder voice files; return its path."""
    export_dir = tmp_path / "ChatExport_synthetic"
    audio_dir = export_dir / "voice_messages"
    audio_dir.mkdir(parents=True)
    messages = []
    for i in range(1, n + 1):
        (audio_dir / f"voice_{i}.ogg").write_bytes(b"")
        messages.append(
            {
                "id": i,
                "type": "message",
                "date": f"2026-07-24T10:{i:02d}:00",
                "date_unixtime": "0",
                "from": "Автор",
                "text": "",
                "media_type": "voice_message",
                "file": f"voice_messages/voice_{i}.ogg",
                "duration_seconds": 5,
            }
        )
    (export_dir / "result.json").write_text(
        json.dumps({"name": "Тест", "id": 1, "messages": messages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return export_dir


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


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------
def test_version_exits_zero_and_prints_semver() -> None:
    proc = _run_cli("--version")

    assert proc.returncode == 0
    # "telegram_to_md.py 0.1.0" when installed, "0.0.0.dev0" when unpackaged
    assert re.fullmatch(r"telegram_to_md\.py \d+\.\d+\.\d+(\.dev\d+)?", proc.stdout.strip())


# ---------------------------------------------------------------------------
# friendly errors instead of tracebacks (subprocess: real script)
# ---------------------------------------------------------------------------
def test_missing_result_json_is_friendly(tmp_path) -> None:
    export_dir = tmp_path / "ChatExport_no_result"
    export_dir.mkdir()

    proc = _run_cli(str(export_dir))

    assert proc.returncode == 1
    assert "result.json" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


def test_export_dir_check_is_friendly() -> None:
    proc = _run_cli(GHOST_EXPORT)

    assert proc.returncode == 1
    assert "директория не найдена" in proc.stderr
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# model-load failure: exit 1 with actionable hint, no traceback (in-process)
# ---------------------------------------------------------------------------
def test_model_load_failure_is_friendly(tmp_path, monkeypatch, capsys) -> None:
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "telegram_to_md.py",
            str(export_dir),
            "--model",
            "tiny",
            "--device",
            "cuda",
            "--output",
            str(tmp_path / "out.md"),
        ],
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA library not found")

    monkeypatch.setattr(cli, "Transcriber", _boom)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "не удалось загрузить модель" in err
    assert "--device cpu" in err
    assert "Traceback" not in err
    assert not (tmp_path / "out.md").exists()  # nothing written after failure


# ---------------------------------------------------------------------------
# per-file tolerance: one bad file skips, rest transcribe, summary printed
# ---------------------------------------------------------------------------
def test_per_file_failure_skips_and_continues(tmp_path, monkeypatch, capsys) -> None:
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=2)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    # Drop the progress bar so test output stays clean (tqdm imported inside main).
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _FakeTranscriber:
        """No model: cache-empty stub that fails on exactly one file."""

        def __init__(self, *args, **kwargs) -> None:
            self._cache: dict[str, str] = {}
            self._cache_path: Path | None = None

        def transcribe(self, fp: str) -> str:
            if Path(fp).name == "voice_1.ogg":
                raise RuntimeError("ошибка декодера (тестовая)")
            return "текст второго файла"

        def flush_cache(self) -> None:
            pass

    monkeypatch.setattr(cli, "Transcriber", _FakeTranscriber)

    cli.main()  # must not raise and must not sys.exit

    md = output_path.read_text(encoding="utf-8")
    assert "> *Расшифровка:* текст второго файла" in md
    assert "> *(расшифровка недоступна)*" in md  # the failed file has no entry

    captured = capsys.readouterr()
    assert "Не удалось расшифровать" in captured.err
    assert "1 из 2" in captured.err  # failure summary line
    assert "Traceback" not in captured.err


def test_per_file_success_writes_no_failure_summary(tmp_path, monkeypatch, capsys) -> None:
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=1)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _FakeTranscriber:
        def __init__(self, *args, **kwargs) -> None:
            self._cache: dict[str, str] = {}
            self._cache_path: Path | None = None

        def transcribe(self, fp: str) -> str:
            return "единственная расшифровка"

        def flush_cache(self) -> None:
            pass

    monkeypatch.setattr(cli, "Transcriber", _FakeTranscriber)

    cli.main()

    md = output_path.read_text(encoding="utf-8")
    assert "> *Расшифровка:* единственная расшифровка" in md
    assert "недоступна" not in md
    err = capsys.readouterr().err
    assert "Не удалось расшифровать" not in err


# ---------------------------------------------------------------------------
# flush cadence: CACHE_FLUSH_EVERY shrunk via monkeypatch, real main() loop
# ---------------------------------------------------------------------------
def test_cache_flushes_every_n_files_and_on_exit(tmp_path, monkeypatch) -> None:
    """REGRESSION guard for the batching policy: the cache is persisted after
    every CACHE_FLUSH_EVERY files and once more in the finally block. Runs the
    real main() transcription loop in-process — no model, recording stub."""
    import telegram_to_md as cli

    n_files = 4
    export_dir = _make_voice_export(tmp_path, n=n_files)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(cli, "CACHE_FLUSH_EVERY", 2)
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _RecordingTranscriber:
        """Records every transcribe call and the transcribed-file snapshot
        at every flush_cache call (filesystem-effect assertion)."""

        def __init__(self, *args, **kwargs) -> None:
            self._cache: dict[str, str] = {}
            self._cache_path: Path | None = None
            self.transcribed: list[str] = []
            self.flush_snapshots: list[list[str]] = []

        def transcribe(self, fp: str) -> str:
            self.transcribed.append(fp)
            return f"текст {len(self.transcribed)}"

        def flush_cache(self) -> None:
            self.flush_snapshots.append(list(self.transcribed))

    recorder = _RecordingTranscriber()
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **kw: recorder)

    cli.main()

    assert len(recorder.transcribed) == n_files  # once per file
    # Cadence 2 with 4 files: flush after file 2, after file 4, then finally.
    assert [len(s) for s in recorder.flush_snapshots] == [2, 4, 4]
    # The markdown carries every transcript.
    md = output_path.read_text(encoding="utf-8")
    for i in range(1, n_files + 1):
        assert f"> *Расшифровка:* текст {i}" in md
