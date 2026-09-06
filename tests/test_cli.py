"""Unit tests for the CLI surface and orchestration.

Hermetic: no audio decoding, no model load, no network. Covers:
  - the real CLI parser accepts --device cpu;
  - removed/dead flags (--workers, --cpu) and invalid --model/--device
    choices are rejected with an argparse error (exit code 2);
  - `telegram_to_md.py --help` exits 0 and advertises --device without
    mentioning --workers or --cpu;
  - src.transcriber imports cleanly and keeps its CLI seam (transcribe,
    flush_cache);
  - --version exits 0 with a semver-ish string;
  - a missing result.json exits 1 with a friendly message, not a traceback;
  - --no-cache wiring: cache_dir=None vs the export dir, one transcribe call
    per file (recording stub, in-process main);
  - flush cadence via CACHE_FLUSH_EVERY (see below);
  - KeyboardInterrupt durability: cache flushed, exit 1, no output written;
  - a no-media export never constructs a Transcriber and still writes md;
  - a failing model load exits 1 with an actionable hint, not a traceback;
  - a per-file transcription failure is skipped with a warning and a summary
    line, and the run still produces Markdown.
"""

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
# Transcriber stub seam
# ---------------------------------------------------------------------------
class _CacheAwareTranscriber:
    """Shared fake for the CLI's Transcriber seam: an in-memory cache plus the
    public stats API (cached_count, missing_from_cache) that
    telegram_to_md.main drives. Subclasses add call recording / faults."""

    def __init__(self, *args, **kwargs) -> None:
        self._cache: dict[str, str] = {}
        self._cache_path: Path | None = None

    def cached_count(self, filepaths: list[str]) -> int:
        return sum(1 for fp in filepaths if fp in self._cache)

    def missing_from_cache(self, filepaths: list[str]) -> list[str]:
        return [fp for fp in filepaths if fp not in self._cache]


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


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--model", "bogus"), ("--device", "bogus")],
)
def test_invalid_choice_is_rejected_with_exit_2(flag, value) -> None:
    proc = _run_cli(GHOST_EXPORT, flag, value)

    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr


def test_help_exits_zero_and_describes_real_flags() -> None:
    proc = _run_cli("--help")

    assert proc.returncode == 0
    assert "--device" in proc.stdout
    assert "--workers" not in proc.stdout
    assert "--cpu" not in proc.stdout
    assert "--no-cache" in proc.stdout


# ---------------------------------------------------------------------------
# src.transcriber: import smoke + the CLI duck-contract seam
# ---------------------------------------------------------------------------
def test_transcriber_import_smoke_and_cli_seam() -> None:
    """The import at the top of this file is the smoke check (the module must
    load without faster-whisper installed — every subprocess CLI test below
    relies on it). The class-level callable check documents the seam the
    in-process tests stub: telegram_to_md.main drives instances through
    transcribe(filepath), flush_cache(), cached_count(...) and
    missing_from_cache(...)."""
    assert callable(transcriber_mod.Transcriber.transcribe)
    assert callable(transcriber_mod.Transcriber.flush_cache)
    assert callable(transcriber_mod.Transcriber.cached_count)
    assert callable(transcriber_mod.Transcriber.missing_from_cache)


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
    # The actionable hint survives: tell the user which folder shape to pass.
    assert "Укажите путь к папке экспорта" in proc.stderr


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

    class _FakeTranscriber(_CacheAwareTranscriber):
        """No model: cache-empty stub that fails on exactly one file."""

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

    class _FakeTranscriber(_CacheAwareTranscriber):
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

    class _RecordingTranscriber(_CacheAwareTranscriber):
        """Records every transcribe call and the transcribed-file snapshot
        at every flush_cache call (filesystem-effect assertion)."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
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


def test_no_cache_flag_wiring_and_one_transcribe_call_per_file(
    tmp_path, monkeypatch
) -> None:
    """--no-cache must reach Transcriber as cache_dir=None; without the flag
    the export dir is the cache dir. Transcribe runs exactly once per file."""
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=2)
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)
    instances: list = []

    class _WiringTranscriber(_CacheAwareTranscriber):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cache_dir = kwargs.get("cache_dir")
            self.calls = 0
            instances.append(self)

        def transcribe(self, fp: str) -> str:
            self.calls += 1
            return f"текст {self.calls}"

        def flush_cache(self) -> None:
            pass

    monkeypatch.setattr(cli, "Transcriber", _WiringTranscriber)

    # Default: the export directory is the cache directory.
    with_cache = tmp_path / "with_cache.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(with_cache)],
    )
    cli.main()
    assert instances[-1].cache_dir == export_dir

    # --no-cache: cache_dir=None, nothing may be read or written on disk.
    no_cache = tmp_path / "no_cache.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--no-cache", "--output", str(no_cache)],
    )
    cli.main()
    assert instances[-1].cache_dir is None

    for instance in instances:
        assert instance.calls == 2  # one transcribe call per media file


def test_cache_hits_are_counted_and_not_re_transcribed(tmp_path, monkeypatch, capsys) -> None:
    """Cache-hit reporting flows through the public stats API: a file already
    in the cache is counted on stdout ("📦 … уже в кэше"), reported as missing
    by the count of the "Расшифровываю" line, and its cached text lands in the
    markdown without a second transcribe() call."""
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=2)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    cached_file = str((export_dir / "voice_messages" / "voice_1.ogg").resolve())

    class _CachingTranscriber(_CacheAwareTranscriber):
        """Mirrors the real short-circuit: cache hits never reach the model."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.transcribed: list[str] = []

        def transcribe(self, fp: str) -> str:
            if fp in self._cache:
                return self._cache[fp]
            self.transcribed.append(fp)
            return "новая расшифровка"

        def flush_cache(self) -> None:
            pass

    stub = _CachingTranscriber()
    stub._cache = {cached_file: "расшифровка из кэша"}
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **kw: stub)

    cli.main()

    assert stub.transcribed == [str((export_dir / "voice_messages" / "voice_2.ogg").resolve())]
    out = capsys.readouterr().out
    assert "1 файлов уже в кэше" in out
    assert "🎤 Расшифровываю 1 файлов…" in out
    md = output_path.read_text(encoding="utf-8")
    assert "> *Расшифровка:* расшифровка из кэша" in md
    assert "> *Расшифровка:* новая расшифровка" in md


def test_keyboard_interrupt_flushes_cache_and_exits_1(tmp_path, monkeypatch) -> None:
    """Ctrl-C mid-run must persist the partial cache (durability) and exit 1
    without writing the markdown."""
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=3)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _InterruptingTranscriber(_CacheAwareTranscriber):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.transcribed = 0
            self.flushes = 0

        def transcribe(self, fp: str) -> str:
            self.transcribed += 1
            if self.transcribed == 2:
                raise KeyboardInterrupt
            return "текст"

        def flush_cache(self) -> None:
            self.flushes += 1

    stub = _InterruptingTranscriber()
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **kw: stub)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert stub.transcribed == 2  # interrupted on the second file
    # Partial progress was persisted before exit (the interrupt handler flushes;
    # sys.exit then still passes through the finally flush — both are benign,
    # idempotent writes, so we assert durability, not an exact flush count).
    assert stub.flushes >= 1
    assert not output_path.exists()  # no markdown written after the abort


# ---------------------------------------------------------------------------
# flush failure resilience: a cache write error must not abort the run or
# turn Ctrl-C into a traceback (the cache is a performance optimization)
# ---------------------------------------------------------------------------
def test_flush_failure_warns_and_run_completes(tmp_path, monkeypatch, capsys) -> None:
    """A failing cache flush (e.g. full disk) must not abort transcription:
    the run warns on stderr, keeps going, and still writes the markdown with
    the in-memory transcripts (chat.md is the artifact; the cache is not)."""
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=2)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(cli, "CACHE_FLUSH_EVERY", 1)  # flush after every file
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _FlushBrokenTranscriber(_CacheAwareTranscriber):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.transcribed = 0

        def transcribe(self, fp: str) -> str:
            self.transcribed += 1
            return f"текст {self.transcribed}"

        def flush_cache(self) -> None:
            raise OSError("disk full (тестовая)")

    stub = _FlushBrokenTranscriber()
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **kw: stub)

    cli.main()  # must not raise, must not sys.exit

    assert stub.transcribed == 2  # both files were transcribed
    err = capsys.readouterr().err
    assert "Не удалось сохранить кэш расшифровок" in err
    assert "Traceback" not in err
    # The markdown still carries both transcripts from memory.
    md = output_path.read_text(encoding="utf-8")
    assert "> *Расшифровка:* текст 1" in md
    assert "> *Расшифровка:* текст 2" in md


def test_keyboard_interrupt_with_failing_flush_still_exits_1_cleanly(
    tmp_path, monkeypatch, capsys
) -> None:
    """Ctrl-C while the cache cannot be written must not become a traceback:
    the interrupt handler warns, and exit code 1 stays the only outcome."""
    import telegram_to_md as cli

    export_dir = _make_voice_export(tmp_path, n=3)
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )
    monkeypatch.setattr("tqdm.tqdm", lambda iterable, **kwargs: iterable)

    class _InterruptingBrokenFlush(_CacheAwareTranscriber):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.transcribed = 0

        def transcribe(self, fp: str) -> str:
            self.transcribed += 1
            if self.transcribed == 2:
                raise KeyboardInterrupt
            return "текст"

        def flush_cache(self) -> None:
            raise OSError("disk full (тестовая)")

    stub = _InterruptingBrokenFlush()
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **kw: stub)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Не удалось сохранить кэш расшифровок" in err
    assert "Traceback" not in err
    assert not output_path.exists()


def test_no_media_export_never_constructs_transcriber(tmp_path, monkeypatch, capsys) -> None:
    """A text-only export skips model loading entirely (no Transcriber
    construction), prints the no-media notice, exits 0, and still writes the
    markdown."""
    import telegram_to_md as cli

    export_dir = tmp_path / "ChatExport_text_only"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "from": "Автор",
                        "text": "Только текст",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["telegram_to_md.py", str(export_dir), "--output", str(output_path)],
    )

    def _bomb(*args, **kwargs):
        raise AssertionError("модель не должна загружаться без медиа")

    monkeypatch.setattr(cli, "Transcriber", _bomb)

    cli.main()  # no SystemExit: exit 0

    assert "Нет файлов для расшифровки" in capsys.readouterr().out
    md = output_path.read_text(encoding="utf-8")
    assert "Только текст" in md
