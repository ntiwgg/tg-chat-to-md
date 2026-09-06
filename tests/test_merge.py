"""Hermetic tests for merge_exports: message union, cache re-keying, guards.

Builds tiny synthetic exports (result.json + real placeholder audio files +
legacy-keyed _transcripts_cache.json) under tmp_path, then runs the real
merge entry point (merge_exports.main with sys.argv patched) with --output
pointing into tmp_path — nothing outside the fixture is ever written.

Covers:
  - union of messages from two exports, deduped by id (earlier wins);
  - legacy-keyed caches from BOTH exports migrated against their OWN export
    roots and merged, so transcripts survive and land on canonical keys;
  - a poisoned legacy key (file exists only in the other export) is dropped,
    not silently attached to the wrong file;
  - an all-empty input does not crash and reports an empty date range;
  - a corrupt cache warns on stderr and the merge continues.
"""

import json
import sys
from pathlib import Path

import merge_exports
from src.cache import CACHE_FILE_NAME

OLD_NAME = "ChatExport_2026-07-24 (1)"
NEW_NAME = "ChatExport_2026-08-10 (1)"
OLD_AUDIO = "voice_messages/old_voice.ogg"
NEW_AUDIO = "voice_messages/new_voice.ogg"


# ---------------------------------------------------------------------------
# fixtures: synthetic exports under tmp_path
# ---------------------------------------------------------------------------
def _write_export(
    root: Path, name: str, messages: list[dict], chat_name: str = "Тестовый чат"
) -> Path:
    """Create an export dir with result.json; return its path."""
    export_dir = root / name
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "result.json").write_text(
        json.dumps({"name": chat_name, "id": 999, "messages": messages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return export_dir


def _touch_audio(export_dir: Path, rel_path: str) -> Path:
    """Create a placeholder audio file inside the export; return its path."""
    audio = export_dir / rel_path
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"")
    return audio


def _text_message(mid: int, text: str, date: str) -> dict:
    return {
        "id": mid,
        "type": "message",
        "date": date,
        "date_unixtime": "0",
        "from": "Автор",
        "text": text,
    }


def _voice_message(mid: int, file_rel: str, date: str) -> dict:
    return {
        "id": mid,
        "type": "message",
        "date": date,
        "date_unixtime": "0",
        "from": "Автор",
        "text": "",
        "media_type": "voice_message",
        "file": file_rel,
        "duration_seconds": 12,
    }


def _write_legacy_cache(export_dir: Path, entries: dict[str, str]) -> None:
    """Write a cache file with (possibly legacy-relative) keys."""
    (export_dir / CACHE_FILE_NAME).write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


def _run_merge(
    tmp_path: Path,
    monkeypatch,
    old: Path,
    new: Path,
    *,
    extra: list[Path] | None = None,
    capsys,
) -> tuple[Path, str, str]:
    """Run merge_exports.main() with patched argv; return (output_path, stdout, stderr)."""
    argv = [
        "merge_exports",
        "--old",
        str(old),
        "--new",
        str(new),
    ]
    for e in extra or []:
        argv += ["--extra", str(e)]
    output_path = tmp_path / "merged.md"
    argv += ["--output", str(output_path)]

    monkeypatch.setattr(sys, "argv", argv)
    merge_exports.main()
    captured = capsys.readouterr()
    return output_path, captured.out, captured.err


# ---------------------------------------------------------------------------
# happy path: union + dedupe, legacy caches migrated per export root
# ---------------------------------------------------------------------------
def test_merge_unions_dedupes_and_preserves_legacy_caches(tmp_path, monkeypatch, capsys):
    old_dir = _write_export(
        tmp_path,
        OLD_NAME,
        [
            _text_message(1, "Привет из старого экспорта", "2026-07-24T10:00:00"),
            _voice_message(2, OLD_AUDIO, "2026-07-24T11:00:00"),
        ],
    )
    _touch_audio(old_dir, OLD_AUDIO)
    # Realistic legacy key: CWD-relative, prefixed with the export folder name.
    # Also poisons a key that only exists in the OTHER export: it must be
    # dropped when migrated against THIS export root.
    _write_legacy_cache(
        old_dir,
        {
            f"{OLD_NAME}/{OLD_AUDIO}": "Голосовое из старого",
            f"{OLD_NAME}/{NEW_AUDIO}": "Мусорная запись для чужого файла",
        },
    )

    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [
            _text_message(2, "Дубль — должен быть отброшен", "2026-08-10T09:00:00"),
            _text_message(3, "Привет из нового экспорта", "2026-08-10T10:00:00"),
            _voice_message(4, NEW_AUDIO, "2026-08-10T11:00:00"),
        ],
    )
    _touch_audio(new_dir, NEW_AUDIO)
    # Plain-relative legacy key (old style, typed from inside the export dir).
    _write_legacy_cache(new_dir, {NEW_AUDIO: "Голосовое из нового"})

    output_path, out, err = _run_merge(tmp_path, monkeypatch, old_dir, new_dir, capsys=capsys)

    md = output_path.read_text(encoding="utf-8")

    # Union of unique messages by id: 1 and 2 (old), 3 and 4 (new).
    assert "Привет из старого экспорта" in md
    assert "Привет из нового экспорта" in md
    assert "Дубль — должен быть отброшен" not in md
    assert "Всего уникальных сообщений: 4" in out
    assert "Дублей отброшено (id уже был в более раннем экспорте): 1" in out

    # Both legacy caches were migrated against their own roots and merged:
    # each transcript lands on the canonical absolute key of ITS file.
    assert "> *Расшифровка:* Голосовое из старого" in md
    assert "> *Расшифровка:* Голосовое из нового" in md
    # The poisoned entry (file exists only in the other export) was dropped.
    assert "Мусорная запись для чужого файла" not in md

    # No transcript missing: both voice files have entries in the merged cache.
    assert "Без расшифровки: 0" in out
    assert "Диапазон дат: 2026-07-24T10:00:00 — 2026-08-10T11:00:00" in out
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# empty inputs: no IndexError, sane statistics
# ---------------------------------------------------------------------------
def test_merge_empty_exports_do_not_crash(tmp_path, monkeypatch, capsys):
    empty_old = _write_export(tmp_path, "ChatExport_empty_1", [])
    empty_new = _write_export(tmp_path, "ChatExport_empty_2", [])

    output_path, out, err = _run_merge(
        tmp_path, monkeypatch, empty_old, empty_new, capsys=capsys
    )

    assert output_path.exists()
    md = output_path.read_text(encoding="utf-8")
    assert md.strip()  # header-only document still written
    assert "Всего уникальных сообщений: 0" in out
    assert "нет сообщений (пустой результат)" in out
    assert "Traceback" not in err


def test_merge_with_one_empty_export_still_merges_other(tmp_path, monkeypatch, capsys):
    empty_old = _write_export(tmp_path, "ChatExport_empty_old", [])
    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [_text_message(5, "Остался только новый", "2026-08-10T12:00:00")],
    )

    output_path, out, _err = _run_merge(
        tmp_path, monkeypatch, empty_old, new_dir, capsys=capsys
    )

    md = output_path.read_text(encoding="utf-8")
    assert "Остался только новый" in md
    assert "Всего уникальных сообщений: 1" in out
    assert "Диапазон дат: 2026-08-10T12:00:00 — 2026-08-10T12:00:00" in out


# ---------------------------------------------------------------------------
# corrupt cache: warn and continue
# ---------------------------------------------------------------------------
def test_merge_corrupt_cache_warns_and_continues(tmp_path, monkeypatch, capsys):
    old_dir = _write_export(
        tmp_path,
        OLD_NAME,
        [_voice_message(7, OLD_AUDIO, "2026-07-24T10:00:00")],
    )
    _touch_audio(old_dir, OLD_AUDIO)
    (old_dir / CACHE_FILE_NAME).write_text("{это не json", encoding="utf-8")
    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [_text_message(8, "Свежий пост", "2026-08-10T10:00:00")],
    )

    output_path, out, err = _run_merge(tmp_path, monkeypatch, old_dir, new_dir, capsys=capsys)

    # Warning on stderr, merge still completes; the voice file simply has no
    # transcript and is reported as such instead of crashing.
    assert "Кэш расшифровок повреждён" in err
    md = output_path.read_text(encoding="utf-8")
    assert "> *(расшифровка недоступна)*" in md
    assert "Без расшифровки: 1" in out
    assert "Свежий пост" in md
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# output goes only where --output says
# ---------------------------------------------------------------------------
def test_merge_never_writes_outside_output_path(tmp_path, monkeypatch, capsys):
    old_dir = _write_export(
        tmp_path, OLD_NAME, [_text_message(9, "Один текст", "2026-07-24T10:00:00")]
    )
    new_dir = _write_export(
        tmp_path, NEW_NAME, [_text_message(10, "Второй текст", "2026-08-10T10:00:00")]
    )

    output_path, out, _err = _run_merge(tmp_path, monkeypatch, old_dir, new_dir, capsys=capsys)

    assert output_path.exists()
    # The only files that exist in tmp are the fixtures plus the single output.
    expected_roots = {old_dir, new_dir, output_path}
    produced = {p for p in tmp_path.rglob("*") if p.is_file()}
    for f in produced:
        assert any(f == root or root in f.parents for root in expected_roots)
    assert "Записан" in out
