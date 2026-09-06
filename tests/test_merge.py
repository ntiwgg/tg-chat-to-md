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
EXTRA_NAME = "ChatExport_2026-09-01 (1)"
OLD_AUDIO = "voice_messages/old_voice.ogg"
NEW_AUDIO = "voice_messages/new_voice.ogg"
EXTRA_AUDIO = "voice_messages/extra_voice.ogg"


# ---------------------------------------------------------------------------
# fixtures: synthetic exports under tmp_path
# ---------------------------------------------------------------------------
def _write_export(
    root: Path,
    name: str,
    messages: list[dict],
    chat_name: str = "Тестовый чат",
    chat_id: int = 999,
) -> Path:
    """Create an export dir with result.json; return its path."""
    export_dir = root / name
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "result.json").write_text(
        json.dumps({"name": chat_name, "id": chat_id, "messages": messages}, ensure_ascii=False),
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


def test_merge_non_object_cache_warns_and_continues(tmp_path, monkeypatch, capsys):
    """REGRESSION: a structurally valid non-dict cache ([]) crashed
    migrate_cache_keys on .items() during merge. It must degrade like any
    other corrupt cache: warning on stderr, merge completes."""
    old_dir = _write_export(
        tmp_path,
        OLD_NAME,
        [_voice_message(7, OLD_AUDIO, "2026-07-24T10:00:00")],
    )
    _touch_audio(old_dir, OLD_AUDIO)
    (old_dir / CACHE_FILE_NAME).write_text("[]", encoding="utf-8")
    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [_text_message(8, "Свежий пост", "2026-08-10T10:00:00")],
    )

    output_path, out, err = _run_merge(tmp_path, monkeypatch, old_dir, new_dir, capsys=capsys)

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


# ---------------------------------------------------------------------------
# --extra: third export merged after --new, its cache migrated against its
# own root
# ---------------------------------------------------------------------------
def test_merge_extra_export_unions_and_reports_per_label(
    tmp_path, monkeypatch, capsys
):
    old_dir = _write_export(
        tmp_path, OLD_NAME, [_text_message(1, "Из старого", "2026-07-24T10:00:00")]
    )
    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [
            _text_message(1, "Дубль старого", "2026-08-10T09:00:00"),
            _text_message(2, "Из нового", "2026-08-10T10:00:00"),
            _voice_message(3, EXTRA_AUDIO, "2026-08-10T11:00:00"),
        ],
    )
    extra_dir = _write_export(tmp_path, EXTRA_NAME, [
        _text_message(2, "Дубль нового", "2026-09-01T09:00:00"),
        _text_message(4, "Из доп. экспорта", "2026-09-01T10:00:00"),
        _voice_message(5, EXTRA_AUDIO, "2026-09-01T11:00:00"),
    ])
    # The extra export's voice file lives in the extra root...
    _touch_audio(extra_dir, EXTRA_AUDIO)
    # ...but the same relative path ALSO exists in the new export: the cache
    # migration must attach the transcript to the extra root's own file.
    _touch_audio(new_dir, EXTRA_AUDIO)
    _write_legacy_cache(extra_dir, {EXTRA_AUDIO: "Расшифровка из доп. экспорта"})

    output_path, out, err = _run_merge(
        tmp_path, monkeypatch, old_dir, new_dir, extra=[extra_dir], capsys=capsys
    )

    md = output_path.read_text(encoding="utf-8")

    # Union of ids 1..5 with both duplicates dropped (earlier wins).
    assert "Из старого" in md
    assert "Из нового" in md
    assert "Из доп. экспорта" in md
    assert "Дубль старого" not in md
    assert "Дубль нового" not in md
    assert "Всего уникальных сообщений: 5" in out
    assert "Дублей отброшено (id уже был в более раннем экспорте): 2" in out
    # Per-label statistics, including the extra export.
    assert "Сообщений в старый экспорте: 1" in out
    assert "Сообщений в новый экспорте: 3" in out
    assert "Сообщений в доп. 1 экспорте: 3" in out
    # The extra cache entry resolved against the extra root and reached the
    # voice message that lives there.
    assert "> *Расшифровка:* Расшифровка из доп. экспорта" in md
    assert "Без расшифровки: 1" in out  # the new export's duplicate file lacks a cache
    assert "Диапазон дат: 2026-07-24T10:00:00 — 2026-09-01T11:00:00" in out
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# chat-id mismatch: warning on stdout, merge still completes
# ---------------------------------------------------------------------------
def test_merge_different_chat_ids_warns(tmp_path, monkeypatch, capsys):
    old_dir = _write_export(
        tmp_path, OLD_NAME, [_text_message(1, "Чат А", "2026-07-24T10:00:00")], chat_id=111
    )
    new_dir = _write_export(
        tmp_path,
        NEW_NAME,
        [_text_message(2, "Чат Б", "2026-08-10T10:00:00")],
        chat_name="Другой чат",
        chat_id=222,
    )

    _output_path, out, err = _run_merge(tmp_path, monkeypatch, old_dir, new_dir, capsys=capsys)

    assert "⚠ Разные id чатов: 111, 222" in out
    assert "Всего уникальных сообщений: 2" in out
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# cache precedence contract: shared canonical key -> first export keeps it
# ---------------------------------------------------------------------------
def test_merge_cache_collision_keeps_earlier_exports_text() -> None:
    """Contract pin (not a bug): caches merge with setdefault semantics —
    'existing keys keep their old value' (_merge_caches docstring), the same
    earlier-wins rule as message dedupe by id. If the later export holds a
    different transcript for the same canonical key, the earlier text wins.
    Order in the merge call follows the CLI: --old, --new, --extra..."""
    shared_key = "/chat/voice_messages/audio_1.ogg"

    merged = merge_exports._merge_caches(
        {shared_key: "текст из старого"}, {shared_key: "текст из нового"}
    )

    assert merged == {shared_key: "текст из старого"}


def test_merge_cache_collision_depends_on_argument_order() -> None:
    """The same contract from the other side: a reversed merge order keeps
    the first argument's value, mirroring how export order on the CLI decides
    message dedupe."""
    shared_key = "/chat/voice_messages/audio_1.ogg"

    merged = merge_exports._merge_caches(
        {shared_key: "текст из нового"}, {shared_key: "текст из старого"}
    )

    assert merged == {shared_key: "текст из нового"}
