"""Unit tests for cache key canonicalization and batch write policy.

Hermetic: no audio decoding, no real whisper model (stubs only), no network.
Covers:
  - _resolve_file returns the same canonical absolute key regardless of CWD
    and of how export_dir was typed on the CLI;
  - migrate_cache_keys (tg_chat_to_md.cache) re-keys legacy export-relative entries to
    canonical absolute paths (prefix stripped until the file exists), keeps
    absolute keys, drops unmatchable ones, and never mutates its input;
  - the cache file is written only by flush_cache(), never by transcribe();
  - a legacy cache on disk is migrated in memory at Transcriber load time and
    persists canonical keys after flush;
  - broken JSON on disk degrades to an empty cache instead of crashing.
"""

import json
import os
from pathlib import Path

import pytest

import tg_chat_to_md.transcriber as transcriber_mod
from tg_chat_to_md.cache import CACHE_FILE_NAME, migrate_cache_keys, read_cache, write_cache
from tg_chat_to_md.parser import _resolve_file
from tg_chat_to_md.transcriber import Transcriber

EXPORT_NAME = "ChatExport_2026-07-24 (1)"
AUDIO_1 = "audio_1@27-06-2026_09-40-01.ogg"
AUDIO_2 = "audio_2@27-06-2026_09-40-04.ogg"
VOICE_SUBDIR = "voice_messages"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _touch(path: Path) -> Path:
    """Create an empty file (creating parents); return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _make_export(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create an export dir with two voice files; return (export, audio1, audio2)."""
    export = tmp_path / EXPORT_NAME
    audio1 = _touch(export / VOICE_SUBDIR / AUDIO_1)
    audio2 = _touch(export / VOICE_SUBDIR / AUDIO_2)
    return export, audio1, audio2


class _StubSegment:
    """Minimal whisper segment: only .text is used by Transcriber."""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubModel:
    """Stand-in for the faster-whisper model: returns fixed segments.

    A real model yields one segment per utterance; a stub that emits exactly
    one segment makes join regressions (" ".join collapsing into "".join or
    "\\n".join) invisible, so tests may inject several padded segments.
    """

    def __init__(self, text: str = "привет мир", segments: list[str] | None = None) -> None:
        self.text = text
        self.segments = segments if segments is not None else [text]
        self.calls = 0

    def transcribe(self, filepath, language=None, beam_size=None, vad_filter=None):
        self.calls += 1
        return iter(_StubSegment(s) for s in self.segments), None


def _transcriber_with_stub_model(cache_path: Path | None) -> tuple[Transcriber, _StubModel]:
    """Transcriber instance without __init__ (no model load) with cache wired."""
    stub = _StubModel()
    t = object.__new__(Transcriber)
    t._cache = {}
    t._cache_path = cache_path
    t._language = "ru"
    t._beam_size = 5
    t._use_batched = False
    t._model = stub
    return t, stub


# ---------------------------------------------------------------------------
# _resolve_file: canonical, CWD/path-form-independent keys
# ---------------------------------------------------------------------------
def test_resolve_file_key_does_not_depend_on_cwd(tmp_path, monkeypatch) -> None:
    export = tmp_path / EXPORT_NAME
    audio1 = _touch(export / VOICE_SUBDIR / AUDIO_1)
    rel_media = f"{VOICE_SUBDIR}/{AUDIO_1}"

    # Run 1: relative export_dir arg typed from inside the parent dir
    monkeypatch.chdir(tmp_path)
    key_rel = _resolve_file(rel_media, Path(EXPORT_NAME))

    # Run 2: absolute export_dir arg typed from a different CWD
    monkeypatch.chdir(tmp_path.parent)
    key_abs = _resolve_file(rel_media, export)

    assert key_rel == key_abs
    assert Path(key_rel).is_absolute()
    assert Path(key_rel) == audio1.resolve()


def test_resolve_file_key_same_for_differently_typed_export_dir(tmp_path, monkeypatch) -> None:
    export = tmp_path / "ChatExport"
    audio = _touch(export / VOICE_SUBDIR / AUDIO_1)
    monkeypatch.chdir(tmp_path)
    rel_media = f"{VOICE_SUBDIR}/{AUDIO_1}"

    forms = ["ChatExport", "./ChatExport", "ChatExport/", str(export)]
    keys = {_resolve_file(rel_media, Path(form)) for form in forms}

    assert keys == {str(audio.resolve())}


def test_resolve_file_returns_none_for_missing_and_placeholders(tmp_path) -> None:
    export = _make_export(tmp_path)[0]

    assert _resolve_file(f"{VOICE_SUBDIR}/ghost.ogg", export) is None
    assert _resolve_file(None, export) is None
    assert _resolve_file("(File not included by export)", export) is None
    assert _resolve_file("(File unavailable)", export) is None


# ---------------------------------------------------------------------------
# migrate_cache_keys
# ---------------------------------------------------------------------------
def test_migrate_rekeys_legacy_export_prefixed_entry(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    legacy_key = f"{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}"

    migrated = migrate_cache_keys({legacy_key: "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_rekeys_plain_relative_entry(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)

    migrated = migrate_cache_keys({f"{VOICE_SUBDIR}/{AUDIO_1}": "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_strips_multiple_leading_components(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    deep_key = f"old/archive/{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}"

    migrated = migrate_cache_keys({deep_key: "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_keeps_only_existing_files_and_drops_the_rest(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    cache = {
        f"{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}": "exists",
        f"{VOICE_SUBDIR}/vanished.ogg": "gone",
        "": "empty key",
    }

    migrated = migrate_cache_keys(cache, export)

    assert migrated == {str(audio1.resolve()): "exists"}


def test_migrate_keeps_existing_absolute_keys_verbatim(tmp_path) -> None:
    export, _audio1, _audio2 = _make_export(tmp_path)
    abs_file = _touch(export / "media" / "present.ogg")  # exists on disk

    migrated = migrate_cache_keys({str(abs_file): "текст"}, export)

    # Existing absolute keys are already canonical — kept as-is, byte for byte.
    assert migrated == {str(abs_file): "текст"}


def test_migrate_drops_nonexistent_absolute_keys(tmp_path) -> None:
    """Contract (pinned by the property test): an absolute key whose file no
    longer exists is dropped like any other unmatchable entry — it could never
    be served (transcripts are only looked up for files that exist), so
    keeping it would re-write dead weight into every future cache file."""
    export, _audio1, _audio2 = _make_export(tmp_path)
    gone_key = str(export / "voice_messages" / "deleted.ogg")  # not on disk

    migrated = migrate_cache_keys({gone_key: "текст"}, export)

    assert migrated == {}


def test_migrate_does_not_mutate_input_dict(tmp_path) -> None:
    export, _audio1, _audio2 = _make_export(tmp_path)
    cache = {
        f"{VOICE_SUBDIR}/{AUDIO_1}": "текст",
        f"{VOICE_SUBDIR}/ghost.ogg": "ghost",
    }
    snapshot = dict(cache)

    migrate_cache_keys(cache, export)

    assert cache == snapshot


# ---------------------------------------------------------------------------
# batch write policy: disk writes happen only in flush_cache()
# ---------------------------------------------------------------------------
def test_transcribe_does_not_write_cache_file_until_flush(tmp_path) -> None:
    audio = _touch(tmp_path / VOICE_SUBDIR / "audio_1.ogg")
    cache_file = tmp_path / "_transcripts_cache.json"
    t, stub = _transcriber_with_stub_model(cache_file)

    text = t.transcribe(str(audio))

    assert text == "привет мир"
    assert not cache_file.exists()  # per-file write must not happen

    t.flush_cache()
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {str(audio): text}
    # compact single-line JSON (no indent)
    assert cache_file.read_text(encoding="utf-8") == json.dumps(
        {str(audio): text}, ensure_ascii=False, separators=(",", ":")
    )


def test_transcribe_cache_hit_skips_model(tmp_path) -> None:
    audio = _touch(tmp_path / VOICE_SUBDIR / "audio_1.ogg")
    cache_file = tmp_path / "_transcripts_cache.json"
    t, stub = _transcriber_with_stub_model(cache_file)

    t.transcribe(str(audio))
    t.transcribe(str(audio))

    assert stub.calls == 1  # the second call was served from the in-memory cache
    t.flush_cache()
    # Exactly one canonical key was persisted.
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {str(audio): "привет мир"}


def test_transcribe_joins_multi_segment_output_with_single_spaces(tmp_path) -> None:
    """REGRESSION: with a single-segment stub, a ' '.join collapsing into
    ''.join or '\\n'.join was invisible. Three padded segments must come back
    stripped, in order, joined by exactly one space each."""
    audio = _touch(tmp_path / VOICE_SUBDIR / "audio_1.ogg")
    t, stub = _transcriber_with_stub_model(tmp_path / "_transcripts_cache.json")
    stub.segments = [" первый ", " второй", "третий  "]

    text = t.transcribe(str(audio))

    assert text == "первый второй третий"
    assert stub.calls == 1
    assert "  " not in text and "\n" not in text


def test_flush_cache_without_cache_dir_writes_no_file_anywhere(tmp_path, monkeypatch) -> None:
    """Without a cache_dir, transcribe + flush must leave no cache file under
    the whole working tree — not merely skip the configured path."""
    monkeypatch.chdir(tmp_path)
    t, _stub = _transcriber_with_stub_model(None)

    t.transcribe(str(_touch(tmp_path / "a.ogg")))
    t.flush_cache()  # must not raise

    assert list(tmp_path.rglob(CACHE_FILE_NAME)) == []


# ---------------------------------------------------------------------------
# load-time migration of a real cache file, then flush persistence
# ---------------------------------------------------------------------------
def test_init_migrates_legacy_cache_file_and_flush_persists_canonical_keys(
    tmp_path, monkeypatch
) -> None:
    export, audio1, audio2 = _make_export(tmp_path)
    legacy_key = f"{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}"  # CWD-dependent old style
    relative_key = f"{VOICE_SUBDIR}/{AUDIO_2}"  # old style, typed from inside export
    cache_file = export / "_transcripts_cache.json"
    cache_file.write_text(
        json.dumps({legacy_key: "старый текст", relative_key: "новый текст"}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Real __init__ with the whisper model factory stubbed out (no download)
    model = _StubModel("не должно транскрибироваться")
    monkeypatch.setattr(transcriber_mod, "WhisperModel", lambda *a, **k: model)
    t = Transcriber(model_size="tiny", device="cpu", cache_dir=export)

    # Both files resolve against the export root and are served from the
    # migrated cache: the model is never invoked.
    assert t.transcribe(str(audio1.resolve())) == "старый текст"
    assert t.transcribe(str(audio2.resolve())) == "новый текст"
    assert model.calls == 0
    # loading must not rewrite the file (first flush persists the migration)
    raw = json.loads(cache_file.read_text(encoding="utf-8"))
    assert set(raw) == {legacy_key, relative_key}

    t.flush_cache()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {
        str(audio1.resolve()): "старый текст",
        str(audio2.resolve()): "новый текст",
    }


# ---------------------------------------------------------------------------
# robustness: broken JSON on disk
# ---------------------------------------------------------------------------
def test_broken_cache_json_reads_as_empty(tmp_path) -> None:
    cache_file = tmp_path / "_transcripts_cache.json"
    cache_file.write_text("{это не json", encoding="utf-8")

    assert read_cache(cache_file) == {}


def test_missing_cache_file_reads_as_empty(tmp_path) -> None:
    assert read_cache(tmp_path / "_transcripts_cache.json") == {}


@pytest.mark.parametrize("raw", ["[]", '"x"', "42"])
def test_non_object_cache_json_degrades_to_empty(tmp_path, raw, capsys) -> None:
    """REGRESSION: syntactically valid non-dict JSON ([] / string / number)
    passed json.loads untouched and crashed migrate_cache_keys on .items().
    A structurally valid cache must be a dict; anything else is corrupt."""
    cache_file = tmp_path / "_transcripts_cache.json"
    cache_file.write_text(raw, encoding="utf-8")

    assert read_cache(cache_file) == {}

    err = capsys.readouterr().err
    assert "⚠ Кэш расшифровок повреждён, начинаю с пустого:" in err
    assert " (ожидался JSON-объект)\n" in err  # standalone literal, not embedded in a wrapper


@pytest.mark.parametrize("raw", ["[]", '"x"', "42"])
def test_transcriber_init_survives_non_object_cache(tmp_path, monkeypatch, raw) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    cache_file = export / "_transcripts_cache.json"
    cache_file.write_text(raw, encoding="utf-8")
    model = _StubModel("текст из модели")
    monkeypatch.setattr(transcriber_mod, "WhisperModel", lambda *a, **k: model)

    t = Transcriber(model_size="tiny", device="cpu", cache_dir=export)  # must not crash

    # The degraded cache is empty, so transcribe takes the real model path...
    text = t.transcribe(str(audio1.resolve()))
    assert text == "текст из модели"
    assert model.calls == 1
    # ...and flush persists a proper dict cache over the corrupt file.
    t.flush_cache()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {
        str(audio1.resolve()): "текст из модели"
    }


# ---------------------------------------------------------------------------
# atomic write_cache: crash mid-write never corrupts or loses the old cache
# ---------------------------------------------------------------------------
def _tmp_suffix(path: Path) -> Path:
    """The temp sibling write_cache stages before the atomic rename."""
    return path.with_name(path.name + ".tmp")


def test_write_cache_success_leaves_no_tmp_and_content_correct(tmp_path) -> None:
    cache_file = tmp_path / CACHE_FILE_NAME
    data = {"ключ": "текст с юникодом: 日本語"}

    write_cache(cache_file, data)

    assert not _tmp_suffix(cache_file).exists()  # no staging litter
    assert json.loads(cache_file.read_text(encoding="utf-8")) == data
    # compact single-line JSON, unchanged from the pre-atomic format
    assert cache_file.read_text(encoding="utf-8") == json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.parametrize("fail_on_replace", [False, True])
def test_failed_write_preserves_original_cache_and_leaves_no_tmp(
    tmp_path, monkeypatch, fail_on_replace
) -> None:
    """Crash-consistency: an OSError at either stage of the atomic write (temp
    write or rename) must leave the ORIGINAL cache file byte-identical, no
    .tmp litter behind, and the old dict still readable — a failed flush never
    loses the transcripts that were already on disk."""
    cache_file = tmp_path / CACHE_FILE_NAME
    old_data = {"старый ключ": "старый текст"}
    write_cache(cache_file, old_data)
    original_bytes = cache_file.read_bytes()

    if fail_on_replace:
        def _boom_replace(src, dst):
            raise OSError("rename failed (тестовая)")
        monkeypatch.setattr(os, "replace", _boom_replace)
    else:
        real_write_text = Path.write_text

        def _failing_write_text(self, *args, **kwargs):
            if self == _tmp_suffix(cache_file):
                raise OSError("write failed (тестовая)")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _failing_write_text)

    with pytest.raises(OSError):
        write_cache(cache_file, {"новый ключ": "новый текст"})

    # The previous cache survived untouched and still reads as the old dict.
    assert cache_file.read_bytes() == original_bytes
    assert read_cache(cache_file) == old_data
    assert not _tmp_suffix(cache_file).exists()  # no staging litter after failure
