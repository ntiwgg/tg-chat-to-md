"""Unit tests for cache key canonicalization and batch write policy.

Hermetic: no audio decoding, no real whisper model (stubs only), no network.
Covers:
  - _resolve_file returns the same canonical absolute key regardless of CWD
    and of how export_dir was typed on the CLI;
  - _migrate_cache_keys re-keys legacy export-relative entries to canonical
    absolute paths (prefix stripped until the file exists), keeps absolute
    keys, drops unmatchable ones, and never mutates its input;
  - the cache file is written only by flush_cache(), never by transcribe();
  - a legacy cache on disk is migrated in memory at Transcriber load time and
    persists canonical keys after flush;
  - broken JSON on disk degrades to an empty cache instead of crashing.
"""

import json
from pathlib import Path

import src.transcriber as transcriber_mod
from src.parser import _resolve_file
from src.transcriber import Transcriber, _migrate_cache_keys, _read_cache

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
    """Stand-in for the faster-whisper model: returns fixed segments."""

    def __init__(self, text: str = "привет мир") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, filepath, language=None, beam_size=None, vad_filter=None):
        self.calls += 1
        return iter([_StubSegment(self.text)]), None


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
# _migrate_cache_keys
# ---------------------------------------------------------------------------
def test_migrate_rekeys_legacy_export_prefixed_entry(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    legacy_key = f"{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}"

    migrated = _migrate_cache_keys({legacy_key: "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_rekeys_plain_relative_entry(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)

    migrated = _migrate_cache_keys({f"{VOICE_SUBDIR}/{AUDIO_1}": "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_strips_multiple_leading_components(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    deep_key = f"old/archive/{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}"

    migrated = _migrate_cache_keys({deep_key: "текст"}, export)

    assert migrated == {str(audio1.resolve()): "текст"}


def test_migrate_keeps_only_existing_files_and_drops_the_rest(tmp_path) -> None:
    export, audio1, _audio2 = _make_export(tmp_path)
    cache = {
        f"{EXPORT_NAME}/{VOICE_SUBDIR}/{AUDIO_1}": "exists",
        f"{VOICE_SUBDIR}/vanished.ogg": "gone",
        "": "empty key",
    }

    migrated = _migrate_cache_keys(cache, export)

    assert migrated == {str(audio1.resolve()): "exists"}


def test_migrate_keeps_absolute_keys_verbatim(tmp_path) -> None:
    export, _audio1, _audio2 = _make_export(tmp_path)
    abs_key = str(export / VOICE_SUBDIR / "elsewhere.ogg")  # need not exist on disk

    migrated = _migrate_cache_keys({abs_key: "текст"}, export)

    assert migrated == {abs_key: "текст"}


def test_migrate_does_not_mutate_input_dict(tmp_path) -> None:
    export, _audio1, _audio2 = _make_export(tmp_path)
    cache = {
        f"{VOICE_SUBDIR}/{AUDIO_1}": "текст",
        f"{VOICE_SUBDIR}/ghost.ogg": "ghost",
    }
    snapshot = dict(cache)

    _migrate_cache_keys(cache, export)

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
    assert t._cache == {str(audio): text}

    t.flush_cache()
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {str(audio): text}
    # compact single-line JSON (no indent)
    assert cache_file.read_text(encoding="utf-8") == json.dumps(
        {str(audio): text}, ensure_ascii=False, separators=(",", ":")
    )


def test_transcribe_cache_hit_skips_model(tmp_path) -> None:
    audio = _touch(tmp_path / VOICE_SUBDIR / "audio_1.ogg")
    t, stub = _transcriber_with_stub_model(tmp_path / "_transcripts_cache.json")

    t.transcribe(str(audio))
    t.transcribe(str(audio))

    assert stub.calls == 1
    assert t._cache[str(audio)] == "привет мир"


def test_flush_cache_is_noop_without_cache_dir(tmp_path) -> None:
    t, _stub = _transcriber_with_stub_model(None)

    t.transcribe(str(_touch(tmp_path / "a.ogg")))

    assert t._cache_path is None
    t.flush_cache()  # must not raise


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
    monkeypatch.setattr(transcriber_mod, "WhisperModel", lambda *a, **k: _StubModel())
    t = Transcriber(model_size="tiny", device="cpu", cache_dir=export)

    assert t._cache == {
        str(audio1.resolve()): "старый текст",
        str(audio2.resolve()): "новый текст",
    }
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

    assert _read_cache(cache_file) == {}


def test_missing_cache_file_reads_as_empty(tmp_path) -> None:
    assert _read_cache(tmp_path / "_transcripts_cache.json") == {}
