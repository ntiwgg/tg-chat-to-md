"""Hermetic Transcriber tests through the faster-whisper duck-contract seam.

src.transcriber must stay importable without faster-whisper installed (the
CLI and the rest of the suite rely on it), so a real WhisperModel can never
be constructed in tests. The documented seam is the model itself: tests
monkeypatch the module-level HAS_WHISPER flag and the WhisperModel /
BatchedInferencePipeline classes with recorders, then run the real
Transcriber.__init__ / transcribe code paths end to end.

Covers: the not-installed RuntimeError, "auto" compute-type resolution per
device, CPU thread defaults, exact constructor/transcribe call arguments
(language, beam_size, vad_filter, batch_size), batched-vs-plain routing,
cache wiring (path, legacy migration, cache hit), and explicit settings
passthrough. No audio, no network, no real model.
"""

import pytest

import src.transcriber as transcriber_mod
from src.cache import CACHE_FILE_NAME, write_cache
from src.transcriber import Transcriber


class _Segment:
    """Minimal whisper segment: only .text is consumed by Transcriber."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    """Records WhisperModel(...) and model.transcribe(...) call arguments.

    Transcriber.__init__ stores no reference besides self._model, so the
    recorder instance is also registered on the class for assertions.
    """

    instances: list["_FakeWhisperModel"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.transcribe_calls: list[dict] = []
        type(self).instances.append(self)

    def transcribe(self, filepath, language=None, beam_size=None, vad_filter=None) -> tuple:
        self.transcribe_calls.append(
            {
                "filepath": filepath,
                "language": language,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
            }
        )
        return iter([_Segment("привет мир")]), None


class _FakeBatchedPipeline:
    """Records BatchedInferencePipeline(...) and batched.transcribe(...) args."""

    instances: list["_FakeBatchedPipeline"] = []

    def __init__(self, model=None) -> None:
        self.model = model
        self.transcribe_calls: list[dict] = []
        type(self).instances.append(self)

    def transcribe(self, filepath, language=None, beam_size=None, batch_size=None) -> tuple:
        self.transcribe_calls.append(
            {
                "filepath": filepath,
                "language": language,
                "beam_size": beam_size,
                "batch_size": batch_size,
            }
        )
        return iter([_Segment("привет мир")]), None


def _enable_whisper(monkeypatch, cpu_count: int = 1) -> None:
    """Point Transcriber.__init__ at the recorder fakes (no real model)."""
    _FakeWhisperModel.instances = []
    _FakeBatchedPipeline.instances = []
    monkeypatch.setattr(transcriber_mod, "HAS_WHISPER", True)
    monkeypatch.setattr(transcriber_mod, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcriber_mod, "BatchedInferencePipeline", _FakeBatchedPipeline)
    monkeypatch.setattr(transcriber_mod.mp, "cpu_count", lambda: cpu_count)


# ---------------------------------------------------------------------------
# not-installed guard
# ---------------------------------------------------------------------------
def test_init_without_whisper_raises_actionable_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(transcriber_mod, "HAS_WHISPER", False)

    with pytest.raises(
        RuntimeError, match=r"^faster-whisper is not installed\. Run: pip install faster-whisper$"
    ):
        Transcriber()


# ---------------------------------------------------------------------------
# __init__ defaults and call arguments (cpu / cuda)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cpu_count", "expected_threads"),
    [
        # max(1, n//2) must clamp at 1, not at 2...
        (1, 1),
        # ...and divide by two, not by three
        (6, 3),
    ],
)
def test_init_cpu_defaults_resolve_auto_compute_and_threads(
    monkeypatch, cpu_count, expected_threads
) -> None:
    _enable_whisper(monkeypatch, cpu_count=cpu_count)

    t = Transcriber(device="cpu")  # compute_type defaults to "auto"

    model = _FakeWhisperModel.instances[-1]
    assert model.args == ("medium",)
    assert model.kwargs == {
        "device": "cpu",
        "compute_type": "int8",  # auto + cpu -> int8
        "cpu_threads": expected_threads,
        "num_workers": 2,
    }
    assert isinstance(model.kwargs["cpu_threads"], int)
    assert t._use_batched is False
    assert t._cache_path is None
    assert t._cache == {}


def test_init_cuda_is_the_default_device(monkeypatch) -> None:
    """The constructor defaults to GPU (device='cuda'): auto compute_type
    resolves to float16 and the batched pipeline is built."""
    _enable_whisper(monkeypatch)

    t = Transcriber()  # no device argument: the signature default is "cuda"

    assert _FakeWhisperModel.instances[-1].kwargs == {
        "device": "cuda",
        "compute_type": "float16",  # auto + cuda -> float16
        "cpu_threads": 1,
        "num_workers": 2,
    }
    assert t._use_batched is True
    assert _FakeBatchedPipeline.instances[-1].model is t._model


def test_init_passes_explicit_settings_through(monkeypatch) -> None:
    _enable_whisper(monkeypatch)

    t = Transcriber(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        beam_size=1,
        cpu_threads=2,
    )

    assert _FakeWhisperModel.instances[-1].args == ("tiny",)
    assert _FakeWhisperModel.instances[-1].kwargs == {
        "device": "cpu",
        "compute_type": "int8",  # explicit: the auto branch is skipped
        "cpu_threads": 2,  # explicit: mp.cpu_count() never consulted
        "num_workers": 2,
    }
    assert t._language == "en"
    assert t._beam_size == 1


# ---------------------------------------------------------------------------
# cache wiring inside __init__
# ---------------------------------------------------------------------------
def test_init_with_cache_dir_reads_and_migrates_cache(tmp_path, monkeypatch) -> None:
    _enable_whisper(monkeypatch)
    export = tmp_path / "ChatExport_seam"
    audio = export / "voice_messages" / "audio_1.ogg"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"")
    legacy_key = "ChatExport_seam/voice_messages/audio_1.ogg"
    write_cache(export / CACHE_FILE_NAME, {legacy_key: "расшифровка"})

    t = Transcriber(cache_dir=export)

    assert t._cache_path == export / CACHE_FILE_NAME
    assert t._cache == {str(audio.resolve()): "расшифровка"}
    # a cache hit must never reach the model
    text = t.transcribe(str(audio.resolve()))
    assert text == "расшифровка"
    assert _FakeWhisperModel.instances[-1].transcribe_calls == []


def test_init_without_cache_dir_stays_in_memory_only(monkeypatch) -> None:
    _enable_whisper(monkeypatch)

    t = Transcriber()

    assert t._cache_path is None
    assert t._cache == {}


# ---------------------------------------------------------------------------
# public cache stats API (cached_count / missing_from_cache)
# ---------------------------------------------------------------------------
def _bare_transcriber_with_cache(cache: dict[str, str]) -> Transcriber:
    """A Transcriber instance without the whisper model: the stats methods
    only read the private cache dict, so object.__new__ skips __init__."""
    t = object.__new__(Transcriber)
    t._cache = cache
    return t


def test_cached_count_reports_exactly_the_cached_filepaths() -> None:
    t = _bare_transcriber_with_cache({"a.ogg": "текст", "b.ogg": "текст"})

    assert t.cached_count(["a.ogg", "b.ogg", "c.ogg"]) == 2
    assert t.cached_count(["c.ogg", "d.ogg"]) == 0
    assert t.cached_count(["a.ogg"]) == 1
    assert t.cached_count([]) == 0


def test_missing_from_cache_lists_only_absent_filepaths_in_order() -> None:
    t = _bare_transcriber_with_cache({"a.ogg": "текст", "b.ogg": "текст"})

    assert t.missing_from_cache(["a.ogg", "b.ogg"]) == []
    assert t.missing_from_cache(["a.ogg", "b.ogg", "c.ogg", "d.ogg"]) == ["c.ogg", "d.ogg"]
    # order of the INPUT is preserved, not the cache's
    assert t.missing_from_cache(["c.ogg", "a.ogg", "d.ogg"]) == ["c.ogg", "d.ogg"]
    assert t.missing_from_cache([]) == []



# ---------------------------------------------------------------------------
# transcribe(): call-argument contract per device
# ---------------------------------------------------------------------------
def test_transcribe_cpu_forwards_language_beam_and_vad_filter(monkeypatch) -> None:
    _enable_whisper(monkeypatch)

    t = Transcriber(device="cpu")  # language defaults to "ru", beam_size to 5
    text = t.transcribe("chat/voice.ogg")

    assert text == "привет мир"
    assert _FakeWhisperModel.instances[-1].transcribe_calls == [
        {"filepath": "chat/voice.ogg", "language": "ru", "beam_size": 5, "vad_filter": True}
    ]
    assert t._cache == {"chat/voice.ogg": "привет мир"}


def test_transcribe_cuda_routes_to_batched_pipeline(monkeypatch) -> None:
    _enable_whisper(monkeypatch)

    t = Transcriber(device="cuda", language="en", beam_size=1)
    text = t.transcribe("chat/voice.ogg")

    assert text == "привет мир"
    assert _FakeBatchedPipeline.instances[-1].transcribe_calls == [
        {"filepath": "chat/voice.ogg", "language": "en", "beam_size": 1, "batch_size": 16}
    ]
    assert _FakeWhisperModel.instances[-1].transcribe_calls == []  # plain path unused
    assert t._cache == {"chat/voice.ogg": "привет мир"}
