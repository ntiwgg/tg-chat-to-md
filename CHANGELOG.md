# Changelog

All notable changes to TelegramAnaL are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet

## [0.1.0] - 2026-09-06

### Added

- `telegram-to-md` CLI: parse → transcribe → format pipeline from a Telegram Desktop export to one chronological Markdown file.
- faster-whisper transcription of voice messages and round videos — GPU (batched, `float16`) and CPU (`int8` + VAD) modes.
- Durable JSON transcript cache with canonical absolute-path keys and automatic migration of legacy relative keys.
- `merge_exports.py` helper: combine overlapping partial exports of the same chat, deduplicated by message id.
- MIT license and project metadata (`requires-python >= 3.11`, ruff + strict mypy config, dev extra).
- Synthetic privacy-safe demo export in `examples/sample_export/` — no personal data, no media files, no model download.
- Golden-master test: regenerated demo output is byte-identical to the committed `chat.md`.
- Hypothesis property tests: day-order sorting with an independent RU-month oracle; cache-key migration idempotence.
- Adversarial parser fixtures covering crash-prone export shapes.
- GitHub Actions CI on Python 3.11–3.14: ruff, mypy, pytest with a 95% coverage gate.
- `--version` flag backed by `importlib.metadata` (falls back to `0.0.0.dev0`).
- Atomic cache writes (temp file + `os.replace`) so interrupted runs never corrupt the cache.
- Per-file transcription failure tolerance with an end-of-run summary (no traceback; the artifact is still written).
- Docs: English README case study and `docs/DESIGN.md` with architecture, decisions, and the testing strategy.

### Changed

- Cache writes batched every 50 files plus on clean exit and Ctrl-C, instead of after every file.
- Partial transcription failures now exit with code 1; the Markdown artifact is still written, but scripts can detect that the run was incomplete.
- Cache keys canonicalized via `Path.resolve()` — results no longer depend on the working directory.
- Python floor widened from 3.14 to 3.11 (ruff/mypy targets, CI matrix, and README updated to match).
- README rewritten from scratch as an English case study.
- Media message captions are now rendered in the Markdown output.
- Code entities now render as inline backticks.
- Cache corruption policy unified in shared `src/cache.py`: warn and start empty, never crash.
- `merge_exports.py` migrates legacy cache keys against each source export root before combining.

### Fixed

- Unparseable day groups sorted to the start of the document instead of the end.
- `format_markdown` output depended on input message order; now canonicalized by message id.
- Media captions were silently dropped from the output.
- Crash on `text: null` inside entity segments.
- Crash on non-dict cache JSON (e.g. a list or string in `_transcripts_cache.json`).
- `IndexError` in `merge_exports.py` on empty exports; empty merges are guarded.
- `migrate_cache_keys` kept dead keys that matched nothing on disk.
- Code entities were rendered as plain text, breaking code formatting in the output.
- Call service messages with an unknown outcome rendered as "Звонок звонок"; the fallback label is now "Неизвестный".

### Removed

- Dead CPU multiprocessing pool (`transcribe_parallel_cpu`, `_worker_transcribe`, `transcribe_all`) and the inert `--workers` flag.
- Dataclass-tautology tests that asserted fields against themselves.
- The `inspect.getsource` archaeology test that pinned implementation details.
- Two dead formatter match arms (`text_link`, `custom_emoji`).

### Security

- Personal chat exports and transcript caches excluded from git via `.gitignore`.
- Committed sample data is fully synthetic — no real conversations or media ship in the repo.
