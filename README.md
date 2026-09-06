# TelegramAnaL

Turn a Telegram Desktop chat export into one readable, chronological Markdown file — and transcribe every voice message and round video to text with faster-whisper, locally, no cloud.

<!-- REPLACE USER with your GitHub username after pushing -->
[![CI](https://github.com/USER/TelegramAnaL/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/TelegramAnaL/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The problem

Telegram's built-in export is built for backup, not for reading. You get a folder of month-by-month HTML files plus media: voice notes as `.ogg`, round videos as `.mp4`. Open it later and you're clicking through fragmented pages where the most personal part of the conversation — the voice — is invisible. You can't read it end-to-end, and you can't search it.

This tool closes that gap: point it at an export folder (`result.json` + media) and it produces **one** Markdown document — chronological, readable in any editor, greppable — where voice messages and round videos appear as transcribed text right where they were said.

## Proof, first

This is real output from the committed demo export (`examples/sample_export/`), a synthetic Russian chat with no personal data:

```markdown
## 4 сентября 2026

**09:03** | **Борис**: Я за. Только давай **без опозданий** — в прошлый раз ждали полчаса.

> ↩ В ответ на **Аня** (09:01):
>> Всем привет! Планируем субботний пикник. …
**09:21** | **Аня** (0:14)
**🎤 Голосовое сообщение:**
> *(расшифровка недоступна)*    ← demo ships no audio; real exports show Whisper text here
```

Bold entities survive, replies become quote blocks, voice messages carry duration + transcript. Run it yourself:

```bash
pip install .                    # once — exposes the `telegram-to-md` command
telegram-to-md examples/sample_export
```

That regenerates the file above in under a second: no model download, no network, no GPU. It exercises the real parse → transcribe → format pipeline, just with nothing to transcribe.

## Features

- **One self-contained Markdown file** — messages grouped under Russian day headers (`## 4 сентября 2026`), sorted chronologically, not lexicographically.
- **faster-whisper transcription** of voice messages and round videos — GPU by default via a batched inference pipeline (`batch_size=16`, `float16`), CPU mode with `int8` + VAD filter.
- **Durable transcript cache** — survive Ctrl-C, migrate old cache formats, and never re-transcribe what you already paid for (details below).
- **Replies, forwards, reactions, edits, service messages** — reply quotes with author/time/preview (including a transcript excerpt when the reply target is a voice note), forwarded-from markers, reaction counts with names, "(отредактировано …)" timestamps, and system lines for calls (completed / missed / rejected) and pinned messages.
- **Proper Markdown escaping** — `*`, `_`, `` ` ``, `[`, `]` and `\` in user text never break the document; Telegram entities (bold, italic, mentions, hashtags, phone numbers, links) map to Markdown.
- **Robust to missing media** — Telegram's `(File not included…)` / `(File unavailable…)` placeholders and files absent from disk keep their message in the output; transcription is simply skipped.
- **`merge_exports.py`** — glue overlapping partial exports of the same chat (e.g. from different dates) into one chronological document, deduplicated by message id, with caches combined.

## Install

Python 3.11+.

```bash
pip install .                    # from the repo root; installs the `telegram-to-md` command
pip install -e ".[dev]"          # same + pytest, pytest-cov, ruff, mypy
```

The first real transcription downloads the chosen Whisper weights via faster-whisper (Hugging Face cache) — after that, transcription works offline. The demo export never triggers a download.

## Usage

```bash
telegram-to-md "ChatExport_2026-07-24 (1)/"
telegram-to-md "ChatExport_2026-07-24 (1)/" --model small --output chat.md
telegram-to-md "ChatExport_2026-07-24 (1)/" --device cpu
```

| Argument | Meaning | Default |
|---|---|---|
| `export_dir` (positional) | Telegram Desktop export folder containing `result.json` | required |
| `--model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, `turbo` | `medium` |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--output` | Output Markdown path | `<export_dir>/chat.md` |
| `--no-cache` | Skip reading *and* writing the transcript cache | off |
| `--language` | Transcription language code | `ru` |
| `--beam-size` | Whisper beam-search width | `5` |

There is deliberately **no `--workers` and no `--cpu` flag** — they were removed in a refactor. CPU mode is chosen with `--device cpu`; parallelism lives inside ctranslate2 (`cpu_threads` defaults to half your cores, `num_workers=2`). On CUDA, pip-installed NVIDIA libraries are preloaded automatically so ctranslate2 can find them.

## How it works

**1. Parse.** `src/parser.py` reads `result.json` into typed `Message` dataclasses (`src/models.py`). Media paths are resolved to canonical absolute paths — the same key doubles as the transcription-cache key, so results never depend on where you ran the command from.

**2. Transcribe.** The CLI collects every voice message and round video, loads the Whisper model once, and transcribes each file — skipping cache hits. Progress goes through tqdm; a Ctrl-C flushes the cache so hours of GPU work are never lost.

**3. Format.** `src/formatter.py` renders messages to Markdown: day groups with Russian headers in true chronological order, reply quote blocks, media markers, reactions, edits, service messages. One file is written to `<export_dir>/chat.md` (or your `--output`).

## The transcript cache

Stored next to the export as `<export_dir>/_transcripts_cache.json`:

- **Keys are canonical absolute paths** (via `Path.resolve()`), so running from a different directory, with `./` prefixes, or with absolute paths always hits the same cache.
- **Moving the export folder invalidates the cache** — keys embed the folder's absolute location, so expect a one-time re-transcription after a relocation.
- **Legacy caches migrate automatically.** Old relative keys (`ChatExport_x/voice_messages/audio.ogg`) are re-matched to files on disk and re-keyed; entries matching nothing are dropped (they'd be re-transcribed anyway).
- **Writes are batched** — every 50 files, on clean finish, and on Ctrl-C — so an interrupted run keeps everything already transcribed.
- `--no-cache` turns the whole mechanism off.

## Project layout

```
telegram_to_md.py        CLI entry point: args, Parse → Transcribe → Format, cache flushing
src/parser.py            result.json → Message objects; canonical media path resolution
src/models.py            dataclasses: Message, TextEntity, Reaction, ReactionRecent, LocationInfo
src/transcriber.py       faster-whisper GPU/CPU transcription; cache load/migrate/flush
src/formatter.py         Message → Markdown: day groups, replies, service messages, entities
merge_exports.py         merge overlapping exports of one chat; dedup by id; combine caches
examples/sample_export/  synthetic demo export + its generated chat.md (privacy-safe)
tests/                   210 hermetic tests — no audio, no model, no network
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/          # 210 tests, hermetic: model & decoding are stubbed
.venv/bin/python -m pytest --cov           # coverage gate: fail_under 95% on src, per pyproject.toml
.venv/bin/ruff check .
.venv/bin/mypy src telegram_to_md.py merge_exports.py
```

The `.[dev]` extra installs pytest, ruff, mypy, pytest-cov, and Hypothesis. Lint and type rules live in `pyproject.toml` (ruff, strict mypy), and the coverage gate is configured there too: `[tool.coverage.run] source = ["src"]` with `fail_under = 95` — so `pytest --cov` fails the run below 95%. Property tests (day-order sorting, cache-key migration) and one-shot mutmut audits (last pass: 768/768 mutants killed) are described in docs/DESIGN.md.

Note: this project was developed with active use of an AI assistant (the deepseek-v4-flash model).

## Limitations, honestly

- **The document chrome is Russian-first**: day headers use Russian month names and console output is in Russian. Chat content is untouched, but the tool is not localized.
- **Whisper can hallucinate** on silence or noise. A VAD filter mitigates this on the CPU path; the GPU batched pipeline currently doesn't apply VAD (a faster-whisper limitation), which is a trade-off for speed.
- **No automatic CUDA→CPU fallback** — if model init fails on CUDA the run stops with exit code 1 and a hint; retry with `--device cpu`.
- **The output is a snapshot**: no per-message anchors/permalink ids — search the file instead. It is one-way; don't expect to round-trip back into Telegram.
- **GPU is the default device** and recommended for large chats (`--device cpu` works but is slower). The `medium` default is a reasonable speed/quality midpoint — pick `small` for speed, `large-v3` for quality.
- **`merge_exports.py` combines caches, it doesn't fill them** — files missing from every source export are reported in the statistics and stay without transcripts (there is no re-transcription pass).

## Privacy

Real Telegram exports contain private conversations. The repo's `.gitignore` excludes `ChatExport*/` folders and `*_transcripts_cache.json`, and the committed `examples/sample_export/` is entirely synthetic. Keep your own exports out of git — and remember transcripts sit in plain text next to the export.

## License

[MIT](LICENSE). Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [tqdm](https://github.com/tqdm/tqdm) — all processing happens on your machine, nothing is uploaded anywhere.
