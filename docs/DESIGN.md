# TelegramAnaL — Design & Decision Record

TelegramAnaL converts a Telegram Desktop chat export (a `result.json` plus its media files) into a single Markdown chat log, transcribing voice messages and round videos ("videokruzhki") to text with faster-whisper. The run is a three-stage pipeline — **parse → transcribe → format** — driven by a small argparse CLI. The design stance is deliberate: correctness and durability of long GPU jobs win over raw speed, and privacy is a hard default (personal exports and transcripts are never meant to enter version control). The test suite is hermetic: it runs without a GPU, without faster-whisper installed, and without network.

## Architecture

```
Telegram Desktop export folder (result.json + media files)
        |
        v
+---------------------- telegram_to_md.py (CLI) ----------------------+
| argparse: export_dir --model --device --output --no-cache          |
|           --language --beam-size                                   |
| _setup_cuda_libs(): ctypes-preloads pip-installed CUDA .so files,  |
| before src.transcriber imports faster-whisper (module-level E402   |
| is the one deliberate ruff exception)                              |
+--------------------------------------------------------------------+
                              |
           Phase 1/2          |          Phase 3
     +----------------+       |    +------------------------------+
     | src.parser     |       |    | src.transcriber              |
     | result.json -> |       |    | WhisperModel loaded ONCE     |
     | Message list,  |       |    | GPU: BatchedInference        |
     | sorted by id   |       |    |   pipeline batch_size=16,    |
     +-------+--------+       |    |   float16                    |
             |                |    | CPU: model.transcribe with   |
             v                |    |   vad_filter=True, int8,     |
     msg.file = canonical     |    |   cpu_threads=nproc/2,       |
     absolute path or None    |    |   num_workers=2              |
             |                |    +-------------+----------------+
             |                |                  |
             |                |                  v
             |                |    transcripts: dict[str, str]
             |                |    { canonical file path: text }
             |                |    path == msg.file == cache key
             |                |                  |
             |                |       +----------+-----------+
             |                |       | <export_dir>/        |
             |                |       | _transcripts_cache.json |
             |                |       | batched flush:       |
             |                |       | every 50 files, exit,|
             |                |       | Ctrl-C               |
             |                |       +----------------------+
             |                |
             v                v
     +------------------------------------------+
     | Phase 4: src.formatter                   |
     | group by calendar day, chronological     |
     | sort (unparseable day groups -> END)     |
     | message -> Markdown, MD escaping,        |
     | reply quotes via {id: Message} index     |
     +--------------------+---------------------+
                          |
                          v
                  chat.md (<export_dir> or --output)
```

`merge_exports.py` is a separate side flow, not a mode of the main CLI:

```
merge_exports.py --old --new [--extra ...]
        |
        v
  parse each export with src.parser
        |
        v
  merge messages by id -- earlier export wins (dict.setdefault)
        |
        v
  merge each export's _transcripts_cache.json (setdefault, keys as-is)
        |
        v
  single format_markdown pass -> --output chat.md (default ./chat.md)
```

## Data flow details

- **Message lifecycle.** `result.json` (JSON) → `src.parser.parse_export` builds `Message` dataclasses and sorts them by id, which is the export's chronological order. The CLI then derives two auxiliary structures from the message list: a reply index `{message_id: Message}` and the list of files to transcribe (messages with `media_type` `voice_message` or `video_message` whose `file` is set).
- **File paths as identity.** `Message.file` is not the raw export-relative string from the JSON. The parser resolves it to a canonical absolute path (`Path.resolve()`) and stores it as a `str`; media placeholders (`"(File not included..."`, `"(File unavailable..."`) and files missing from disk become `None`. The formatter looks transcripts up by `msg.file`, the transcriber's in-memory cache uses the same strings, and the on-disk cache uses them as keys — one canonical string is the join point of all three.
- **Cache.** Persisted at `<export_dir>/_transcripts_cache.json` (compact JSON, `ensure_ascii=False`). `--no-cache` simply passes `cache_dir=None`, so nothing is read or written. Writes are batched (see decision b). The transcriber migrates legacy keys once per load (see decision a); `merge_exports` does not — it reads caches as-is.

## Key decisions & trade-offs

**a) Canonical absolute cache keys + automatic legacy migration** *(755efb8)*
*Decision:* every media path is `(export_root / file).resolve()` before it becomes a `Message.file` and a cache key; on load, `_migrate_cache_keys` re-keys legacy relative entries by dropping leading path components until the remainder points to an existing file.
*Why:* the cache must not depend on the CWD or on how `export_dir` was typed (relative vs. absolute). Running from a different directory must hit the same cache, or long GPU jobs get silently re-run.
*Trade-off:* keys embed the folder location, so physically moving the export folder invalidates entries and forces re-transcription. Accepted; the migration runs once per load and drops entries that match no file.

**b) Batched cache writes, not per-file** *(755efb8)*
*Decision:* the CLI flushes the cache every 50 files, in a `finally` block at the end, and in the `KeyboardInterrupt` handler.
*Why:* the earlier per-file write rewrote the whole dict each time — O(n²) JSON I/O over hundreds of voice messages.
*Trade-off:* a hard kill (`kill -9`) can lose up to 49 completed transcripts. Accepted as the price of keeping I/O negligible during a GPU-bound loop.

**c) Removed the CPU multiprocessing pool and the `--workers` flag** *(7b2e527)*
*Decision:* `transcribe_parallel_cpu` / `_worker_transcribe` and `--workers` were deleted; CPU parallelism is delegated to ctranslate2's internal threads (`cpu_threads=max(1, cpu_count//2)`, `num_workers=2`) inside a single process. CLI tests assert `--workers` is rejected as unrecognized.
*Why:* the worker function constructed a fresh `WhisperModel` per file — catastrophic model-load overhead — and a multiprocessing pool would hold roughly 1.5+ GB of model RAM per worker (int8 `medium`).
*Trade-off:* no true multi-process transcription, so one long CPU run occupies one core-pair budget; an honest CLI is preferred over a fake surface.

**d) Sort invariant: unparseable dates go to the END of the document** *(1dc4ba8)*
*Decision:* day groups are sorted by `(parsed?, datetime)` where parsed days yield `(False, datetime)` and unparseable days yield `(True, raw_key)`.
*Why:* day headers are Russian ("2 июля 2026"), which does not sort lexicographically, so groups must be ordered by parsed date. The fix commit exists because the earlier code sorted unparseable groups to the top even though the comment promised the opposite.
*Trade-off:* a corrupt/malformed date lands at the bottom of the output rather than being rejected — visible but never silently misordered.

**e) Python floor lowered to 3.11** *(0b6f4bf)*
*Decision:* `requires-python = ">=3.11"` (was `>=3.14`), with ruff `target-version` and mypy `python_version` pinned to 3.11; an unused `typing.override` import was removed along the way.
*Why:* nothing left in the codebase needs 3.12+-only syntax, and a 3.14 floor would alienate reviewers and CI runners. A CI matrix over 3.12–3.14 is planned but not yet committed (no `.github/workflows` in the repo today).
*Trade-off:* we forgo the newest syntax conveniences; in exchange the tool builds and tests on any mainstream interpreter.

**f) Root modules shipped for the console script** *(5136e7d)*
*Decision:* `[project.scripts] telegram-to-md = "telegram_to_md:main"` plus `[tool.setuptools] py-modules = ["telegram_to_md", "merge_exports"]`.
*Why:* the script entry point must import `telegram_to_md`, which lives at the repo root rather than inside `src/`; shipping it as a py-module makes the installed wheel work.
*Trade-off:* two packaging mechanisms (root py-modules + `src` package) instead of one uniform layout; running `python telegram_to_md.py` from a checkout keeps working because the file's own directory is on `sys.path`.

**g) Privacy by design**
*Decision:* `.gitignore` excludes `/ChatExport*/` (anchored to the repo root) and `*_transcripts_cache.json` anywhere, plus generated `/chat.md` and `/analysis*` artifacts. The only committed dataset is `examples/sample_export`, whose messages explicitly declare themselves fictional demo data *(ed5fab1)*.
*Why:* Telegram exports and their transcripts are personal data; the repo must stay safe to publish as a portfolio without an audit.
*Trade-off:* sample/example coverage is synthetic only — media files in tests are stubs, so realistic end-to-end runs remain manual.

**h) Media placeholders degrade gracefully**
*Decision:* parser turns `"(File not included..."` / `"(File unavailable..."` and files absent from disk into `file=None`; the formatter then skips transcript lookup instead of crashing.
*Why:* Telegram exports routinely omit media (not-downloaded or unavailable), and those messages must still render as chat history.
*Trade-off:* silent omission hides which media was skipped; the header still marks the media type, so the loss is visible in the output itself.

## Robustness & failure modes

| Failure | Behavior |
|---|---|
| Corrupt/unreadable cache JSON | `src.transcriber` silently starts empty — files are simply re-transcribed (or reported). `merge_exports` is deliberately stricter: its cache read raises `RuntimeError`, because a merge run has no transcription phase, so a silently empty cache would permanently drop transcripts from the output. |
| Missing `result.json` | `parse_export` raises `FileNotFoundError("result.json not found in ...")`; the CLI pre-checks the directory and exits 1 with a clear message. |
| Media file referenced but absent on disk | Resolves to `None`; the message renders with a media label and "(расшифровка недоступна)" instead of crashing. |
| Reply to a deleted message | Counted and reported as a warning ("N ответов ссылаются на удалённые сообщения"); the reply quote is omitted in the document. |
| Merge of exports that are all empty | **Unprotected today:** the statistics block reads `messages[0]` / `messages[-1]` unconditionally and would raise `IndexError`. An empty-export guard is an open TODO. |
| `KeyboardInterrupt` mid-transcription | Cache flushed, progress printed, exit code 1 — completed transcripts survive. |
| faster-whisper not installed | Import is wrapped (`HAS_WHISPER`), so `src.transcriber` stays importable for tests; constructing a `Transcriber` raises `RuntimeError` with a `pip install faster-whisper` hint. |

## Known limitations & future work

- **No per-message permalinks** in the output, even though `t.me` links are derivable from the chat id.
- **No published benchmarks**: real-time-factor and VRAM numbers per model are planned but not yet measured.
- **No automatic CPU fallback**: if CUDA model init fails the run crashes with the underlying error; `--device cpu` is a manual retry. Candidate improvement.
- **Russian-first UI**: console output, day headers, and placeholders are Russian; transcription defaults to `ru`. One `--language` per run — mixed-language chats need a second pass.
- **Whole-document output**: every run regenerates one `chat.md`; there is no incremental formatting.
- **Merge limitations**: `merge_exports` neither transcribes missing files (it only reports them) nor migrates legacy cache keys.
- **Empty-merge guard** from the table above is the most immediate correctness gap.

---

*This document tracks implementation history: each decision cites the commit that landed it, and the per-commit rationale lives in `git log`. When the code and this file disagree, the code and the commits win — update this file.*
