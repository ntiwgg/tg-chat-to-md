# tg-chat-to-md — Design & Decision Record

tg-chat-to-md converts a Telegram Desktop chat export (a `result.json` plus its media files) into a single Markdown chat log, transcribing voice messages and round videos ("videokruzhki") to text with faster-whisper. The run is a three-stage pipeline — **parse → transcribe → format** — driven by a small argparse CLI (`tg-chat-to-md`, package module `tg_chat_to_md/cli.py`). The design stance is deliberate: correctness and durability of long GPU jobs win over raw speed, and privacy is a hard default (personal exports and transcripts are never meant to enter version control). The test suite is hermetic: it runs without a GPU, without faster-whisper installed, and without network.

## Architecture

```
Telegram Desktop export folder (result.json + media files)
        |
        v
+------------------- tg_chat_to_md/cli.py (CLI) --------------------+
| argparse: export_dir --model --device --output --no-cache         |
|           --language --beam-size                                  |
| _setup_cuda_libs(): ctypes-preloads pip-installed CUDA .so files, |
| before tg_chat_to_md.transcriber imports faster-whisper (module-  |
| level E402 is the one deliberate ruff exception)                  |
+-------------------------------------------------------------------+
                              |
           Phase 1/2          |          Phase 3
     +----------------+       |    +----------------------------+
     | tg_chat_to_md. |       |    | tg_chat_to_md.transcriber  |
     | parser         |       |    | WhisperModel loaded ONCE   |
     | result.json -> |       |    | GPU: BatchedInference      |
     | Message list,  |       |    |   pipeline batch_size=16,  |
     | sorted by id   |       |    |   float16                  |
     +-------+--------+       |    | CPU: model.transcribe with |
             |                |    |   vad_filter=True, int8,   |
             v                |    |   cpu_threads=nproc/2,     |
     msg.file = canonical     |    |   num_workers=2            |
     absolute path or None    |    +-------------+--------------+
             |                |                  |
             |                |                  v
             |                |    transcripts: dict[str, str]
             |                |    { canonical file path: text }
             |                |    path == msg.file == cache key
             |                |                  |
              |               |        +----------+-----------+
              |               |        | <export_dir>/        |
              |               |        | _transcripts_cache.json |
              |               |        | I/O: tg_chat_to_md.  |
              |               |        |   cache (shared)     |
              |               |        | batched flush:       |
              |               |        | every 50 files, exit,|
              |               |        | Ctrl-C               |
              |               |        +----------------------+
             |                |
             v                v
+------------------- Phase 4: tg_chat_to_md/formatter.py -------------------+
| group by calendar day, chronological                                      |
| sort (unparseable day groups -> END)                                      |
| message -> Markdown, MD escaping,                                         |
| reply quotes via {id: Message} index                                      |
+----------------------------------------------------------------------------+
                          |
                          v
                  chat.md (<export_dir> or --output)
```

The merge helper (`tg_chat_to_md/merge.py`) is a separate side flow, not a mode of the main CLI:

```
tg-chat-merge --old --new [--extra ...]     # or: python -m tg_chat_to_md.merge
        |
        v
  parse each export with tg_chat_to_md.parser
        |
        v
  merge messages by id -- earlier export wins (dict.setdefault)
        |
        v
  merge transcripts: each export's cache read via shared
  tg_chat_to_md.cache, keys migrated against that export's own
  root, then setdefault-merged
        |
        v
  single format_markdown pass -> --output chat.md (default ./chat.md)
```

## Data flow details

- **Message lifecycle.** `result.json` (JSON) → `tg_chat_to_md.parser.parse_export` builds `Message` dataclasses and sorts them by id, which is the export's chronological order. The CLI then derives two auxiliary structures from the message list: a reply index `{message_id: Message}` and the list of files to transcribe (messages with `media_type` `voice_message` or `video_message` whose `file` is set).
- **File paths as identity.** `Message.file` is not the raw export-relative string from the JSON. The parser resolves it to a canonical absolute path (`Path.resolve()`) and stores it as a `str`; media placeholders (`"(File not included..."`, `"(File unavailable..."`) and files missing from disk become `None`. The formatter looks transcripts up by `msg.file`, the transcriber's in-memory cache uses the same strings, and the on-disk cache uses them as keys — one canonical string is the join point of all three.
- **Cache.** Persisted at `<export_dir>/_transcripts_cache.json` (compact JSON, `ensure_ascii=False`). All cache I/O is implemented once, in the shared module `tg_chat_to_md/cache.py` (`read_cache`, `write_cache`, `migrate_cache_keys`, `CACHE_FILE_NAME`), which both `tg_chat_to_md.transcriber` and `tg_chat_to_md.merge` import *(6bf73ea)* — a past bug was the two tools drifting apart on cache semantics. `--no-cache` simply passes `cache_dir=None`, so nothing is read or written. Writes are batched (see decision b). Both consumers migrate legacy keys on load (see decision a): the transcriber against its export root, the merge helper against each source export's own root.

## Key decisions & trade-offs

**a) Canonical absolute cache keys + automatic legacy migration** *(755efb8, generalized in 6bf73ea)*
*Decision:* every media path is `(export_root / file).resolve()` before it becomes a `Message.file` and a cache key. Cache I/O and key migration live in the shared module `tg_chat_to_md/cache.py` — `read_cache`, `write_cache`, `migrate_cache_keys` — used by `tg_chat_to_md.transcriber` and the merge helper alike *(6bf73ea)*. On load, `migrate_cache_keys` re-keys legacy relative entries by dropping leading path components until the remainder points to an existing file; the merge helper runs it per source export, against that export's own root, before merging the caches.
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
*Why:* nothing left in the codebase needs 3.12+-only syntax, and a 3.14 floor would alienate reviewers and CI runners. GitHub Actions CI is committed (`.github/workflows/ci.yml`) and runs the ruff check, strict mypy, and pytest with the 95% coverage gate on Python 3.11–3.14 *(e40ee96)*.
*Trade-off:* we forgo the newest syntax conveniences; in exchange the tool builds and tests on any mainstream interpreter.

**f) One package, two console scripts** *(0.1.0 rename)*
*Decision:* all modules live in the single `tg_chat_to_md` package at the repo root (no py-modules, no `src/`): `[project.scripts]` maps `tg-chat-to-md = "tg_chat_to_md.cli:main"` and `tg-chat-merge = "tg_chat_to_md.merge:main"`; `python -m tg_chat_to_md` runs the main CLI via `tg_chat_to_md/__main__.py`.
*Why:* the project was renamed from `telegramanal` (repo `TelegramAnaL`) to `tg-chat-to-md`, and the code moved out of a top-level package literally named `src` into the importable distribution package — the earlier layout shipped the CLI and merge helper as root py-modules so their console scripts could import them, which split the code between two packaging mechanisms.
*Trade-off:* `tg_chat_to_md/cli.py` preloads CUDA libraries at import time (module-level E402 is the one deliberate ruff exception), so the package `__init__.py` deliberately does not import the CLI — importing `tg_chat_to_md` stays lightweight and never touches faster-whisper.

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
| Corrupt/unreadable cache JSON | Both tools share the `tg_chat_to_md/cache.py` read path: the cache degrades to empty with a warning on stderr — never an exception *(6bf73ea)*. The transcriber re-transcribes the affected files (or reports them); the merge helper proceeds with the surviving caches and counts the missing transcripts in its statistics. |
| Cache write interrupted mid-persist | Writes are atomic — payload fully serialized, written to a `.tmp` sibling, then `os.replace`d over the target *(dee5a2f)* — so a crash or `kill -9` can never leave a truncated cache behind; readers only ever see the old or the new file. A failed flush degrades to a stderr warning and the run continues (the cache is an optimization, not a source of truth); the CLI's `finally`/`KeyboardInterrupt` paths still flush best-effort. |
| Missing `result.json` | `parse_export` raises `FileNotFoundError("result.json not found in ...")`; the CLI pre-checks the directory and exits 1 with a clear message. |
| Media file referenced but absent on disk | Resolves to `None`; the message renders with a media label and "(расшифровка недоступна)" instead of crashing. |
| Per-file transcription failure | The file is skipped with a warning; the rest are transcribed, the Markdown artifact is still written with the transcripts that succeeded, and the run exits 1 — a scriptable signal that the run was incomplete (a fully successful run exits 0) *(88e8476)*. |
| Reply to a deleted message | Counted and reported as a warning ("N ответов ссылаются на удалённые сообщения"); the reply quote is omitted in the document. |
| Merge of exports that are all empty | Guarded: the statistics block reads `messages[0]` / `messages[-1]` only when the merged list is non-empty; an all-empty merge reports `нет сообщений (пустой результат)` and finishes cleanly *(6bf73ea)*. |
| `KeyboardInterrupt` mid-transcription | Cache flushed, progress printed, exit code 1 — completed transcripts survive. |
| faster-whisper not installed | Import is wrapped (`HAS_WHISPER`), so `tg_chat_to_md.transcriber` stays importable for tests; constructing a `Transcriber` raises `RuntimeError` with a `pip install faster-whisper` hint. |

## Testing strategy

The suite's job is to make the pipeline's real behaviors unregressable, not to inflate a percentage. The layers below form the actual quality loop; everything runs hermetically (no audio files, no model downloads, no network — see the hermeticity rule at the end).

- **Coverage is a floor, not a goal.** `fail_under = 95` on `tg_chat_to_md/` is the gate that stops a commit from silently losing whole code paths (Fowler/Marick: coverage measures what ran, not what is checked). The 95% floor buys nothing by itself — the checks below are what catch regressions; the floor only ensures new code enters the checked world.
- **Golden master.** `examples/sample_export/chat.md` is the committed contract artifact. `tests/test_golden.py` re-parses the sample and re-formats it, asserting byte-identical output for any input order *(1e93b15)*. Updating the golden file is a deliberate act: regenerate only via a human-reviewed diff (ApprovalTests workflow), never silently — a changed golden is either a reviewed format change or a caught regression.
- **Property tests for the historically broken invariants.** Day-group ordering (independent RU-month oracle; unparseable groups sort last) and cache-key migration (idempotence; absolute, existing output keys) are the two classes that actually regressed in this codebase *(1dc4ba8, 755efb8)*. Hypothesis makes them structurally unregressable: instead of one hand-picked example, hundreds of generated inputs must satisfy the invariant *(dee5a2f)*.
- **Mutation loop (mutmut).** One-shot audits mutate `tg_chat_to_md/` and run the real suite against every mutant (config: `[tool.mutmut]` in `pyproject.toml`; the `mutants/` sandbox is gitignored). Baseline pass: 868 mutants, 610 killed (70.3%). Every survivor was triaged: productive ones (a mutated comparison/range/condition no test noticed — e.g. parser field lookups, transcriber call arguments, caption rendering on sticker/GIF/file branches) got a new exact-oracle test; genuinely unproductive ones (RU label copy, locale-encoding choices, dead branches, provably equivalent conditions) got a `# pragma: no mutate` on the exact statement line; two no-op match arms were deleted. Final pass: **768/768 mutants killed (100%)** *(23175ca)*. Two caveats, both by design: the score is a review input, not a CI gate (Google practice — raw percentages gate nothing, the triage is the value), and mutmut does not mutate decorated functions, so `tg_chat_to_md/models.py` (dataclasses + `@property` getters) generates no mutants and stays guarded by its direct unit tests instead. The package lives under a normal importable name now, so mutmut runs out of the box with no venv adaptation.
- **Hermeticity rule.** Tests never load a model, decode audio, or touch the network. The only legitimate seam is the faster-whisper model itself: `tests/test_transcriber.py` monkeypatches `HAS_WHISPER`/`WhisperModel`/`BatchedInferencePipeline` with recorders (the documented duck contract) and runs the real `Transcriber.__init__`/`transcribe` paths, asserting exact call arguments. Everything else runs through real code and the real filesystem (`tmp_path`).
- **Failure-mode discipline (SQLite precedent).** Malformed input degrades with a warning instead of crashing: syntax-broken and non-dict cache JSON read as empty *(6bf73ea)*, adversarial parser fixtures (`tests/fixtures/*.json`) cover forwarded/reply/action/text-shape anomalies, atomic cache writes have crash-consistency tests (a failing write leaves the old cache intact and no `.tmp` litter), and per-file transcription failure tolerance is asserted at the CLI level.
- **Determinism.** `format_markdown` canonicalizes by message id, and the golden tests shuffle both the message list and the transcript dict to prove byte-identity. No clock, no `random` (seeded shuffles only), no `sleep` anywhere in the suite — a failure is reproducible on the first rerun.

## Known limitations & future work

- **No per-message permalinks** in the output, even though `t.me` links are derivable from the chat id.
- **No published benchmarks**: real-time-factor and VRAM numbers per model are planned but not yet measured.
- **No automatic CUDA→CPU fallback**: if CUDA model init fails, the run exits 1 with a friendly error and a `--device cpu` hint instead of a traceback *(bd80002)*; retrying manually is required. Candidate improvement.
- **GPU path skips the VAD filter**: the batched GPU pipeline runs without the VAD filter the CPU path applies (`vad_filter=True`), so silence/noise can hallucinate more readily in fast batched runs — a deliberate speed trade-off.
- **Moving the export folder invalidates the cache**: keys embed each file's canonical absolute path (decision a), so a relocation forces a one-time re-transcription.
- **Russian-first UI**: console output, day headers, and placeholders are Russian; transcription defaults to `ru`. One `--language` per run — mixed-language chats need a second pass.
- **Whole-document output**: every run regenerates one `chat.md`; there is no incremental formatting.
- **Merge limitations**: the merge helper does not transcribe missing files — it only reports them in the statistics, so files absent from every source cache stay without transcripts.

---

*This document tracks implementation history: each decision cites the commit that landed it, and the per-commit rationale lives in `git log`. When the code and this file disagree, the code and the commits win — update this file.*
