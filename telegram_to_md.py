#!/usr/bin/env python3
"""Telegram Chat Export → Markdown converter with speech-to-text.

Usage:
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/"
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/" --model small --output chat.md
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/" --device cpu --model small

Options:
    --model        Whisper model size: tiny, base, small, medium, large-v3, turbo (default: medium)
    --device       Compute device: cuda or cpu (default: cuda)
    --output       Output markdown path (default: <export_dir>/chat.md)
    --no-cache     Disable transcript caching
    --language     Language code for transcription (default: ru)
    --beam-size    Whisper beam size (default: 5)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Flush the transcript cache to disk after every N transcribed files (plus on
# exit and on KeyboardInterrupt). Kept as a module constant so tests can
# shrink the cadence instead of transcribing 50 files.
CACHE_FLUSH_EVERY = 50


def _cli_version() -> str:
    """Version of the installed distribution; dev fallback when run unpackaged."""
    try:
        return version("telegramanal")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def _setup_cuda_libs() -> None:
    """Preload pip-installed CUDA libraries so ctranslate2 can find them.

    Setting LD_LIBRARY_PATH via os.environ does NOT work after process start —
    the dynamic linker reads it only at exec time. We must preload with ctypes.
    """
    import ctypes  # noqa: PLC0415
    import glob  # noqa: PLC0415

    # Find site-packages relative to the venv Python
    sp_candidates = [
        os.path.join(os.path.dirname(sys.executable), "..", "lib"),
        os.path.join(sys.prefix, "lib"),
    ]
    site_packages = None
    for base in sp_candidates:
        if not os.path.isdir(base):
            continue
        for root, _dirs, _files in os.walk(base):
            if root.endswith("site-packages"):
                site_packages = root
                break
        if site_packages:
            break

    if not site_packages:
        return

    # These are the specific libs ctranslate2 needs at runtime
    patterns = [
        "**/nvidia/cublas/lib/libcublas.so*",
        "**/nvidia/cublas/lib/libcublasLt.so*",
        "**/nvidia/cudnn/lib/libcudnn.so*",
        "**/nvidia/cuda_nvrtc/lib/libnvrtc*.so*",
    ]
    loaded = 0
    for pat in patterns:
        for full in glob.glob(pat, root_dir=site_packages, recursive=True):
            full_path = os.path.join(site_packages, full)
            if os.path.isfile(full_path):
                try:
                    ctypes.CDLL(full_path)
                    loaded += 1
                except OSError:
                    pass

    if loaded:
        print(f"   ✓ Предзагружено {loaded} CUDA-библиотек", file=sys.stderr)


_setup_cuda_libs()

from src.formatter import format_markdown
from src.parser import parse_export
from src.transcriber import Transcriber


def _flush_cache_or_warn(transcriber: Transcriber) -> None:
    """Flush the transcript cache; warn on failure instead of raising.

    The cache is a performance optimization, not source data: a flush error
    (e.g. a full disk) must not abort transcription that already succeeded —
    the in-memory transcripts still reach the Markdown — and must never turn
    the Ctrl-C durability flush (except/finally paths below) into an
    unexpected traceback. A later flush may succeed once the condition clears.
    """
    try:
        transcriber.flush_cache()
    except OSError as exc:
        print(f"⚠ Не удалось сохранить кэш расшифровок: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Telegram Chat Export to a single Markdown file "
            "with voice/video transcription."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_cli_version()}"
    )
    parser.add_argument(
        "export_dir",
        help="Path to the Telegram Chat Export directory (contains result.json)",
    )
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output markdown path (default: <export_dir>/chat.md)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable transcript caching",
    )
    parser.add_argument(
        "--language", default="ru",
        help="Language code for transcription (default: ru)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5,
        help="Whisper beam size (default: 5)",
    )

    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"Ошибка: директория не найдена: {export_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (export_dir / "chat.md")

    # ------------------------------------------------------------------
    # Phase 1: Parse
    # ------------------------------------------------------------------
    print(f"📖 Читаю {export_dir / 'result.json'}…")
    t0 = time.monotonic()
    try:
        chat_name, chat_id, messages = parse_export(export_dir)
    except FileNotFoundError:
        print(
            f"Ошибка: в папке {export_dir} нет файла result.json.\n"
            "   Укажите путь к папке экспорта Telegram Desktop "
            "(внутри неё лежит result.json).",
            file=sys.stderr,
        )
        sys.exit(1)
    t1 = time.monotonic()
    print(f"   ✓ {len(messages)} сообщений ({t1 - t0:.1f}с)")

    # Build message index for reply resolution
    msg_index = {m.id: m for m in messages}
    missing_replies = sum(
        1 for m in messages if m.reply_to_message_id and m.reply_to_message_id not in msg_index
    )
    if missing_replies:
        print(f"   ⚠ {missing_replies} ответов ссылаются на удалённые сообщения")

    # ------------------------------------------------------------------
    # Phase 2: Find media files to transcribe
    # ------------------------------------------------------------------
    to_transcribe: list[str] = []
    for m in messages:
        if (m.is_voice or m.is_round_video) and m.file:
            to_transcribe.append(m.file)

    print(f"\n🎙️  Медиа для расшифровки: {len(to_transcribe)} файлов")
    voice_count = sum(1 for m in messages if m.is_voice and m.file)
    video_count = sum(1 for m in messages if m.is_round_video and m.file)
    print(f"   Голосовых: {voice_count}, видеокружков: {video_count}")

    # ------------------------------------------------------------------
    # Phase 3: Transcribe
    # ------------------------------------------------------------------
    transcripts: dict[str, str] = {}
    failures: list[str] = []

    if to_transcribe:
        cache_dir = None if args.no_cache else export_dir
        print(f"\n🔊 Загружаю модель '{args.model}' на {args.device}…")

        try:
            transcriber = Transcriber(
                model_size=args.model,
                device=args.device,
                language=args.language,
                beam_size=args.beam_size,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 — any model-load failure is user-facing
            print(
                f"Ошибка: не удалось загрузить модель '{args.model}' "
                f"на устройстве '{args.device}': {exc}",
                file=sys.stderr,
            )
            print(
                "   Совет: попробуйте --device cpu или установите CUDA-библиотеки "
                "(см. README, раздел «Требования»).",
                file=sys.stderr,
            )
            sys.exit(1)
        t2 = time.monotonic()
        print(f"   ✓ Модель готова ({t2 - t1:.1f}с)")

        # Check cache hits (public stats API — never touch transcriber._cache)
        if not args.no_cache:
            cached = transcriber.cached_count(to_transcribe)
            if cached:
                print(f"   📦 {cached} файлов уже в кэше")

        missing = transcriber.missing_from_cache(to_transcribe)
        if missing:
            print(f"\n🎤 Расшифровываю {len(missing)} файлов…")

        from tqdm import tqdm  # type: ignore[import-untyped]

        try:
            for i, fp in enumerate(tqdm(to_transcribe, desc="Транскрибация", unit="файл")):
                try:
                    text = transcriber.transcribe(fp)
                except Exception as exc:  # noqa: BLE001 — skip one bad file, keep going
                    failures.append(fp)
                    print(
                        f"\n   ⚠ Не удалось расшифровать {Path(fp).name}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                transcripts[fp] = text
                if (i + 1) % CACHE_FLUSH_EVERY == 0:
                    _flush_cache_or_warn(transcriber)
        except KeyboardInterrupt:
            print("\n⚠ Прервано пользователем. Сохраняю кэш…")
            _flush_cache_or_warn(transcriber)
            print(f"   Кэш сохранён. Прогресс: {len(transcripts)}/{len(to_transcribe)}")
            sys.exit(1)
        finally:
            _flush_cache_or_warn(transcriber)

        t3 = time.monotonic()
        print(f"\n   ✓ Расшифровано за {t3 - t2:.1f}с")
        if failures:
            print(
                f"   ⚠ Не удалось расшифровать {len(failures)} из {len(to_transcribe)} файлов; "
                "они останутся без расшифровки в Markdown.",
                file=sys.stderr,
            )
    else:
        print("   Нет файлов для расшифровки")

    # ------------------------------------------------------------------
    # Phase 4: Format & write
    # ------------------------------------------------------------------
    print(f"\n📝 Форматирую markdown → {output_path}…")
    t4 = time.monotonic()

    md_content = format_markdown(messages, chat_name, transcripts, msg_index)
    output_path.write_text(md_content, encoding="utf-8")

    t5 = time.monotonic()
    size_kb = output_path.stat().st_size / 1024
    print(f"   ✓ Готово: {output_path} ({size_kb:.0f} КБ, {t5 - t4:.1f}с)")

    total = t5 - t0
    print(f"\n✅ Завершено за {total:.1f}с ({total/60:.1f} мин)")

    # Some files never made it into transcripts: the artifact is written but
    # the run was incomplete. Exit non-zero so scripts can tell a partial run
    # from a fully successful one (a clean run falls through to exit 0).
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
