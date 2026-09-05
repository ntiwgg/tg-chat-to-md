#!/usr/bin/env python3
"""Telegram Chat Export → Markdown converter with speech-to-text.

Usage:
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/"
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/" --model medium --output chat.md
    python telegram_to_md.py "ChatExport_2026-07-24 (1)/" --cpu --workers 8

Options:
    --model        Whisper model size: tiny, base, small, medium, large-v3 (default: medium)
    --device       cuda or cpu (default: cuda)
    --output       Output markdown path (default: {export_dir}/chat.md)
    --no-cache     Disable transcript caching
    --workers      Number of CPU workers when --cpu is used (default: cpu_count - 1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


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
        for root, dirs, _files in os.walk(base):
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

from src.parser import parse_export
from src.transcriber import Transcriber
from src.formatter import format_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Telegram Chat Export to a single Markdown file with voice/video transcription.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--workers", type=int, default=None,
        help="Number of CPU workers when --device cpu (default: cpu_count - 1)",
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
    chat_name, chat_id, messages = parse_export(export_dir)
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

    if to_transcribe:
        cache_dir = None if args.no_cache else export_dir
        print(f"\n🔊 Загружаю модель '{args.model}' на {args.device}…")

        transcriber = Transcriber(
            model_size=args.model,
            device=args.device,
            language=args.language,
            beam_size=args.beam_size,
            cache_dir=cache_dir,
        )
        t2 = time.monotonic()
        print(f"   ✓ Модель готова ({t2 - t1:.1f}с)")

        # Check cache hits
        if not args.no_cache:
            cached = sum(1 for fp in to_transcribe if fp in transcriber._cache)
            if cached:
                print(f"   📦 {cached} файлов уже в кэше")

        remaining = [fp for fp in to_transcribe if fp not in transcriber._cache]
        if remaining:
            print(f"\n🎤 Расшифровываю {len(remaining)} файлов…")

        from tqdm import tqdm  # type: ignore[import-untyped]

        try:
            for i, fp in enumerate(tqdm(to_transcribe, desc="Транскрибация", unit="файл")):
                text = transcriber.transcribe(fp)
                transcripts[fp] = text
                if (i + 1) % 50 == 0:
                    transcriber.flush_cache()
        except KeyboardInterrupt:
            print("\n⚠ Прервано пользователем. Сохраняю кэш…")
            transcriber.flush_cache()
            print(f"   Кэш сохранён. Прогресс: {len(transcripts)}/{len(to_transcribe)}")
            sys.exit(1)
        finally:
            transcriber.flush_cache()

        t3 = time.monotonic()
        print(f"\n   ✓ Расшифровано за {t3 - t2:.1f}с")
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


if __name__ == "__main__":
    main()
