#!/usr/bin/env python3
"""Merge any number of Telegram Chat Exports into a single Markdown file.

Merge order: --old, --new, --extra... — earlier exports win on id collisions.

Usage:
    python merge_exports.py --old "ChatExport_2026-07-24 (1)" \
                            --new "ChatExport_2026-08-10 (1)" \
                            --extra "ChatExport_2026-08-10 (2)" \
                            --output chat.md

The --extra flag may be repeated to merge additional exports after --new.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cache import CACHE_FILE_NAME, migrate_cache_keys, read_cache
from src.formatter import format_markdown
from src.models import Message
from src.parser import parse_export


def _merge_messages(*exports: list[Message]) -> list[Message]:
    """Merge by id: earlier exports win on collision. Sorted by id."""
    merged: dict[int, Message] = {}
    for messages in exports:
        for m in messages:
            merged.setdefault(m.id, m)
    return [merged[mid] for mid in sorted(merged)]


def _merge_caches(*caches: dict[str, str]) -> dict[str, str]:
    """Merge transcript caches; existing keys keep their old value."""
    merged: dict[str, str] = {}
    for cache in caches:
        for key, value in cache.items():
            merged.setdefault(key, value)
    return merged


def _missing_transcripts(
    messages: list[Message], transcripts: dict[str, str]
) -> list[Message]:
    """Voice/video messages whose file exists but has no transcript in the cache."""
    return [
        m for m in messages
        if (m.is_voice or m.is_round_video) and m.file and m.file not in transcripts
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Telegram exports into one chronological Markdown file: "
                    "--old, --new, then any number of --extra (earlier wins on id collisions)."
    )
    parser.add_argument("--old", required=True, help="Oldest export directory (relative path)")
    parser.add_argument("--new", required=True, help="Newer export directory (relative path)")
    parser.add_argument("--extra", action="append", default=[], help=(
        "Additional export directory; may be repeated (merged after --new)"
    ))
    parser.add_argument(
        "--output", default="chat.md",
        help="Output markdown path (default: chat.md)",
    )
    args = parser.parse_args()

    export_dirs = [args.old, args.new, *args.extra]
    export_labels = ["старый", "новый", *[f"доп. {i}" for i in range(1, len(args.extra) + 1)]]

    # ------------------------------------------------------------------
    # Parse every export
    # ------------------------------------------------------------------
    parsed = [parse_export(d) for d in export_dirs]
    for (_, _, msgs), label, d in zip(
        parsed, export_labels, export_dirs, strict=True
    ):
        print(f"📖 {label.capitalize()} экспорт: {len(msgs)} сообщений ({d})")

    chat_ids = {chat_id for _, chat_id, _ in parsed}
    if len(chat_ids) > 1:
        print("   ⚠ Разные id чатов: " + ", ".join(str(i) for i in sorted(chat_ids)))

    # ------------------------------------------------------------------
    # Merge messages and caches
    # ------------------------------------------------------------------
    all_messages = [msgs for _, _, msgs in parsed]
    messages = _merge_messages(*all_messages)
    dropped_duplicates = sum(len(msgs) for msgs in all_messages) - len(messages)

    # Each cache is migrated against the export directory it came from (its
    # own root), so legacy export-relative keys still land on the canonical
    # absolute keys that parsed messages use. Corrupt or missing caches read
    # as empty (with a stderr warning from read_cache) — transcripts are then
    # simply absent, and the statistics below report them as missing.
    caches = [
        migrate_cache_keys(read_cache(Path(d) / CACHE_FILE_NAME), Path(d))
        for d in export_dirs
    ]
    cache_summary = ", ".join(
        f"{label} {len(cache)} ключей"
        for label, cache in zip(export_labels, caches, strict=True)
    )
    print(f"   Кэш расшифровок: {cache_summary}")
    transcripts = _merge_caches(*caches)

    # ------------------------------------------------------------------
    # Format and write
    # ------------------------------------------------------------------
    msg_index = {m.id: m for m in messages}
    chat_name = next((name for name, _, _ in parsed if name), "Unknown Chat")
    md_content = format_markdown(messages, chat_name, transcripts, msg_index)

    output_path = Path(args.output)
    output_path.write_text(md_content, encoding="utf-8")
    print(f"💾 Записан {output_path} ({output_path.stat().st_size / 1024:.0f} КБ)")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    if messages:
        first, last = messages[0], messages[-1]
        date_range = f"{first.date} — {last.date}"
    else:
        date_range = "нет сообщений (пустой результат)"
    voice_total = sum(1 for m in messages if m.is_voice and m.file)
    video_total = sum(1 for m in messages if m.is_round_video and m.file)
    missing = _missing_transcripts(messages, transcripts)
    missing_voice = sum(1 for m in missing if m.is_voice)
    missing_video = sum(1 for m in missing if m.is_round_video)

    print()
    print("📊 Статистика:")
    for label, msgs in zip(export_labels, all_messages, strict=True):
        print(f"   Сообщений в {label} экспорте: {len(msgs)}")
    print(f"   Всего уникальных сообщений: {len(messages)}")
    print(f"   Дублей отброшено (id уже был в более раннем экспорте): {dropped_duplicates}")
    print(f"   Диапазон дат: {date_range}")
    print(f"   Голосовых/видеокружков: {voice_total} голосовых, {video_total} видеокружков")
    print(
        f"   Без расшифровки: {len(missing)} "
        f"(из них голосовых {missing_voice}, видеокружков {missing_video})"
    )


if __name__ == "__main__":
    main()
