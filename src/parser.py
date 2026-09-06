"""Parse Telegram Chat Export result.json into Message objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import LocationInfo, Message, Reaction, ReactionRecent, TextEntity


def parse_export(export_dir: str | Path) -> tuple[str, int, list[Message]]:
    """Parse result.json and return (chat_name, chat_id, sorted messages).

    Messages are sorted by id (chronological order within the export).
    """
    export_path = Path(export_dir) / "result.json"
    if not export_path.exists():
        raise FileNotFoundError(f"result.json not found in {export_dir}")

    with open(export_path, encoding="utf-8") as f:
        raw = json.load(f)

    chat_name = raw.get("name", "Unknown Chat")
    chat_id = raw.get("id", 0)
    raw_messages: list[dict[str, Any]] = raw.get("messages", [])

    messages = [_parse_message(m, export_path.parent) for m in raw_messages]
    messages.sort(key=lambda m: m.id)

    return chat_name, chat_id, messages


def _parse_message(raw: dict[str, Any], export_root: Path) -> Message:
    """Parse a single message dict into a Message object."""
    return Message(
        id=raw["id"],
        type=raw["type"],
        date=raw.get("date", ""),
        date_unixtime=raw.get("date_unixtime", ""),
        from_name=raw.get("from"),
        from_id=raw.get("from_id"),
        text=raw.get("text"),
        text_entities=_parse_text_entities(raw.get("text_entities", [])),
        # media
        media_type=raw.get("media_type"),
        file=_resolve_file(raw.get("file"), export_root),
        file_name=raw.get("file_name"),
        file_size=raw.get("file_size"),
        mime_type=raw.get("mime_type"),
        duration_seconds=raw.get("duration_seconds"),
        thumbnail=_resolve_file(raw.get("thumbnail"), export_root),
        # photo
        photo=raw.get("photo"),
        photo_file_size=raw.get("photo_file_size"),
        width=raw.get("width"),
        height=raw.get("height"),
        # sticker
        sticker_emoji=raw.get("sticker_emoji"),
        # forwarded
        forwarded_from=raw.get("forwarded_from"),
        forwarded_from_id=raw.get("forwarded_from_id"),
        # reply
        reply_to_message_id=raw.get("reply_to_message_id"),
        # reactions
        reactions=_parse_reactions(raw.get("reactions", [])),
        # edits
        edited=raw.get("edited"),
        edited_unixtime=raw.get("edited_unixtime"),
        # service
        action=raw.get("action"),
        actor=raw.get("actor"),
        actor_id=raw.get("actor_id"),
        message_id=raw.get("message_id"),
        discard_reason=raw.get("discard_reason"),
        # misc
        via_bot=raw.get("via_bot"),
        self_destruct_period_seconds=raw.get("self_destruct_period_seconds"),
        location_information=_parse_location(raw.get("location_information")),
        live_location_period_seconds=raw.get("live_location_period_seconds"),
    )


def _parse_text_entities(raw_entities: list[dict[str, Any]]) -> list[TextEntity]:
    return [
        # text may be null in real exports; coalesce at the boundary so the
        # TextEntity invariant holds: text is always a str.
        TextEntity(type=e["type"], text=e.get("text") or "")
        for e in raw_entities
        if "type" in e and "text" in e
    ]


def _parse_reactions(raw_reactions: list[dict[str, Any]]) -> list[Reaction]:
    result: list[Reaction] = []
    for r in raw_reactions:
        recent_raw = r.get("recent") or []
        recent = [
            ReactionRecent(
                from_name=rr.get("from", ""),
                from_id=rr.get("from_id", ""),
                date=rr.get("date", ""),
            )
            for rr in recent_raw
        ]
        result.append(
            Reaction(
                type=r.get("type", ""),
                count=r.get("count", 0),
                emoji=r.get("emoji", ""),
                recent=recent,
            )
        )
    return result


def _parse_location(raw: dict[str, Any] | None) -> LocationInfo | None:
    if not raw:
        return None
    return LocationInfo(latitude=raw["latitude"], longitude=raw["longitude"])


def _resolve_file(file_path: str | None, export_root: Path) -> str | None:
    """Resolve a media file path to a canonical absolute path; None for placeholders.

    The returned string doubles as the transcription cache key, so it must not
    depend on the CWD or on how export_dir was typed on the CLI. .resolve()
    canonicalizes the joined path regardless of path form or current directory.
    """
    if file_path is None:
        return None
    if file_path.startswith("(File not included") or file_path.startswith("(File unavailable"):
        return None
    resolved = (export_root / file_path).resolve()
    if resolved.exists():
        return str(resolved)
    return None  # file referenced but doesn't exist on disk
