"""Format parsed Telegram messages into a single Markdown document."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone

from .models import Message, TextEntity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def format_markdown(
    messages: Iterable[Message],
    chat_name: str,
    transcripts: dict[str, str],
    message_index: dict[int, Message] | None = None,
) -> str:
    """Convert a stream of messages into a complete Markdown string.

    Args:
        messages: Messages in chronological order.
        chat_name: Display name of the chat.
        transcripts: {absolute_file_path: transcribed_text}.
        message_index: {message_id: Message} for resolving reply quotes.
                       If None, replies will lack inline quotes.
    """
    idx = message_index or {}

    # Group by calendar day
    by_date: dict[str, list[Message]] = defaultdict(list)
    day_order: dict[str, datetime] = {}
    for msg in messages:
        date_key = _date_key(msg)
        by_date[date_key].append(msg)
        if date_key not in day_order:
            parsed_day = _parse_date_or_none(msg.date)
            if parsed_day is not None:
                day_order[date_key] = parsed_day

    # Build output
    parts: list[str] = []
    parts.append(f"# Чат: {chat_name}\n\n---\n")

    # Days must appear chronologically: Russian headers ("2 июля", "10 июля")
    # do not sort lexicographically, so order groups by their parsed date.
    for date_key in sorted(by_date.keys(), key=lambda k: _day_sort_key(day_order, k)):
        parts.append(f"## {date_key}\n")
        for msg in by_date[date_key]:
            parts.append(_format_message(msg, transcripts, idx))
            parts.append("")
        parts.append("---\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single message formatting
# ---------------------------------------------------------------------------
def _format_message(msg: Message, transcripts: dict[str, str], idx: dict[int, Message]) -> str:
    """Format a single message as a Markdown block."""
    lines: list[str] = []

    # Header line: time | sender
    time_str = _format_time(msg.date)
    sender = msg.from_name or "Неизвестный"
    header = f"**{time_str}** | **{sender}**"

    # Service messages get special treatment
    if msg.is_service:
        return _format_service(msg, header)

    # --- Reply block ---
    if msg.is_reply and msg.reply_to_message_id:
        replied = idx.get(msg.reply_to_message_id)
        if replied:
            lines.append(f"> ↩ В ответ на **{replied.from_name or 'Неизвестный'}** ({_format_time(replied.date)}):")
            reply_quote = _reply_preview(replied, transcripts)
            if reply_quote.strip():
                lines.append(f">> {reply_quote}")

    # --- Forwarded block ---
    if msg.is_forwarded:
        source_kind = "канал" if msg.forwarded_source_kind == "channel" else "пользователь"
        lines.append(f"> ↪ Переслано от **{msg.forwarded_from}** ({source_kind})")

    # --- Media: voice / round video ---
    if msg.is_voice or msg.is_round_video:
        media_label = "🎤 Голосовое сообщение" if msg.is_voice else "📹 Видеосообщение"
        duration = f" ({_format_duration(msg.duration_seconds)})" if msg.duration_seconds else ""
        lines.append(f"{header}{duration}")
        lines.append(f"**{media_label}:**")

        transcript = _find_transcript(msg, transcripts)
        if transcript:
            lines.append(f"> *Расшифровка:* {transcript}")
        else:
            lines.append("> *(расшифровка недоступна)*")

    # --- Media: sticker ---
    elif msg.is_sticker:
        emoji = f" {msg.sticker_emoji}" if msg.sticker_emoji else ""
        lines.append(f"{header} [🎭 Стикер{emoji}]")

    # --- Media: photo ---
    elif msg.is_photo:
        lines.append(f"{header} [📷 Фото]")

    # --- Media: video file ---
    elif msg.is_video_file:
        fname = f" — *{msg.file_name}*" if msg.file_name else ""
        duration = f" ({_format_duration(msg.duration_seconds)})" if msg.duration_seconds else ""
        lines.append(f"{header} [🎬 Видео{duration}{fname}]")

    # --- Media: animation / GIF ---
    elif msg.is_animation:
        lines.append(f"{header} [🎞️ GIF]")

    # --- Media: generic file ---
    elif msg.file_name and not msg.media_type:
        lines.append(f"{header} [📎 Файл: *{msg.file_name}*]")

    # --- Plain text / text with media ---
    else:
        text_md = _format_text(msg)
        if text_md:
            lines.append(f"{header}: {text_md}")
        else:
            lines.append(header)

    # --- Reactions ---
    if msg.reactions:
        reaction_parts: list[str] = []
        for r in msg.reactions:
            detail = ""
            if r.recent:
                names = ", ".join(rr.from_name for rr in r.recent[:3])
                if len(r.recent) > 3:
                    names += f" и ещё {len(r.recent) - 3}"
                detail = f" ({names})"
            reaction_parts.append(f"{r.emoji} ×{r.count}{detail}")
        lines.append(f"  *{', '.join(reaction_parts)}*")

    # --- Edited marker ---
    if msg.edited:
        lines.append(f"  *(отредактировано {_format_time(msg.edited)})*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_text(msg: Message) -> str:
    """Convert message text + entities into Markdown."""
    if not msg.text_entities:
        # No entities — plain text
        return _escape_md(msg.plain_text)

    parts: list[str] = []
    for ent in msg.text_entities:
        t = _escape_md(ent.text)
        match ent.type:
            case "bold":
                parts.append(f"**{t}**")
            case "italic":
                parts.append(f"*{t}*")
            case "link":
                parts.append(f"<{t}>")
            case "text_link":
                # text_link has the URL embedded — but in our model it's just text
                parts.append(t)
            case "mention":
                parts.append(f"`{t}`")
            case "hashtag":
                parts.append(f"`{t}`")
            case "phone":
                parts.append(f"`{t}`")
            case "custom_emoji":
                parts.append(t)
            case "blockquote":
                parts.append(f"> {t}")
            case _:
                parts.append(t)

    return "".join(parts)


def _escape_md(text: str) -> str:
    """Escape only characters that can break Markdown inline formatting."""
    # Order matters: backslash first
    for ch in ("\\", "*", "_", "`", "[", "]"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _reply_preview(msg: Message, transcripts: dict[str, str]) -> str:
    """Get a text preview of a replied-to message, handling all media types."""
    if msg.is_voice:
        t = _find_transcript(msg, transcripts)
        return f"[🎤 Голосовое{' — ' + t[:150] + '…' if t and len(t) > 150 else (' — ' + t if t else '')}]"
    if msg.is_round_video:
        t = _find_transcript(msg, transcripts)
        return f"[📹 Видеосообщение{' — ' + t[:150] + '…' if t and len(t) > 150 else (' — ' + t if t else '')}]"
    if msg.is_sticker:
        emoji = f" {msg.sticker_emoji}" if msg.sticker_emoji else ""
        return f"[🎭 Стикер{emoji}]"
    if msg.is_photo:
        return "[📷 Фото]"
    if msg.is_video_file:
        return f"[🎬 Видео]"
    if msg.is_animation:
        return "[🎞️ GIF]"
    text = msg.plain_text
    if len(text) > 200:
        text = text[:200] + "…"
    return text


def _find_transcript(msg: Message, transcripts: dict[str, str]) -> str | None:
    """Find the transcript for a voice/video message by its file path."""
    if not msg.file:
        return None
    return transcripts.get(msg.file)


def _parse_date_or_none(iso_string: str) -> datetime | None:
    """Parse an ISO datetime; return None instead of raising."""
    try:
        return datetime.fromisoformat(iso_string)
    except ValueError:
        return None


def _day_sort_key(day_order: dict[str, datetime], date_key: str) -> tuple[bool, datetime | str]:
    """Chronological sort key for day groups: real dates first, unparseable last."""
    day = day_order.get(date_key)
    if day is None:
        return (False, date_key)
    return (True, day)


def _date_key(msg: Message) -> str:
    """Extract human-readable date from ISO timestamp."""
    dt = _parse_date_or_none(msg.date)
    if dt is None:
        return msg.date[:10]
    # Russian month names
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    ]
    return f"{dt.day} {months[dt.month - 1]} {dt.year}"


def _format_time(iso_string: str) -> str:
    """Extract HH:MM from ISO timestamp."""
    if "T" in iso_string:
        return iso_string.split("T")[1][:5]
    return iso_string[:5]


def _format_duration(seconds: int | None) -> str:
    """Format seconds as M:SS."""
    if seconds is None:
        return ""
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def _format_service(msg: Message, header: str) -> str:
    """Format a service (system) message."""
    match msg.action:
        case "phone_call":
            reason = {
                "hangup": "Завершён",
                "missed": "Пропущенный",
                "busy": "Отклонён (занято)",
            }.get(msg.discard_reason or "", "Звонок")
            duration = f" ({_format_duration(msg.duration_seconds)})" if msg.duration_seconds else ""
            return f"⚡ **Системное**: {reason} звонок от **{msg.actor or '?'}**{duration}"
        case "pin_message":
            return f"⚡ **Системное**: **{msg.actor or '?'}** закрепил сообщение #{msg.message_id}"
        case _:
            return f"⚡ **Системное**: {msg.action or 'неизвестное действие'}"
