"""Data models for Telegram Chat Export messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextEntity:
    """A formatted text segment within a message.

    type is one of: plain, bold, italic, link, text_link, mention,
    hashtag, phone, custom_emoji, blockquote.
    """

    type: str
    text: str


@dataclass(slots=True)
class ReactionRecent:
    """Who recently reacted."""

    from_name: str
    from_id: str
    date: str


@dataclass(slots=True)
class Reaction:
    """A reaction on a message."""

    type: str
    count: int
    emoji: str
    recent: list[ReactionRecent] = field(default_factory=list)


@dataclass(slots=True)
class LocationInfo:
    """Shared location."""

    latitude: float
    longitude: float


@dataclass(slots=True)
class Message:
    """Single Telegram message — text, media, forwarded, reply, or service."""

    id: int
    type: str  # "message" | "service"
    date: str  # ISO 8601, e.g. "2026-06-26T19:01:21"
    date_unixtime: str  # string, not int (!)

    # Sender
    from_name: str | None = None
    from_id: str | None = None

    # Content — can be str or list[str | dict] (when format entities are present)
    text: str | list[Any] | None = None
    text_entities: list[TextEntity] = field(default_factory=list)

    # -------------------- media --------------------
    # media kind: voice_message | video_message | sticker | video_file | animation
    media_type: str | None = None
    file: str | None = None  # relative path or "(File not included...)"
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    duration_seconds: int | None = None
    thumbnail: str | None = None

    # Photo (separate from file — photos are NOT downloaded in this export)
    photo: str | None = None  # "(File not included...)" or "(File unavailable...)"
    photo_file_size: int | None = None
    width: int | None = None
    height: int | None = None

    # Sticker
    sticker_emoji: str | None = None

    # -------------------- forwarded --------------------
    forwarded_from: str | None = None
    forwarded_from_id: str | None = None  # "userNNN" | "channelNNN"

    # -------------------- reply --------------------
    reply_to_message_id: int | None = None

    # -------------------- reactions --------------------
    reactions: list[Reaction] = field(default_factory=list)

    # -------------------- edits --------------------
    edited: str | None = None
    edited_unixtime: str | None = None

    # -------------------- service --------------------
    action: str | None = None  # "phone_call" | "pin_message"
    actor: str | None = None
    actor_id: str | None = None
    message_id: int | None = None  # for pin: the pinned message id
    discard_reason: str | None = None  # hangup | missed | busy

    # -------------------- misc --------------------
    via_bot: str | None = None
    self_destruct_period_seconds: int | None = None
    location_information: LocationInfo | None = None
    live_location_period_seconds: int | None = None

    # -------------------- helpers --------------------

    @property
    def is_voice(self) -> bool:
        return self.media_type == "voice_message"

    @property
    def is_round_video(self) -> bool:
        return self.media_type == "video_message"

    @property
    def is_sticker(self) -> bool:
        return self.media_type == "sticker"

    @property
    def is_video_file(self) -> bool:
        return self.media_type == "video_file"

    @property
    def is_animation(self) -> bool:
        return self.media_type == "animation"

    @property
    def is_photo(self) -> bool:
        return self.photo is not None

    @property
    def is_forwarded(self) -> bool:
        return self.forwarded_from is not None

    @property
    def is_reply(self) -> bool:
        return self.reply_to_message_id is not None

    @property
    def is_service(self) -> bool:
        return self.type == "service"

    @property
    def has_media_file(self) -> bool:
        """Whether the media file actually exists on disk (not a placeholder)."""
        return (
            self.file is not None
            and not self.file.startswith("(File not included")
            and not self.file.startswith("(File unavailable")
        )

    @property
    def has_text(self) -> bool:
        """Whether the message has any text content."""
        if self.text is None:
            return False
        if isinstance(self.text, str):
            return len(self.text.strip()) > 0
        if isinstance(self.text, list):
            return any(
                (isinstance(t, str) and t.strip())
                or (isinstance(t, dict) and t.get("text", "").strip())
                for t in self.text
            )
        return False

    @property
    def plain_text(self) -> str:
        """Extract plain text from the message, regardless of text format."""
        if self.text is None:
            return ""
        if isinstance(self.text, str):
            return self.text
        if isinstance(self.text, list):
            parts: list[str] = []
            for item in self.text:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    @property
    def forwarded_source_kind(self) -> str:
        """'user' or 'channel' based on forwarded_from_id prefix."""
        if not self.forwarded_from_id:
            return "unknown"
        if self.forwarded_from_id.startswith("channel"):
            return "channel"
        return "user"
