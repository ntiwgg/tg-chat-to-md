"""Unit tests for src.models.Message helper properties.

Hermetic: Message dataclasses are constructed directly; no parsing, no files.
Covers has_text/plain_text over all text shapes, forwarded_source_kind,
media-kind properties, and the placeholder/None handling of has_media_file.
"""

import pytest

from src.models import Message


def _message(**overrides) -> Message:
    defaults = {
        "id": 1,
        "type": "message",
        "date": "2026-07-24T10:00:00",
        "date_unixtime": "0",
    }
    defaults.update(overrides)
    return Message(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# has_text: str, list of str, list with entity dicts, empty/None
# ---------------------------------------------------------------------------
def test_has_text_plain_str() -> None:
    assert _message(text="просто текст").has_text is True
    assert _message(text="   ").has_text is False
    assert _message(text="").has_text is False


def test_has_text_list_of_strings() -> None:
    assert _message(text=["a", "b"]).has_text is True
    assert _message(text=["", "  "]).has_text is False


def test_has_text_list_with_entity_dicts() -> None:
    text = ["Привет, ", {"type": "bold", "text": "мир"}, "!"]
    assert _message(text=text).has_text is True
    assert _message(text=[{"type": "bold", "text": "   "}]).has_text is False


def test_has_text_entity_with_null_text_does_not_crash() -> None:
    """REGRESSION: {'type': 'bold', 'text': None} hit .strip() on None."""
    assert _message(text=[{"type": "bold", "text": None}]).has_text is False
    mixed = ["Привет", {"type": "bold", "text": None}]
    assert _message(text=mixed).has_text is True


def test_has_text_none() -> None:
    assert _message(text=None).has_text is False


def test_has_text_unsupported_type_is_false() -> None:
    assert _message(text=42).has_text is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# plain_text: formatting stripped, entities flattened
# ---------------------------------------------------------------------------
def test_plain_text_str_passthrough() -> None:
    assert _message(text="сырой текст").plain_text == "сырой текст"


def test_plain_text_list_of_strings_joins() -> None:
    assert _message(text=["один ", "два"]).plain_text == "один два"


def test_plain_text_list_with_entities_flattens_plain() -> None:
    text = [
        "Смотри ",
        {"type": "bold", "text": "жирный"},
        ", ",
        {"type": "italic", "text": "курсив"},
    ]
    msg = _message(text=text)
    assert msg.plain_text == "Смотри жирный, курсив"
    assert msg.has_text is True


def test_plain_text_entities_bold_link_mention_code() -> None:
    text = [
        {"type": "bold", "text": "bold"},
        " | ",
        {"type": "link", "text": "https://example.com"},
        " | ",
        {"type": "mention", "text": "@user"},
        " | ",
        {"type": "code", "text": "x = 1"},
    ]
    assert _message(text=text).plain_text == "bold | https://example.com | @user | x = 1"


def test_plain_text_none_is_empty() -> None:
    assert _message(text=None).plain_text == ""


def test_plain_text_entity_with_null_text_coalesces_to_empty() -> None:
    """REGRESSION: {'type': 'bold', 'text': None} broke the "".join."""
    msg = _message(text=["A", {"type": "bold", "text": None}, "B"])
    assert msg.plain_text == "AB"
    assert _message(text=[{"type": "bold", "text": None}]).plain_text == ""


# ---------------------------------------------------------------------------
# forwarded_source_kind
# ---------------------------------------------------------------------------
def test_forwarded_source_kind_channel_and_user_and_unknown() -> None:
    assert _message(forwarded_from_id="channel123").forwarded_source_kind == "channel"
    assert _message(forwarded_from_id="user42").forwarded_source_kind == "user"
    assert _message(forwarded_from_id="").forwarded_source_kind == "unknown"
    assert _message(forwarded_from_id=None).forwarded_source_kind == "unknown"
    assert _message().forwarded_source_kind == "unknown"


# ---------------------------------------------------------------------------
# media-kind properties: discriminating cross-product with negatives
# ---------------------------------------------------------------------------
ALL_MEDIA_FLAGS = [
    "is_voice",
    "is_round_video",
    "is_sticker",
    "is_video_file",
    "is_animation",
    "is_photo",
]


def _active_flags(msg: Message, flags: list[str]) -> list[str]:
    return [name for name in flags if getattr(msg, name)]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # Exactly ONE media flag may be true per message; a copy-paste bug
        # (e.g. is_voice True for video_message) must fail the set equality.
        ({"media_type": "voice_message"}, {"is_voice"}),
        ({"media_type": "video_message"}, {"is_round_video"}),
        ({"media_type": "sticker"}, {"is_sticker"}),
        ({"media_type": "video_file"}, {"is_video_file"}),
        ({"media_type": "animation"}, {"is_animation"}),
        ({"photo": "(File not included by export)"}, {"is_photo"}),
        # No media at all: every flag False (the negatives).
        ({}, set()),
    ],
)
def test_media_kind_flags_are_mutually_exclusive(fields, expected) -> None:
    msg = _message(**fields)

    assert set(_active_flags(msg, ALL_MEDIA_FLAGS)) == expected


def test_is_photo_and_has_media_file() -> None:
    assert _message(photo="(File not included)").is_photo is True
    assert _message().is_photo is False


def test_has_media_file_placeholder_and_none() -> None:
    assert _message(file="voice_messages/a.ogg").has_media_file is True
    assert _message(file="(File not included by export)").has_media_file is False
    assert _message(file="(File unavailable, please try again later)").has_media_file is False
    assert _message(file=None).has_media_file is False


# ---------------------------------------------------------------------------
# is_service / is_reply / is_forwarded: exact flag sets with negatives
# ---------------------------------------------------------------------------
STATE_FLAGS = ["is_service", "is_reply", "is_forwarded"]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"type": "service", "action": "phone_call"}, {"is_service"}),
        # A service message must not read as reply/forwarded...
        ({"type": "service", "action": "phone_call", "reply_to_message_id": 3},
         {"is_service", "is_reply"}),  # ...but flags stay independent, no cross-suppression
        ({"reply_to_message_id": 3}, {"is_reply"}),
        ({"forwarded_from": "Кто-то"}, {"is_forwarded"}),
        ({"reply_to_message_id": 3, "forwarded_from": "Кто-то"}, {"is_reply", "is_forwarded"}),
        ({"type": "message"}, set()),
        ({}, set()),
    ],
)
def test_state_flags_exact_set_with_negatives(fields, expected) -> None:
    msg = _message(**fields)

    assert set(_active_flags(msg, STATE_FLAGS)) == expected
