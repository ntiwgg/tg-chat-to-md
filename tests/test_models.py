"""Unit tests for src.models.Message helper properties.

Hermetic: Message dataclasses are constructed directly; no parsing, no files.
Covers has_text/plain_text over all text shapes, forwarded_source_kind,
media-kind properties, and the placeholder/None handling of has_media_file.
"""

from src.models import LocationInfo, Message, Reaction, ReactionRecent, TextEntity


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
# media-kind properties
# ---------------------------------------------------------------------------
def test_media_type_flags() -> None:
    assert _message(media_type="voice_message").is_voice is True
    assert _message(media_type="video_message").is_round_video is True
    assert _message(media_type="sticker").is_sticker is True
    assert _message(media_type="video_file").is_video_file is True
    assert _message(media_type="animation").is_animation is True
    assert _message(media_type=None).is_voice is False


def test_is_photo_and_has_media_file() -> None:
    assert _message(photo="(File not included)").is_photo is True
    assert _message().is_photo is False


def test_has_media_file_placeholder_and_none() -> None:
    assert _message(file="voice_messages/a.ogg").has_media_file is True
    assert _message(file="(File not included by export)").has_media_file is False
    assert _message(file="(File unavailable, please try again later)").has_media_file is False
    assert _message(file=None).has_media_file is False


def test_is_service_is_reply_is_forwarded() -> None:
    service = _message(type="service", action="phone_call")
    assert service.is_service is True
    assert service.is_reply is False
    assert _message(type="message").is_service is False

    assert _message(reply_to_message_id=3).is_reply is True
    assert _message(reply_to_message_id=None).is_reply is False

    assert _message(forwarded_from="Кто-то").is_forwarded is True
    assert _message(forwarded_from=None).is_forwarded is False


# ---------------------------------------------------------------------------
# misc fields
# ---------------------------------------------------------------------------
def test_location_information_field() -> None:
    loc = LocationInfo(latitude=55.75, longitude=37.61)
    msg = _message(location_information=loc)

    assert msg.location_information is not None
    assert msg.location_information.latitude == 55.75
    assert msg.location_information.longitude == 37.61
    assert _message().location_information is None


def test_text_entity_dataclass_shapes() -> None:
    entity = TextEntity(type="bold", text="жирный")
    assert (entity.type, entity.text) == ("bold", "жирный")

    recent = ReactionRecent(from_name="Аня", from_id="user101", date="2026-09-01T00:00:00")
    assert recent.from_name == "Аня"

    reaction = Reaction(type="emoji", count=3, emoji="👍", recent=[recent])
    assert reaction.count == 3
    assert reaction.recent[0].date == "2026-09-01T00:00:00"
