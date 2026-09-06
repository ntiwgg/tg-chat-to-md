"""Hermetic parser tests against the synthetic sample export + tmp fixtures.

The sample export (examples/sample_export/) contains real Telegram export
shapes: text as str, text as list of segments with entity dicts, replies,
forwarded messages, service messages (phone_call/pin_message), stickers,
voice/video placeholders, reactions, edits — but no personal data and no
media files on disk (every file reference is a "(File not included...)" or
"(File unavailable...)" placeholder).
"""

import json
import random
from pathlib import Path

import pytest

from tg_chat_to_md.models import Reaction, TextEntity
from tg_chat_to_md.parser import _parse_message, parse_export

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample_export"


def _parse_sample() -> tuple[str, int, list]:
    return parse_export(SAMPLE_DIR)


# ---------------------------------------------------------------------------
# sample export: count, order, shapes
# ---------------------------------------------------------------------------
def test_sample_export_count_name_and_id() -> None:
    chat_name, chat_id, messages = _parse_sample()

    assert chat_name == "Пикник у озера"
    assert chat_id == 123456789
    assert len(messages) == 30


def test_parse_sorts_shuffled_messages_by_id(tmp_path) -> None:
    """Sorting must hold for ANY array order in result.json — not just the
    committed fixture, which happens to be stored already ordered (a
    self-comparing assert there is vacuous: deleting the parser's sort would
    not fail). Rebuild the same messages in a shuffled array order and pin
    the exact expected chronological sequence."""
    raw = json.loads((SAMPLE_DIR / "result.json").read_text(encoding="utf-8"))
    messages_raw = list(raw["messages"])
    rng = random.Random(0)
    rng.shuffle(messages_raw)
    assert messages_raw != raw["messages"]  # guard: the shuffle actually permutes

    export_dir = tmp_path / "ChatExport_shuffled"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps({"name": raw["name"], "id": raw["id"], "messages": messages_raw}),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    assert [m.id for m in messages] == list(range(1, 31))


def test_sample_export_contains_both_text_shapes() -> None:
    _chat_name, _chat_id, messages = _parse_sample()

    text_as_str = [m for m in messages if isinstance(m.text, str)]
    text_as_list = [m for m in messages if isinstance(m.text, list)]

    assert len(text_as_str) == 24
    assert len(text_as_list) == 6
    # a list-of-segments message contains entity dicts, not only plain strings
    seg_types = {
        "type" if isinstance(seg, dict) else "str"
        for m in text_as_list
        for seg in m.text  # type: ignore[union-attr]
    }
    assert seg_types == {"type", "str"}


def test_sample_export_text_entities_parsed() -> None:
    _chat_name, _chat_id, messages = _parse_sample()
    msg = next(m for m in messages if m.id == 2)  # bold segment

    assert isinstance(msg.text_entities, list)
    assert all(isinstance(e, TextEntity) for e in msg.text_entities)
    bold = [e for e in msg.text_entities if e.type == "bold"]
    assert bold and bold[0].text == "без опозданий"
    assert msg.plain_text == "Я за. Только давай без опозданий — в прошлый раз ждали полчаса."


def test_sample_export_placeholder_media_files_resolve_to_none() -> None:
    _chat_name, _chat_id, messages = _parse_sample()

    voice = next(m for m in messages if m.id == 9)  # voice_message placeholder
    video = next(m for m in messages if m.id == 12)  # video_message placeholder
    sticker = next(m for m in messages if m.id == 7)
    photo = next(m for m in messages if m.id == 8)

    assert voice.is_voice
    assert voice.file is None  # "(File not included...)" -> None
    assert video.is_round_video
    assert video.file is None  # "(File unavailable...)" -> None
    assert sticker.is_sticker
    assert sticker.file is None
    assert photo.is_photo
    assert photo.file is None
    assert not any(m.file for m in messages)  # demo has zero real files


def test_sample_export_reply_forward_service_and_edit_fields() -> None:
    _chat_name, _chat_id, messages = _parse_sample()
    by_id = {m.id: m for m in messages}

    reply = by_id[3]
    assert reply.is_reply
    assert reply.reply_to_message_id == 1

    forwarded = by_id[15]
    assert forwarded.is_forwarded
    assert forwarded.forwarded_from == "Лес и парк"
    assert forwarded.forwarded_source_kind == "channel"

    call = by_id[6]
    assert call.is_service
    assert call.action == "phone_call"
    assert call.discard_reason == "hangup"
    assert call.duration_seconds == 43

    pin = by_id[11]
    assert pin.is_service
    assert pin.action == "pin_message"
    assert pin.message_id == 2

    edited = by_id[24]
    assert edited.edited == "2026-09-06T09:11:02"
    assert edited.reactions
    assert all(isinstance(r, Reaction) for r in edited.reactions)
    assert edited.reactions[0].emoji == "🌚"
    assert edited.reactions[0].recent[0].from_name == "Борис"


# ---------------------------------------------------------------------------
# synthetic tmp exports: error and placeholder paths
# ---------------------------------------------------------------------------
def test_parse_export_missing_result_json_raises(tmp_path) -> None:
    empty_dir = tmp_path / "ChatExport_empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="result.json not found in"):
        parse_export(empty_dir)


def test_parse_export_placeholder_file_value_is_none(tmp_path) -> None:
    export_dir = tmp_path / "ChatExport_placeholder"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "text": "",
                        "media_type": "voice_message",
                        "file": "(File not included by export)",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    assert len(messages) == 1
    assert messages[0].file is None
    assert not messages[0].has_media_file


def test_parse_export_missing_file_on_disk_is_none(tmp_path) -> None:
    """A file reference that does not exist on disk is dropped (None)."""
    export_dir = tmp_path / "ChatExport_ghost"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "text": "",
                        "media_type": "voice_message",
                        "file": "voice_messages/ghost.ogg",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    assert messages[0].file is None


def test_parse_export_null_text_entity_coalesces_to_empty_string(tmp_path) -> None:
    """Entity segments with a null 'text' must parse to '' (never None), so
    the formatter's _escape_md never sees a None text."""
    export_dir = tmp_path / "ChatExport_null_entity"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "text": [{"type": "bold", "text": None}],
                        "text_entities": [{"type": "bold", "text": None}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    entity = messages[0].text_entities[0]
    assert entity.type == "bold"
    assert entity.text == ""
    assert messages[0].has_text is False
    assert messages[0].plain_text == ""


# ---------------------------------------------------------------------------
# mutation oracles: full-field round trip and absent-key defaults
# ---------------------------------------------------------------------------
def _full_raw_message(export_root: Path, voice_path: Path, thumb_path: Path) -> dict:
    """A message raw dict carrying EVERY Message field (no defaults active).

    file/thumbnail must point at real files, otherwise resolution returns
    None and the round trip could not tell a dropped field from a resolved
    None. Values are chosen to differ from every field default, so a parser
    mutation that skips, renames, or defaults any lookup changes the result.
    """
    return {
        "id": 42,
        "type": "message",
        "date": "2026-07-24T10:00:00",
        "date_unixtime": "1784916000",
        "from": "Аня",
        "from_id": "user42",
        "text": [{"type": "bold", "text": "привет"}, " мир"],
        "text_entities": [
            {"type": "bold", "text": "привет"},
            {"type": "text_link", "text": "пример"},
        ],
        "media_type": "voice_message",
        "file": str(voice_path.relative_to(export_root)),
        "file_name": "audio.ogg",
        "file_size": 1234,
        "mime_type": "audio/ogg",
        "duration_seconds": 61,
        "thumbnail": str(thumb_path.relative_to(export_root)),
        "photo": "photo.jpg",
        "photo_file_size": 99,
        "width": 640,
        "height": 480,
        "sticker_emoji": "🎈",
        "forwarded_from": "Лес и парк",
        "forwarded_from_id": "channel9",
        "reply_to_message_id": 7,
        "reactions": [
            {
                "type": "emoji",
                "count": 3,
                "emoji": "🔥",
                "recent": [
                    {"from": "Ира", "from_id": "user1", "date": "2026-07-24T10:00:00"},
                    {"from": "Петя", "from_id": "user2", "date": "2026-07-24T10:05:00"},
                ],
            }
        ],
        "edited": "2026-07-24T11:00:00",
        "edited_unixtime": "1784919600",
        "action": "phone_call",
        "actor": "Аня",
        "actor_id": "user42",
        "message_id": 5,
        "discard_reason": "busy",
        "via_bot": "BotFather",
        "self_destruct_period_seconds": 10,
        "location_information": {"latitude": 55.75, "longitude": 37.61},
        "live_location_period_seconds": 300,
    }


def test_parse_message_round_trips_every_field(tmp_path) -> None:
    """One message carrying every field, parsed and asserted attribute-wise.

    This is the exact oracle for the parser's mapping layer: a mutation that
    drops a call, replaces it with None, or corrupts a lookup key must change
    at least one asserted attribute. Absent-key defaults are covered by
    test_parse_export_minimal_document_defaults below.
    """
    voice = tmp_path / "voice.ogg"
    thumb = tmp_path / "thumb.jpg"
    voice.write_bytes(b"x")
    thumb.write_bytes(b"x")
    raw = _full_raw_message(tmp_path, voice, thumb)

    msg = _parse_message(raw, tmp_path)

    assert msg.id == 42
    assert msg.type == "message"
    assert msg.date == "2026-07-24T10:00:00"
    assert msg.date_unixtime == "1784916000"
    assert msg.from_name == "Аня"
    assert msg.from_id == "user42"
    assert msg.text == [{"type": "bold", "text": "привет"}, " мир"]
    assert [(e.type, e.text) for e in msg.text_entities] == [
        ("bold", "привет"),
        ("text_link", "пример"),
    ]
    assert msg.media_type == "voice_message"
    assert msg.file == str(voice.resolve())
    assert msg.file_name == "audio.ogg"
    assert msg.file_size == 1234
    assert msg.mime_type == "audio/ogg"
    assert msg.duration_seconds == 61
    assert msg.thumbnail == str(thumb.resolve())
    assert msg.photo == "photo.jpg"
    assert msg.photo_file_size == 99
    assert msg.width == 640
    assert msg.height == 480
    assert msg.sticker_emoji == "🎈"
    assert msg.forwarded_from == "Лес и парк"
    assert msg.forwarded_from_id == "channel9"
    assert msg.reply_to_message_id == 7
    assert msg.edited == "2026-07-24T11:00:00"
    assert msg.edited_unixtime == "1784919600"
    assert msg.action == "phone_call"
    assert msg.actor == "Аня"
    assert msg.actor_id == "user42"
    assert msg.message_id == 5
    assert msg.discard_reason == "busy"
    assert msg.via_bot == "BotFather"
    assert msg.self_destruct_period_seconds == 10
    assert msg.live_location_period_seconds == 300

    assert len(msg.reactions) == 1
    reaction = msg.reactions[0]
    assert (reaction.type, reaction.count, reaction.emoji) == ("emoji", 3, "🔥")
    assert [(r.from_name, r.from_id, r.date) for r in reaction.recent] == [
        ("Ира", "user1", "2026-07-24T10:00:00"),
        ("Петя", "user2", "2026-07-24T10:05:00"),
    ]
    assert msg.location_information is not None
    assert (msg.location_information.latitude, msg.location_information.longitude) == (
        55.75,
        37.61,
    )


def test_parse_export_minimal_document_defaults(tmp_path) -> None:
    """An export and a message with every optional key absent: the parser must
    apply its documented defaults ("" / 0 / [] / None) instead of crashing."""
    export_dir = tmp_path / "ChatExport_minimal"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"id": 1, "type": "message", "reactions": [{"recent": [{}]}]}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chat_name, chat_id, messages = parse_export(export_dir)

    assert chat_name == "Unknown Chat"
    assert chat_id == 0
    assert len(messages) == 1
    msg = messages[0]
    assert msg.date == ""
    assert msg.date_unixtime == ""
    assert msg.from_name is None
    assert msg.text is None
    assert msg.media_type is None
    assert msg.file is None
    assert msg.edited is None
    assert msg.forwarded_from is None
    assert msg.reply_to_message_id is None
    # the sole reaction is key-less, and its "recent" entry is too
    reaction = msg.reactions[0]
    assert reaction.type == ""
    assert reaction.count == 0
    assert reaction.emoji == ""
    assert [(r.from_name, r.from_id, r.date) for r in reaction.recent] == [
        ("", "", "")
    ]


@pytest.mark.parametrize(
    "placeholder",
    ["(File not included by export)", "(File unavailable)"],
)
def test_resolve_file_placeholder_prefix_wins_over_existing_file(tmp_path, placeholder) -> None:
    """Placeholder detection must not depend on disk contents: even if a file
    with a placeholder name exists, the reference resolves to None."""
    export_dir = tmp_path / "ChatExport_placeholder_on_disk"
    export_dir.mkdir()
    media = export_dir / placeholder
    media.write_bytes(b"x")
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "media_type": "voice_message",
                        "file": placeholder,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    assert messages[0].file is None


def test_text_entity_missing_text_key_is_filtered_out(tmp_path) -> None:
    """An entity dict that lacks 'text' entirely is not an entity: it must be
    filtered out (and OR here would include it with an empty string)."""
    export_dir = tmp_path / "ChatExport_entity_no_text"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Тест",
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2026-07-24T10:00:00",
                        "date_unixtime": "0",
                        "text_entities": [{"type": "bold"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _chat_name, _chat_id, messages = parse_export(export_dir)

    assert messages[0].text_entities == []


def test_parse_export_without_messages_key_returns_empty(tmp_path) -> None:
    """A result.json with no 'messages' key at all parses to an empty chat
    (the parser must not crash on the missing key)."""
    export_dir = tmp_path / "ChatExport_no_messages"
    export_dir.mkdir()
    (export_dir / "result.json").write_text(
        json.dumps({"name": "Тест", "id": 1}),
        encoding="utf-8",
    )

    chat_name, chat_id, messages = parse_export(export_dir)

    assert chat_name == "Тест"
    assert chat_id == 1
    assert messages == []
