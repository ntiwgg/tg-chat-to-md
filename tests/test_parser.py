"""Hermetic parser tests against the synthetic sample export + tmp fixtures.

The sample export (examples/sample_export/) contains real Telegram export
shapes: text as str, text as list of segments with entity dicts, replies,
forwarded messages, service messages (phone_call/pin_message), stickers,
voice/video placeholders, reactions, edits — but no personal data and no
media files on disk (every file reference is a "(File not included...)" or
"(File unavailable...)" placeholder).
"""

import json
from pathlib import Path

import pytest

from src.models import Reaction, TextEntity
from src.parser import parse_export

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
    # chronological by id, even though raw JSON may not be sorted
    assert [m.id for m in messages] == sorted(m.id for m in messages)


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

    with pytest.raises(FileNotFoundError):
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
