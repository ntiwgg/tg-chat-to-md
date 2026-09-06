"""Adversarial fixtures: malformed/edge-case result.json must parse and render.

SQLite's "malformed input must not crash" precedent applied to Telegram
exports: real-world result.json files carry shapes the happy path never
produces — forwarded messages without ids, replies into the void, unknown
service actions, empty text lists, placeholder media strings, ISO garbage in
the date field, null texts, and entity segments whose text is null. Each
fixture in tests/fixtures/ is parsed with EXACT field expectations and then
rendered through format_markdown, which must not raise and must still produce
a document with day-group headers.

Every fixture is synthetic (1-3 tiny messages, no personal data, no media
files on disk), so the whole file stays hermetic.
"""

import json
from pathlib import Path

import pytest

from src.formatter import format_markdown
from src.parser import parse_export

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_NAMES = sorted(p.stem for p in FIXTURES_DIR.glob("adv_*.json"))


def _parse_fixture(tmp_path: Path, name: str):
    """Copy a fixture into a tmp export dir and parse it (parse_export needs a
    real result.json on disk; the fixtures directory itself holds many files)."""
    export_dir = tmp_path / f"ChatExport_{name}"
    export_dir.mkdir()
    raw = json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    (export_dir / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    chat_name, chat_id, messages = parse_export(export_dir)
    return chat_name, chat_id, messages


def _render(chat_name: str, messages) -> str:
    """Render exactly like the CLI phase-4 call."""
    msg_index = {m.id: m for m in messages}
    return format_markdown(messages, chat_name, {}, msg_index)


def _headers(md: str) -> list[str]:
    return [line for line in md.splitlines() if line.startswith("## ")]


# ---------------------------------------------------------------------------
# umbrella: every fixture parses sorted by id and renders without crashing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_adversarial_fixture_parses_and_renders(name, tmp_path) -> None:
    chat_name, chat_id, messages = _parse_fixture(tmp_path, name)

    assert chat_id >= 9000  # fixture ids are namespaced, never a real chat
    assert [m.id for m in messages] == sorted(m.id for m in messages)

    md = _render(chat_name, messages)  # must not raise

    assert md.startswith(f"# Чат: {chat_name}")
    assert len(_headers(md)) >= 1  # every message lands in some day group


# ---------------------------------------------------------------------------
# forwarded messages: channel vs user vs missing source ids
# ---------------------------------------------------------------------------
def test_forwarded_channel_keeps_name_and_channel_kind(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_forwarded_channel")

    assert len(messages) == 2
    with_channel_id, without_id = messages
    assert with_channel_id.forwarded_from == "Парк культуры и отдыха"
    assert with_channel_id.forwarded_from_id == "channel42"
    assert with_channel_id.forwarded_source_kind == "channel"
    assert without_id.forwarded_from == "Лесной журнал"
    assert without_id.forwarded_from_id is None
    assert without_id.forwarded_source_kind == "unknown"

    md = _render("Тест", messages)
    assert "> ↪ Переслано от **Парк культуры и отдыха** (канал)" in md
    assert "> ↪ Переслано от **Лесной журнал** (пользователь)" in md


def test_forwarded_user_keeps_name_and_user_kind(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_forwarded_user")

    assert len(messages) == 2
    with_user_id, saved_messages = messages
    assert with_user_id.forwarded_from == "Иван Петров"
    assert with_user_id.forwarded_source_kind == "user"
    assert saved_messages.forwarded_from == "Избранное"
    assert saved_messages.forwarded_source_kind == "unknown"  # no forwarded_from_id

    md = _render("Тест", messages)
    assert "> ↪ Переслано от **Иван Петров** (пользователь)" in md


# ---------------------------------------------------------------------------
# reply pointing at a message id that does not exist (and is even larger)
# ---------------------------------------------------------------------------
def test_reply_to_later_missing_message_is_safe(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_reply_to_later")

    assert [m.id for m in messages] == [1, 3]  # gaps in ids survive
    reply = messages[1]
    assert reply.reply_to_message_id == 999
    assert reply.is_reply

    # The index has no id 999: no crash, no phantom quote, text still rendered.
    md = _render("Тест", messages)
    assert "Ссылаюсь на сообщение, которого ещё нет" in md
    assert "В ответ на" not in md


# ---------------------------------------------------------------------------
# unknown service actions render the generic label, never crash
# ---------------------------------------------------------------------------
def test_unknown_service_action_renders_generic(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_service_unknown_action")

    assert len(messages) == 2
    assert all(m.is_service for m in messages)
    assert messages[0].action == "join_group_by_link"
    assert messages[1].action == "history_cleared"

    md = _render("Тест", messages)
    assert "⚡ **Системное**: join_group_by_link" in md
    assert "⚡ **Системное**: history_cleared" in md


# ---------------------------------------------------------------------------
# empty text list: [] stays [] (not None), message renders header-only
# ---------------------------------------------------------------------------
def test_empty_text_list_message(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_empty_text_list")

    assert len(messages) == 1
    assert messages[0].text == []
    assert messages[0].has_text is False
    assert messages[0].plain_text == ""

    md = _render("Тест", messages)
    body = [line for line in md.splitlines() if line.startswith("**")]
    assert body == ["**12:00** | **Оля**"]  # header only, no colon text


# ---------------------------------------------------------------------------
# media files that are not on disk: every placeholder form resolves to None
# ---------------------------------------------------------------------------
def test_absent_media_files_resolve_to_none(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_file_absent")

    assert len(messages) == 3
    voice, round_video, ghost_clip = messages
    assert voice.is_voice and voice.file is None  # "(File not included...)"
    assert round_video.is_round_video and round_video.file is None  # "(File unavailable...)"
    assert ghost_clip.is_video_file and ghost_clip.file is None  # missing on disk
    assert not any(m.has_media_file for m in messages)

    md = _render("Тест", messages)
    assert "🎤 Голосовое сообщение" in md
    assert "📹 Видеосообщение" in md
    assert "🎬 Видео" in md
    assert "*(расшифровка недоступна)*" in md


# ---------------------------------------------------------------------------
# garbage dates: raw values preserved, [:10] fallback groups after parsed ones
# ---------------------------------------------------------------------------
def test_garbage_dates_preserved_and_grouped_last(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_garbage_date")

    # Parser never validates dates: raw garbage survives into the model.
    assert [m.date for m in messages] == ["2026-07-02T10:00:00", "2026-07-02T25:00:00", ""]

    md = _render("Тест", messages)
    # Parsed day first, then the two garbage fallback groups ("2026-07-02" is
    # the [:10] prefix of the invalid-hour date; "" renders as an empty header).
    assert _headers(md) == ["## 2 июля 2026", "## ", "## 2026-07-02"]


# ---------------------------------------------------------------------------
# text: None — header-only render, no crash
# ---------------------------------------------------------------------------
def test_null_text_message(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_text_none")

    assert len(messages) == 1
    assert messages[0].text is None
    assert messages[0].has_text is False

    md = _render("Тест", messages)
    assert "**18:00** | **Коля**" in md
    assert "**18:00** | **Коля**:" not in md  # no colon without text


# ---------------------------------------------------------------------------
# entity segment with null text: coalesced at the boundary, render is safe
# ---------------------------------------------------------------------------
def test_entity_null_text_coalesces_and_renders(tmp_path) -> None:
    _chat_name, _chat_id, messages = _parse_fixture(tmp_path, "adv_entity_null_text")

    assert len(messages) == 1
    msg = messages[0]
    assert msg.text == [{"type": "bold", "text": None}, " просто текст после пустой сущности"]
    # Boundary invariant: TextEntity.text is always a str, never None.
    assert msg.text_entities[0].text == ""
    assert msg.has_text is True
    assert msg.plain_text == " просто текст после пустой сущности"

    md = _render("Тест", messages)
    assert "просто текст после пустой сущности" in md
