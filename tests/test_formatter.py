"""Unit tests for src/formatter.py message rendering and day-group ordering.

Hermetic: constructs Message objects directly, no network/audio/real chat data.
Covers day-group ordering plus full message rendering branches: plain text,
entities, escaping, replies, forwards, service messages, media kinds, voice
transcripts, reactions, and edited markers.
"""

from datetime import datetime

from src.formatter import _date_key, _day_sort_key, format_markdown
from src.models import Message, Reaction, ReactionRecent, TextEntity


def _message(msg_id: int, date: str) -> Message:
    """Minimal plain-text message with the given ISO date field."""
    return Message(id=msg_id, type="message", date=date, date_unixtime="0")


def _section_headers(md: str) -> list[str]:
    """Extract the '## ' day-group headers in document order."""
    return [line for line in md.splitlines() if line.startswith("## ")]


def _full(mid: int, **overrides) -> Message:
    """Message with sender+date defaults for rendering tests."""
    defaults = {
        "id": mid,
        "type": "message",
        "date": "2026-07-24T10:00:00",
        "date_unixtime": "0",
        "from_name": "Аня",
        "text": "текст",
    }
    defaults.update(overrides)
    return Message(**defaults)  # type: ignore[arg-type]


def _render(msg: Message, transcripts: dict[str, str] | None = None, idx=None) -> str:
    """Render one message through format_markdown and strip the header."""
    md = format_markdown([msg], "Тест", transcripts or {}, idx or {})
    return md


def _entities(*pairs: tuple[str, str]) -> list[TextEntity]:
    return [TextEntity(type=t, text=s) for t, s in pairs]


# ---------------------------------------------------------------------------
# _day_sort_key
# ---------------------------------------------------------------------------
def test_day_sort_key_orders_parsed_days_ascending() -> None:
    day_order = {
        "2 марта 2026": datetime(2026, 3, 2),
        "1 января 2026": datetime(2026, 1, 1),
        "2 января 2026": datetime(2026, 1, 2),
    }
    ordered = sorted(day_order, key=lambda k: _day_sort_key(day_order, k))
    assert ordered == ["1 января 2026", "2 января 2026", "2 марта 2026"]


def test_day_sort_key_sends_unparseable_days_after_parsed() -> None:
    day_order = {"2 февраля 2026": datetime(2026, 2, 2)}
    keys = ["не дата", "2 февраля 2026", ""]
    ordered = sorted(keys, key=lambda k: _day_sort_key(day_order, k))
    assert ordered == ["2 февраля 2026", "", "не дата"]


# ---------------------------------------------------------------------------
# format_markdown: end-to-end group ordering
# ---------------------------------------------------------------------------
def test_format_markdown_parsed_days_ascending_out_of_order_input() -> None:
    messages = [
        _message(1, "2026-07-02T10:00:00"),  # 2 июля 2026
        _message(2, "2026-06-26T19:01:21"),  # 26 июня 2026
        _message(3, "2026-01-05T08:00:00"),  # 5 января 2026
    ]
    md = format_markdown(messages, chat_name="Тест", transcripts={})
    assert _section_headers(md) == [
        "## 5 января 2026",
        "## 26 июня 2026",
        "## 2 июля 2026",
    ]


def test_format_markdown_unparseable_day_group_goes_last() -> None:
    messages = [
        _message(1, "2026-06-26T19:01:21"),
        _message(2, ""),
    ]
    md = format_markdown(messages, chat_name="Тест", transcripts={})
    assert _section_headers(md) == ["## 26 июня 2026", "## "]


def test_format_markdown_mixed_parsed_and_unparseable_days() -> None:
    messages = [
        _message(1, "2026-07-02T10:00:00"),  # parsed day
        _message(2, ""),  # unparseable
        _message(3, "2026-06-26T19:01:21"),  # parsed day
        _message(4, "не дата"),  # unparseable
        _message(5, "2026-01-05T08:00:00"),  # parsed day
    ]
    md = format_markdown(messages, chat_name="Тест", transcripts={})
    assert _section_headers(md) == [
        "## 5 января 2026",
        "## 26 июня 2026",
        "## 2 июля 2026",
        "## ",
        "## не дата",
    ]


def test_iso_like_garbage_does_not_merge_with_parsed_day_group() -> None:
    """Collision guard for _date_key: an unparseable date whose 10-char prefix
    looks like a real ISO date ("2026-07-02") must not merge with the parsed
    day whose header is the Russian "2 июля 2026"."""
    messages = [
        _message(1, "2026-07-02T10:00:00"),  # parsed -> header "2 июля 2026"
        _message(2, "2026-07-02T25:00:00"),  # invalid hour -> unparseable, key "2026-07-02"
    ]
    md = format_markdown(messages, chat_name="Тест", transcripts={})
    assert _section_headers(md) == ["## 2 июля 2026", "## 2026-07-02"]


# ---------------------------------------------------------------------------
# _date_key
# ---------------------------------------------------------------------------
def test_date_key_uses_russian_header_for_parsed_dates() -> None:
    assert _date_key(_message(1, "2026-07-02T10:00:00")) == "2 июля 2026"


def test_date_key_falls_back_to_raw_prefix_for_garbage() -> None:
    assert _date_key(_message(1, "не дата")) == "не дата"
    assert _date_key(_message(2, "")) == ""


# ---------------------------------------------------------------------------
# plain text rendering + escaping
# ---------------------------------------------------------------------------
def test_plain_text_message_render() -> None:
    md = _render(_full(1, text="Простое сообщение"))

    assert "**10:00** | **Аня**: Простое сообщение" in md


def test_special_characters_are_escaped() -> None:
    md = _render(_full(1, text="*звезда* _подчёрк_ `бэктик` [скобка] \\бэкслеш"))

    assert "*звезда* _подчёрк_ `бэктик` [скобка] \\бэкслеш" not in md
    assert "\\*звезда\\* \\_подчёрк\\_ \\`бэктик\\` \\[скобка\\] \\\\бэкслеш" in md


def test_message_without_text_renders_header_only() -> None:
    md = _render(_full(1, text=None))

    assert "**10:00** | **Аня**" in md
    assert "**10:00** | **Аня**:" not in md


# ---------------------------------------------------------------------------
# entity rendering: bold/italic/mention/link/hashtag/phone/blockquote
# ---------------------------------------------------------------------------
def test_entities_render_as_markdown() -> None:
    msg = _full(
        1,
        text=[
            {"type": "bold", "text": "bold"},
            {"type": "italic", "text": "italic"},
            {"type": "link", "text": "https://example.com"},
            {"type": "mention", "text": "@user"},
        ],
        text_entities=_entities(
            ("bold", "bold"),
            ("italic", "italic"),
            ("link", "https://example.com"),
            ("mention", "@user"),
        ),
    )

    md = _render(msg)

    assert "**bold**" in md
    assert "*italic*" in md
    assert "<https://example.com>" in md
    assert "`@user`" in md


def test_entity_types_hashtag_phone_text_link_emoji_plain() -> None:
    msg = _full(
        1,
        text=[
            {"type": "hashtag", "text": "#пикник"},
            {"type": "phone", "text": "+7 900 123-45-67"},
            {"type": "text_link", "text": "видимый текст"},
            {"type": "custom_emoji", "text": "🦀"},
            {"type": "plain", "text": " обычный"},
        ],
        text_entities=_entities(
            ("hashtag", "#пикник"),
            ("phone", "+7 900 123-45-67"),
            ("text_link", "видимый текст"),
            ("custom_emoji", "🦀"),
            ("plain", " обычный"),
        ),
    )

    md = _render(msg)

    assert "`#пикник`" in md
    assert "`+7 900 123-45-67`" in md
    assert "видимый текст" in md  # text_link renders as plain text
    assert "🦀 обычный" in md


def test_blockquote_entity_renders_quote() -> None:
    msg = _full(
        1,
        text=[{"type": "blockquote", "text": "цитата"}],
        text_entities=_entities(("blockquote", "цитата")),
    )

    md = _render(msg)

    assert "> цитата" in md


# ---------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------
def test_reply_renders_quote_of_replied_message() -> None:
    target = _message(5, "2026-07-24T09:00:00")
    target.from_name = "Борис"
    target.text = "Исходное сообщение"
    reply = _full(6, text="Мой ответ", reply_to_message_id=5)

    md = _render(reply, idx={5: target})

    assert "> ↩ В ответ на **Борис** (09:00):" in md
    assert ">> Исходное сообщение" in md
    assert "Мой ответ" in md


def test_reply_to_missing_message_has_no_quote() -> None:
    reply = _full(6, text="Ответ в пустоту", reply_to_message_id=404)

    md = _render(reply, idx={})

    assert "В ответ на" not in md
    assert "Ответ в пустоту" in md


def test_reply_to_voice_message_quotes_media_label_with_transcript() -> None:
    voice = _full(
        5,
        from_name="Вера",
        media_type="voice_message",
        file="chat/voice.ogg",
        duration_seconds=7,
        text=None,
    )
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)
    md = _render(reply, transcripts={"chat/voice.ogg": "иди сюда"}, idx={5: voice})

    assert "> ↩ В ответ на **Вера** (10:00):" in md
    assert ">> [🎤 Голосовое — иди сюда]" in md


# ---------------------------------------------------------------------------
# forwarded
# ---------------------------------------------------------------------------
def test_forwarded_from_channel_and_user() -> None:
    channel = _full(1, forwarded_from="Лес и парк", forwarded_from_id="channel123")
    md = _render(channel)
    assert "> ↪ Переслано от **Лес и парк** (канал)" in md

    user = _full(2, forwarded_from="Старый друг", forwarded_from_id="user42")
    md = _render(user)
    assert "> ↪ Переслано от **Старый друг** (пользователь)" in md


# ---------------------------------------------------------------------------
# service messages
# ---------------------------------------------------------------------------
def test_service_phone_call_reasons() -> None:
    hangup = _full(
        1, type="service", action="phone_call", actor="Борис",
        discard_reason="hangup", duration_seconds=43,
    )
    md = _render(hangup)
    assert "⚡ **Системное**: Завершён звонок от **Борис** (0:43)" in md

    missed = _full(2, type="service", action="phone_call", actor="Аня", discard_reason="missed")
    md = _render(missed)
    assert "Пропущенный звонок от **Аня**" in md

    busy = _full(3, type="service", action="phone_call", actor="Дима", discard_reason="busy")
    md = _render(busy)
    assert "Отклонён (занято) звонок от **Дима**" in md

    bare = _full(4, type="service", action="phone_call", actor="Оля", discard_reason=None)
    md = _render(bare)
    assert "Неизвестный звонок от **Оля**" in md


def test_service_pin_message() -> None:
    pin = _full(1, type="service", action="pin_message", actor="Вера", message_id=2)

    md = _render(pin)

    assert "⚡ **Системное**: **Вера** закрепил сообщение #2" in md


def test_service_unknown_action() -> None:
    unknown = _full(1, type="service", action=None, actor="Кто-то")
    md = _render(unknown)
    assert "⚡ **Системное**: неизвестное действие" in md

    exotic = _full(2, type="service", action="game_score", actor="Кто-то")
    md = _render(exotic)
    assert "⚡ **Системное**: game_score" in md


# ---------------------------------------------------------------------------
# media rendering
# ---------------------------------------------------------------------------
def test_voice_message_with_and_without_transcript() -> None:
    voice = _full(
        1, media_type="voice_message", file="chat/voice.ogg",
        duration_seconds=74, text=None,
    )

    without = _render(voice, transcripts={})
    assert "**10:00** | **Аня** (1:14)" in without
    assert "**🎤 Голосовое сообщение:**" in without
    assert "> *(расшифровка недоступна)*" in without

    with_transcript = _render(voice, transcripts={"chat/voice.ogg": "Привет, это я"})
    assert "> *Расшифровка:* Привет, это я" in with_transcript
    assert "недоступна" not in with_transcript


def test_round_video_message_renders() -> None:
    video = _full(
        1, media_type="video_message", file="chat/video.mp4",
        duration_seconds=9, text=None,
    )

    md = _render(video, transcripts={})

    assert "**📹 Видеосообщение:**" in md
    assert "> *(расшифровка недоступна)*" in md


def test_sticker_photo_video_animation_and_generic_file() -> None:
    sticker = _full(1, media_type="sticker", sticker_emoji="👋", file="s.webp", text=None)
    md = _render(sticker)
    assert "**10:00** | **Аня** [🎭 Стикер 👋]" in md

    photo = _full(2, photo="(File not included)", text=None)
    md = _render(photo)
    assert "**10:00** | **Аня** [📷 Фото]" in md

    clip = _full(
        3, media_type="video_file", file="chat/clip.mp4",
        file_name="clip.mp4", duration_seconds=47, text=None,
    )
    md = _render(clip)
    assert "[🎬 Видео (0:47) — *clip.mp4*]" in md

    gif = _full(4, media_type="animation", file="chat/a.gif", text=None)
    md = _render(gif)
    assert "**10:00** | **Аня** [🎞️ GIF]" in md

    doc = _full(5, file_name="отчёт.pdf", file="chat/отчёт.pdf", text=None)
    md = _render(doc)
    assert "**10:00** | **Аня** [📎 Файл: *отчёт.pdf*]" in md


# ---------------------------------------------------------------------------
# reactions + edited marker
# ---------------------------------------------------------------------------
def test_reactions_line_with_overflow_names() -> None:
    recent = [
        ReactionRecent(from_name="Борис", from_id="u1", date="2026-07-24T10:01:00"),
        ReactionRecent(from_name="Вера", from_id="u2", date="2026-07-24T10:02:00"),
        ReactionRecent(from_name="Дима", from_id="u3", date="2026-07-24T10:03:00"),
        ReactionRecent(from_name="Оля", from_id="u4", date="2026-07-24T10:04:00"),
    ]
    reactions = [Reaction(type="emoji", count=5, emoji="👍", recent=recent)]
    msg = _full(1, reactions=reactions)

    md = _render(msg)

    assert "👍 ×5 (Борис, Вера, Дима и ещё 1)" in md


def test_reactions_without_recent_hide_names() -> None:
    reactions = [Reaction(type="emoji", count=5, emoji="👍", recent=[])]
    msg = _full(1, reactions=reactions)

    md = _render(msg)

    assert "👍 ×5" in md
    assert "(" not in md.split("👍 ×5")[1].splitlines()[0]


def test_edited_marker_appended() -> None:
    msg = _full(1, text="Было", edited="2026-07-24T10:05:00")

    md = _render(msg)

    assert "*(отредактировано 10:05)*" in md

