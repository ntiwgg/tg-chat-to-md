"""Unit tests for src/formatter.py message rendering and day-group ordering.

Hermetic: constructs Message objects directly, no network/audio/real chat data.
Rendering tests assert EXACT body lines (via _body_lines) rather than
substrings, so a dropped or duplicated line fails the oracle. Covers:
day-group ordering, plain text, entity kinds (parametrized, exact), escaping,
replies incl. truncation boundaries, forwards, service messages (parametrized),
media kinds (parametrized, exact blocks) incl. captions and durations >= 1h,
reactions with exact name-overflow thresholds, and the edited marker.
"""

from datetime import datetime

import pytest

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
    """Render one message through format_markdown."""
    return format_markdown([msg], "Тест", transcripts or {}, idx or {})


def _body_lines(md: str) -> list[str]:
    """Body lines of a single-message document (document title stripped)."""
    return [
        line
        for line in md.splitlines()
        if line and not line.startswith(("## ", "---"))
    ][1:]  # first non-empty line is the "# Чат: ..." document title


def _entities(*pairs: tuple[str, str]) -> list[TextEntity]:
    return [TextEntity(type=t, text=s) for t, s in pairs]


def _long_text(length: int, fill: str = "а") -> str:
    """Deterministic filler of exactly `length` chars (no markdown specials)."""
    return (fill * (length + 3))[:length]


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
def test_plain_text_message_render_exact_line() -> None:
    md = _render(_full(1, text="Простое сообщение"))

    assert _body_lines(md) == ["**10:00** | **Аня**: Простое сообщение"]


def test_special_characters_are_escaped() -> None:
    md = _render(_full(1, text="*звезда* _подчёрк_ `бэктик` [скобка] \\бэкслеш"))

    assert "*звезда* _подчёрк_ `бэктик` [скобка] \\бэкслеш" not in md
    assert "\\*звезда\\* \\_подчёрк\\_ \\`бэктик\\` \\[скобка\\] \\\\бэкслеш" in md


def test_message_without_text_renders_header_only() -> None:
    md = _render(_full(1, text=None))

    assert _body_lines(md) == ["**10:00** | **Аня**"]


# ---------------------------------------------------------------------------
# entity rendering: one exact oracle per entity kind + interleaved sequence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kind", "raw_text", "expected_md"),
    [
        ("bold", "важно", "**важно**"),
        ("italic", "курсив", "*курсив*"),
        ("code", "x = 1", "`x = 1`"),
        ("link", "https://example.com", "<https://example.com>"),
        ("mention", "@user", "`@user`"),
        ("hashtag", "#пикник", "`#пикник`"),
        ("phone", "+7 900 123-45-67", "`+7 900 123-45-67`"),
        ("text_link", "видимый текст", "видимый текст"),  # renders as plain text
        ("custom_emoji", "🦀", "🦀"),
        ("plain", "обычный текст", "обычный текст"),  # negative: no mangling
        ("blockquote", "цитата", "> цитата"),
    ],
)
def test_entity_kind_renders_exact_markdown(kind, raw_text, expected_md) -> None:
    msg = _full(
        1,
        text=[{"type": kind, "text": raw_text}],
        text_entities=_entities((kind, raw_text)),
    )

    md = _render(msg)

    assert _body_lines(md) == [f"**10:00** | **Аня**: {expected_md}"]
    assert md.count(expected_md) == 1


def test_entity_sequence_renders_exact_line_in_order() -> None:
    """Entities keep document order with plain segments interleaved; a
    mid-line blockquote renders inline (the golden chat.md pins the same
    pattern for message id 23)."""
    msg = _full(
        1,
        text=[
            {"type": "bold", "text": "жирный"},
            " и ",
            {"type": "italic", "text": "курсив"},
            " — ",
            {"type": "link", "text": "https://e.com"},
            " / ",
            {"type": "mention", "text": "@user"},
            " / ",
            {"type": "hashtag", "text": "#пикник"},
            "; далее ",
            {"type": "blockquote", "text": "цитата"},
        ],
        text_entities=_entities(
            ("bold", "жирный"),
            ("plain", " и "),
            ("italic", "курсив"),
            ("plain", " — "),
            ("link", "https://e.com"),
            ("plain", " / "),
            ("mention", "@user"),
            ("plain", " / "),
            ("hashtag", "#пикник"),
            ("plain", "; далее "),
            ("blockquote", "цитата"),
        ),
    )

    md = _render(msg)

    assert _body_lines(md) == [
        "**10:00** | **Аня**: **жирный** и *курсив* — "
        "<https://e.com> / `@user` / `#пикник`; далее > цитата"
    ]


# ---------------------------------------------------------------------------
# replies (exact blocks) + truncation boundaries
# ---------------------------------------------------------------------------
def test_reply_renders_quote_of_replied_message_exact_block() -> None:
    target = _message(5, "2026-07-24T09:00:00")
    target.from_name = "Борис"
    target.text = "Исходное сообщение"
    reply = _full(6, text="Мой ответ", reply_to_message_id=5)

    md = _render(reply, idx={5: target})

    assert _body_lines(md) == [
        "> ↩ В ответ на **Борис** (09:00):",
        ">> Исходное сообщение",
        "**10:00** | **Аня**: Мой ответ",
    ]


def test_reply_to_missing_message_has_no_quote() -> None:
    reply = _full(6, text="Ответ в пустоту", reply_to_message_id=404)

    md = _render(reply, idx={})

    assert _body_lines(md) == ["**10:00** | **Аня**: Ответ в пустоту"]


@pytest.mark.parametrize("text_length", [199, 200, 201, 500])
def test_reply_text_preview_truncation_boundary_at_200(text_length) -> None:
    original = _long_text(text_length)
    target = _message(5, "2026-07-24T09:00:00")
    target.from_name = "Борис"
    target.text = original
    reply = _full(6, text="Мой ответ", reply_to_message_id=5)

    body = _body_lines(_render(reply, idx={5: target}))

    if text_length <= 200:
        assert body[1] == f">> {original}"
    else:
        assert body[1] == f">> {original[:200]}…"


def test_voice_reply_preview_quotes_label_with_transcript() -> None:
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

    assert _body_lines(md) == [
        "> ↩ В ответ на **Вера** (10:00):",
        ">> [🎤 Голосовое — иди сюда]",
        "**10:00** | **Аня**: Спасибо!",
    ]


@pytest.mark.parametrize("transcript_length", [150, 151, 400])
def test_voice_reply_preview_truncates_transcript_suffix_at_150(transcript_length) -> None:
    transcript = _long_text(transcript_length, fill="б")
    voice = _full(
        5,
        from_name="Вера",
        media_type="voice_message",
        file="chat/voice.ogg",
        duration_seconds=7,
        text=None,
    )
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)

    body = _body_lines(
        _render(reply, transcripts={"chat/voice.ogg": transcript}, idx={5: voice})
    )

    suffix = transcript if transcript_length <= 150 else transcript[:150] + "…"
    assert body[1] == f">> [🎤 Голосовое — {suffix}]"


# ---------------------------------------------------------------------------
# forwarded
# ---------------------------------------------------------------------------
def test_forwarded_from_channel_and_user() -> None:
    channel = _full(1, forwarded_from="Лес и парк", forwarded_from_id="channel123")
    md = _render(channel)
    assert _body_lines(md) == [
        "> ↪ Переслано от **Лес и парк** (канал)",
        "**10:00** | **Аня**: текст",
    ]

    user = _full(2, forwarded_from="Старый друг", forwarded_from_id="user42")
    md = _render(user)
    assert _body_lines(md) == [
        "> ↪ Переслано от **Старый друг** (пользователь)",
        "**10:00** | **Аня**: текст",
    ]


# ---------------------------------------------------------------------------
# service messages: parametrized phone_call labels, exact lines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("discard_reason", "actor", "duration_seconds", "expected_line"),
    [
        ("hangup", "Борис", 43, "⚡ **Системное**: Завершён звонок от **Борис** (0:43)"),
        ("missed", "Аня", None, "⚡ **Системное**: Пропущенный звонок от **Аня**"),
        ("busy", "Дима", None, "⚡ **Системное**: Отклонён (занято) звонок от **Дима**"),
        # Unknown/unset reasons all render as the neutral label.
        (None, "Оля", None, "⚡ **Системное**: Неизвестный звонок от **Оля**"),
        ("mystery_reason", "Оля", None, "⚡ **Системное**: Неизвестный звонок от **Оля**"),
        ("", "Оля", None, "⚡ **Системное**: Неизвестный звонок от **Оля**"),
    ],
)
def test_service_phone_call_reason_label_exact(
    discard_reason, actor, duration_seconds, expected_line
) -> None:
    call = _full(
        1,
        type="service",
        action="phone_call",
        actor=actor,
        discard_reason=discard_reason,
        duration_seconds=duration_seconds,
    )

    md = _render(call)

    assert _body_lines(md) == [expected_line]
    assert md.count(expected_line) == 1


def test_service_pin_message_exact_line() -> None:
    pin = _full(1, type="service", action="pin_message", actor="Вера", message_id=2)

    assert _body_lines(_render(pin)) == ["⚡ **Системное**: **Вера** закрепил сообщение #2"]


@pytest.mark.parametrize(
    ("action", "expected_line"),
    [
        (None, "⚡ **Системное**: неизвестное действие"),
        ("game_score", "⚡ **Системное**: game_score"),
    ],
)
def test_service_unknown_action_exact_line(action, expected_line) -> None:
    unknown = _full(1, type="service", action=action, actor="Кто-то")

    assert _body_lines(_render(unknown)) == [expected_line]


# ---------------------------------------------------------------------------
# media rendering: one exact block per media kind (no captions here; see
# caption tests below), durations incl. >= 1h
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mid", "fields", "transcripts", "expected_body"),
    [
        (
            1,
            dict(media_type="voice_message", file="chat/voice.ogg", duration_seconds=74, text=None),
            {},
            [
                "**10:00** | **Аня** (1:14)",
                "**🎤 Голосовое сообщение:**",
                "> *(расшифровка недоступна)*",
            ],
        ),
        (
            2,
            dict(media_type="voice_message", file="chat/voice.ogg", duration_seconds=74, text=None),
            {"chat/voice.ogg": "Привет, это я"},
            [
                "**10:00** | **Аня** (1:14)",
                "**🎤 Голосовое сообщение:**",
                "> *Расшифровка:* Привет, это я",
            ],
        ),
        (
            3,
            dict(media_type="video_message", file="chat/video.mp4", duration_seconds=9, text=None),
            {},
            [
                "**10:00** | **Аня** (0:09)",
                "**📹 Видеосообщение:**",
                "> *(расшифровка недоступна)*",
            ],
        ),
        (
            4,
            dict(media_type="sticker", sticker_emoji="👋", file="s.webp", text=None),
            {},
            ["**10:00** | **Аня** [🎭 Стикер 👋]"],
        ),
        # Sticker without emoji metadata renders the bare label.
        (
            5,
            dict(media_type="sticker", file="s.webp", text=None),
            {},
            ["**10:00** | **Аня** [🎭 Стикер]"],
        ),
        (6, dict(photo="(File not included)", text=None), {}, ["**10:00** | **Аня** [📷 Фото]"]),
        (
            7,
            dict(
                media_type="video_file",
                file="chat/clip.mp4",
                file_name="clip.mp4",
                duration_seconds=47,
                text=None,
            ),
            {},
            ["**10:00** | **Аня** [🎬 Видео (0:47) — *clip.mp4*]"],
        ),
        (
            8,
            dict(media_type="animation", file="chat/a.gif", text=None),
            {},
            ["**10:00** | **Аня** [🎞️ GIF]"],
        ),
        (
            9,
            dict(file_name="отчёт.pdf", file="chat/отчёт.pdf", text=None),
            {},
            ["**10:00** | **Аня** [📎 Файл: *отчёт.pdf*]"],
        ),
    ],
)
def test_media_kind_renders_exact_block(mid, fields, transcripts, expected_body) -> None:
    md = _render(_full(mid, **fields), transcripts=transcripts)

    assert _body_lines(md) == expected_body


@pytest.mark.parametrize(
    ("duration_seconds", "duration_md"),
    [
        (3599, "(59:59)"),
        (3600, "(60:00)"),  # durations >= 1h stay M:SS, no hour field
        (3661, "(61:01)"),
    ],
)
def test_media_duration_format_is_minutes_seconds(duration_seconds, duration_md) -> None:
    clip = _full(
        1,
        media_type="video_file",
        file="chat/clip.mp4",
        file_name="clip.mp4",
        duration_seconds=duration_seconds,
        text=None,
    )

    md = _render(clip)

    assert _body_lines(md) == [f"**10:00** | **Аня** [🎬 Видео {duration_md} — *clip.mp4*]"]


def test_voice_message_caption_renders_below_media_block() -> None:
    """REGRESSION: captions on media messages were silently dropped — the
    media branches never rendered msg.text. The caption must appear in the
    output, rendered through the same entity path as plain messages."""
    voice = _full(
        1,
        media_type="voice_message",
        file="chat/voice.ogg",
        duration_seconds=14,
        text="Скажи спасибо котику",
    )

    md = _render(voice)

    assert _body_lines(md) == [
        "**10:00** | **Аня** (0:14)",
        "**🎤 Голосовое сообщение:**",
        "> *(расшифровка недоступна)*",
        "Скажи спасибо котику",
    ]


def test_voice_message_caption_entities_are_formatted() -> None:
    voice = _full(
        1,
        media_type="voice_message",
        file="chat/voice.ogg",
        duration_seconds=14,
        text=["Скажи ", {"type": "bold", "text": "спасибо"}, " котику"],
        text_entities=_entities(("plain", "Скажи "), ("bold", "спасибо"), ("plain", " котику")),
    )

    md = _render(voice, transcripts={"chat/voice.ogg": "привет"})

    assert _body_lines(md) == [
        "**10:00** | **Аня** (0:14)",
        "**🎤 Голосовое сообщение:**",
        "> *Расшифровка:* привет",
        "Скажи **спасибо** котику",
    ]


def test_photo_caption_renders_after_label() -> None:
    photo = _full(1, photo="(File not included)", text="Наш плед и термос")

    md = _render(photo)

    assert _body_lines(md) == [
        "**10:00** | **Аня** [📷 Фото]",
        "Наш плед и термос",
    ]


# ---------------------------------------------------------------------------
# reactions: exact lines across the "и ещё N" thresholds + edited marker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("recent_names", "count", "expected_line"),
    [
        ([], 3, "  *👍 ×3*"),  # no recents: no names at all
        (["Борис"], 1, "  *👍 ×1 (Борис)*"),
        (["Борис", "Вера", "Дима"], 3, "  *👍 ×3 (Борис, Вера, Дима)*"),  # exactly 3: no overflow
        (["Борис", "Вера", "Дима", "Оля"], 5, "  *👍 ×5 (Борис, Вера, Дима и ещё 1)*"),
        (["Борис", "Вера", "Дима", "Оля", "Коля"], 5, "  *👍 ×5 (Борис, Вера, Дима и ещё 2)*"),
        (
            ["Борис", "Вера", "Дима", "Оля", "Коля", "Таня", "Женя"],
            10,
            "  *👍 ×10 (Борис, Вера, Дима и ещё 4)*",  # overflow = len(recent) - 3
        ),
    ],
)
def test_reaction_line_exact_across_overflow_thresholds(recent_names, count, expected_line) -> None:
    recent = [
        ReactionRecent(from_name=name, from_id=f"u{i}", date=f"2026-07-24T10:0{i}:00")
        for i, name in enumerate(recent_names)
    ]
    msg = _full(1, reactions=[Reaction(type="emoji", count=count, emoji="👍", recent=recent)])

    md = _render(msg)

    assert _body_lines(md) == ["**10:00** | **Аня**: текст", expected_line]
    assert md.count(expected_line) == 1


def test_reactions_do_not_quote_count_without_recent() -> None:
    """count must never be derived from len(recent): a reaction without
    recent names shows only 'emoji ×count' even when count > 0."""
    reactions = [Reaction(type="emoji", count=5, emoji="👍", recent=[])]
    msg = _full(1, reactions=reactions)

    md = _render(msg)

    assert _body_lines(md) == ["**10:00** | **Аня**: текст", "  *👍 ×5*"]


def test_edited_marker_appended() -> None:
    msg = _full(1, text="Было", edited="2026-07-24T10:05:00")

    md = _render(msg)

    assert _body_lines(md) == [
        "**10:00** | **Аня**: Было",
        "  *(отредактировано 10:05)*",
    ]


# ---------------------------------------------------------------------------
# mutation oracles: captions on every caption-capable media kind, media
# headers without optional metadata, and reply previews of media messages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fields", "label_line"),
    [
        (
            dict(media_type="sticker", sticker_emoji="👋", file="s.webp", text="кот"),
            "**10:00** | **Аня** [🎭 Стикер 👋]",
        ),
        (
            dict(media_type="animation", file="chat/a.gif", text="так и было"),
            "**10:00** | **Аня** [🎞️ GIF]",
        ),
        (
            dict(file_name="отчёт.pdf", file="chat/отчёт.pdf", text="читай"),
            "**10:00** | **Аня** [📎 Файл: *отчёт.pdf*]",
        ),
    ],
)
def test_media_kind_caption_renders_after_label(fields, label_line) -> None:
    """The _append_caption call in every media branch must render msg.text —
    a mutation dropping the lines list crashes or loses the caption."""
    md = _render(_full(1, **fields))

    assert _body_lines(md) == [label_line, fields["text"]]


def test_video_file_without_name_renders_bare_label() -> None:
    """file_name is optional: a nameless video_file header must not gain an
    empty or placeholder file segment."""
    clip = _full(1, media_type="video_file", file="chat/clip.mp4", duration_seconds=47, text=None)

    md = _render(clip)

    assert _body_lines(md) == ["**10:00** | **Аня** [🎬 Видео (0:47)]"]


def test_round_video_reply_preview_quotes_label_with_transcript() -> None:
    video = _full(
        5,
        from_name="Вера",
        media_type="video_message",
        file="chat/video.mp4",
        duration_seconds=9,
        text=None,
    )
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)

    md = _render(reply, transcripts={"chat/video.mp4": "иди сюда"}, idx={5: video})

    assert _body_lines(md) == [
        "> ↩ В ответ на **Вера** (10:00):",
        ">> [📹 Видеосообщение — иди сюда]",
        "**10:00** | **Аня**: Спасибо!",
    ]


def test_round_video_reply_preview_without_transcript_has_no_suffix() -> None:
    video = _full(
        5,
        from_name="Вера",
        media_type="video_message",
        file="chat/video.mp4",
        duration_seconds=9,
        text=None,
    )
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)

    md = _render(reply, transcripts={}, idx={5: video})

    assert _body_lines(md) == [
        "> ↩ В ответ на **Вера** (10:00):",
        ">> [📹 Видеосообщение]",
        "**10:00** | **Аня**: Спасибо!",
    ]


def test_sticker_reply_preview_quotes_emoji_when_present() -> None:
    sticker = _full(
        5,
        from_name="Вера",
        media_type="sticker",
        sticker_emoji="👋",
        file="s.webp",
        text=None,
    )
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)

    md = _render(reply, transcripts={}, idx={5: sticker})

    assert _body_lines(md) == [
        "> ↩ В ответ на **Вера** (10:00):",
        ">> [🎭 Стикер 👋]",
        "**10:00** | **Аня**: Спасибо!",
    ]


def test_sticker_reply_preview_without_emoji_has_bare_label() -> None:
    sticker = _full(5, from_name="Вера", media_type="sticker", file="s.webp", text=None)
    reply = _full(6, text="Спасибо!", reply_to_message_id=5)

    md = _render(reply, transcripts={}, idx={5: sticker})

    assert _body_lines(md) == [
        "> ↩ В ответ на **Вера** (10:00):",
        ">> [🎭 Стикер]",
        "**10:00** | **Аня**: Спасибо!",
    ]
