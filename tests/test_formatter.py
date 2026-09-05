"""Unit tests for day-group ordering in src/formatter.py.

Hermetic: constructs Message objects directly, no network/audio/real chat data.
"""

from datetime import datetime

from src.formatter import _date_key, _day_sort_key, format_markdown
from src.models import Message


def _message(msg_id: int, date: str) -> Message:
    """Minimal plain-text message with the given ISO date field."""
    return Message(id=msg_id, type="message", date=date, date_unixtime="0")


def _section_headers(md: str) -> list[str]:
    """Extract the '## ' day-group headers in document order."""
    return [line for line in md.splitlines() if line.startswith("## ")]


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
