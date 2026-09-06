"""Hypothesis property tests over the two historical bug classes.

The day-sorting bug (fixed in commit 1dc4ba8) ordered Russian day-group
headers lexicographically, scrambling "10 июля" before "2 января"; the cache
re-keying bug (fixed in 755efb8) left legacy relative keys pointing at files
that only existed relative to a different CWD. A suite that guards behavior
must hold for arbitrary inputs, not just hand-picked fixtures, so both are
re-tested here as invariants over generated data:

  - test_day_group_headers_property: for any mix of parseable ISO datetimes
    and unparseable garbage dates, format_markdown must emit day groups whose
    parsed headers are chronological, whose garbage groups all land AFTER the
    last parsed group, and whose set of headers matches an oracle built from
    the input (Russian month names parsed back out of the rendered headers —
    the oracle never calls into formatter internals). It must also be
    byte-identical regardless of the message iteration order.
  - test_migrate_cache_keys_property: for any cache dict whose keys point at
    files constructed under a fresh export root (existing relative paths with
    optional bogus leading components, existing/nonexistent absolute paths,
    and junk), migrate_cache_keys must produce exactly the oracle mapping:
    absolute-and-existing keys kept verbatim, everything else dropped or
    re-keyed to the canonical absolute path of the matching existing file.
    Output keys are all absolute and exist on disk; migration is idempotent
    and never mutates its input.

Both are hermetic: tmp_path only, no models, no network. Hypothesis examples
are deterministic, so a failing example is a reproducible regression.
"""

import random
from datetime import datetime
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tg_chat_to_md.cache import migrate_cache_keys
from tg_chat_to_md.formatter import format_markdown
from tg_chat_to_md.models import Message

# ---------------------------------------------------------------------------
# Oracle: parse Russian day-group headers ("5 января 2026") back into
# datetimes with a test-local month table — the header format is pinned here
# independently of tg_chat_to_md.formatter._date_key.
# ---------------------------------------------------------------------------
_RU_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
_MONTH_NUMBER = {name: num for num, name in enumerate(_RU_MONTHS, start=1)}


def _parse_header_date(header: str) -> datetime | None:
    """Parse a day-group header ('D month YYYY'); None when unparseable."""
    parts = header.split()
    if len(parts) != 3 or not parts[0].isdigit() or not parts[2].isdigit():
        return None
    try:
        return datetime(int(parts[2]), _MONTH_NUMBER[parts[1]], int(parts[0]))
    except (KeyError, ValueError):
        return None


def _header_day(dt: datetime) -> tuple[int, int, int]:
    """(year, month, day) tuple — the identity of a calendar day."""
    return (dt.year, dt.month, dt.day)


# ---------------------------------------------------------------------------
# Day-group ordering property (bug class of commit 1dc4ba8)
# ---------------------------------------------------------------------------
# Parseable dates: naive datetimes only (a tz-aware isoformat would still
# parse, but naive keeps the property free of timezone edge cases).
_VALID_DATE = st.datetimes(
    min_value=datetime(2010, 1, 1),
    max_value=datetime(2029, 12, 31, 23, 59, 59),
).map(lambda dt: dt.isoformat())

# Garbage dates: explicit hand-picked junk plus random ASCII text that
# datetime.fromisoformat rejects. ASCII-only fuzzing guarantees no garbage
# string can ever spell a Russian month name, so a garbage group key
# (date[:10] fallback) can never collide with a parsed day's header —
# keeping the header-count oracle exact.
_FIXED_GARBAGE = st.sampled_from(
    [
        "",
        "не дата",
        "дата неизвестна",
        "2026-13-99T99:99",  # invalid month/day/hour, ISO look-alike
        "2026-02-30T25:00:00",
    ]
)
_ASCII_GARBAGE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:T. ",
    min_size=1,
    max_size=30,
).filter(lambda s: _parse_iso(s) is None)
_GARBAGE_DATE = st.one_of(_FIXED_GARBAGE, _ASCII_GARBAGE)


def _parse_iso(date_str: str) -> datetime | None:
    """datetime.fromisoformat that returns None instead of raising."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


@given(dates=st.lists(st.one_of(_VALID_DATE, _GARBAGE_DATE), min_size=1, max_size=30))
def test_day_group_headers_are_chronological_and_garbage_last(dates) -> None:
    """Day-group ordering contract for ANY mix of parseable/garbage dates.

    Oracle (independent of formatter internals): expected parsed-day headers
    and garbage [:10]-fallback keys are computed straight from the input
    date strings; the actual '## ' headers of the rendered document are
    parsed back through the Russian month table. The output must contain
    exactly the expected parsed headers in chronological order, every
    unparseable group after the last parsed one, and exactly the distinct
    garbage [:10] prefixes as groups — plus identical bytes for both the
    canonical (id-sorted) and a shuffled message order.
    """
    messages = [
        Message(id=idx, type="message", date=date_str, date_unixtime="0")
        for idx, date_str in enumerate(dates, start=1)
    ]

    # Expected grouping, straight from the input strings.
    parsed_days: set[tuple[int, int, int]] = set()
    garbage_keys: set[str] = set()
    for date_str in dates:
        parsed = _parse_iso(date_str)
        if parsed is not None:
            parsed_days.add(_header_day(parsed))
        else:
            # The [:10] fallback of the group key mirrors _date_key's garbage
            # branch; it is the documented identity of an unparseable group
            # and is asserted here as part of the group-identity contract.
            garbage_keys.add(date_str[:10])

    # Render twice: canonical order and a shuffled permutation.
    shuffled = list(messages)
    random.Random(42).shuffle(shuffled)
    md_sorted = format_markdown(messages, chat_name="Чат", transcripts={})
    md_shuffled = format_markdown(shuffled, chat_name="Чат", transcripts={})
    assert md_sorted == md_shuffled  # bytes must not depend on iteration order

    headers = [line[3:] for line in md_sorted.splitlines() if line.startswith("## ")]
    parsed_headers: list[datetime] = []
    garbage_headers: list[str] = []
    for header in headers:
        day = _parse_header_date(header)
        (parsed_headers if day is not None else garbage_headers).append(day or header)

    # (1) Parsed headers: exactly the expected days, in chronological order.
    actual_days = {_header_day(dt) for dt in parsed_headers}
    assert actual_days == parsed_days
    assert len(parsed_headers) == len(parsed_days)
    assert parsed_headers == sorted(parsed_headers)

    # (2) Garbage groups: exactly the distinct [:10] fallback keys.
    assert set(garbage_headers) == garbage_keys
    assert len(garbage_headers) == len(garbage_keys)

    # (3) Every unparseable group lands after the last parsed group.
    if parsed_headers and garbage_headers:
        parsed_idx = [i for i, h in enumerate(headers) if _parse_header_date(h) is not None]
        garbage_idx = [i for i, h in enumerate(headers) if _parse_header_date(h) is None]
        assert min(garbage_idx) > max(parsed_idx)

    # (4) No phantom or missing groups beyond the two channels.
    assert len(headers) == len(parsed_headers) + len(garbage_headers)


# ---------------------------------------------------------------------------
# migrate_cache_keys property (bug class of commit 755efb8)
# ---------------------------------------------------------------------------
# Construction uses two DISJOINT segment alphabets so every generated key's
# fate is decidable from the draw alone, without re-running the migration
# loop:
#   - files under the export root live at paths made of "abc12" segments;
#   - junk prefixes / junk-only (missing) keys are made of "xyz" segments and
#     can never match anything created under the root.
_REL_SEG = st.text(alphabet="abc12", min_size=1, max_size=6)
_JUNK_SEG = st.text(alphabet="xyz", min_size=1, max_size=3)
_TEXT = st.text(alphabet="абвгдежзиклмнопрстуфхцчшщыэюя0123 ", min_size=0, max_size=24)

_rel_path = st.lists(_REL_SEG, min_size=1, max_size=3)  # depth <= 3
_junk_path = st.lists(_JUNK_SEG, min_size=1, max_size=3)
# Existing-file names come from the "mnopq" pool; missing-key names from
# "ghik" (g/e letters absent from the mnopq pool) — plus a "gone_" prefix —
# so a missing absolute key can never collide with a real file name.
_abs_existing_name = st.text(alphabet="mnopq", min_size=1, max_size=8)
_abs_missing_name = st.text(alphabet="ghik", min_size=1, max_size=8)


@st.composite
def _cache_entry(draw):
    """One cache entry: kind, path material, and transcript text."""
    kind = draw(st.integers(min_value=0, max_value=4))
    text = draw(_TEXT)
    if kind == 0:  # relative path to an existing file
        return {"kind": "rel", "rel": draw(_rel_path), "text": text}
    if kind == 1:  # existing file under a bogus leading component
        return {"kind": "junk", "rel": draw(_rel_path), "junk": draw(_junk_path), "text": text}
    if kind == 2:  # absolute path to an existing file
        return {"kind": "abs-existing", "name": draw(_abs_existing_name), "text": text}
    if kind == 3:  # absolute path to a nonexistent file
        return {"kind": "abs-missing", "name": draw(_abs_missing_name), "text": text}
    # relative junk path whose file does not exist
    return {"kind": "missing", "path": draw(_junk_path), "text": text}


def _entries_ok(entries: list[dict]) -> bool:
    """Reject draws that would make the exact oracle ambiguous.

    Every input key string must be unique (a duplicate would silently collapse
    in the cache dict), and no file rel may be a segment-prefix of another
    (that would force one file onto a path where another file's ancestor
    directory must live). All alphabets are disjoint, so only within-kind
    collisions are possible.
    """
    rels = [tuple(e["rel"]) for e in entries if e["kind"] in ("rel", "junk")]
    for i, a in enumerate(rels):
        for b in rels[i + 1:]:
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if shorter != longer and longer[: len(shorter)] == shorter:
                return False

    def unique(seq) -> bool:
        """All drawn items distinct (generator-safe)."""
        items = list(seq)
        return len(items) == len(set(items))

    return (
        unique(tuple(e["rel"]) for e in entries if e["kind"] == "rel")
        and unique(tuple(e["junk"] + e["rel"]) for e in entries if e["kind"] == "junk")
        and unique(e["name"] for e in entries if e["kind"] == "abs-existing")
        and unique(e["name"] for e in entries if e["kind"] == "abs-missing")
        and unique(tuple(e["path"]) for e in entries if e["kind"] == "missing")
    )


def _join(segments: list[str]) -> str:
    return "/".join(segments)


def _entry_key(entry: dict, outside_dir: Path) -> str:
    """The literal cache key an entry produces (relative or absolute)."""
    kind = entry["kind"]
    if kind == "rel":
        return _join(entry["rel"])
    if kind == "junk":
        return _join(entry["junk"] + entry["rel"])
    if kind == "abs-existing":
        return str(outside_dir / entry["name"])
    if kind == "abs-missing":
        return str(outside_dir / ("gone_" + entry["name"]))
    return _join(entry["path"])


def _longest_suffix_rel(key: str, rels: set[tuple[str, ...]]) -> tuple[str, ...] | None:
    """The longest drawn file rel that is a suffix of the relative key.

    Under the construction (junk segments can never exist on disk) the
    migration loop's first existing candidate is exactly the longest existing
    suffix, and the longest drawn rel is that suffix — so this classifies
    keys without mirroring the loop.
    """
    for rel in sorted(rels, key=len, reverse=True):
        joined = _join(list(rel))
        if key == joined or key.endswith("/" + joined):
            return rel
    return None


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(entries=st.lists(_cache_entry(), min_size=0, max_size=8).filter(_entries_ok))
def test_migrate_cache_keys_matches_exact_oracle(entries, tmp_path) -> None:
    """Migration contract over generated cache dicts and a real export root.

    Every example builds a fresh export root with the files its entries refer
    to, runs migrate_cache_keys, and compares against the oracle: existing
    absolute keys kept verbatim; relative keys re-keyed to the canonical
    absolute path of their file; keys whose file does not exist dropped.
    Cross-checks: all output keys absolute and existing on disk, idempotence,
    and input dict purity.

    tmp_path is shared across examples by design — the per-example counter
    below guarantees each generated input gets its OWN export and outside
    dirs, so no example can ever observe files created by another one.
    """
    # Fresh per-example directories: hypothesis reuses tmp_path across
    # examples, and a leftover file from an earlier example must never count
    # as "existing" for a later one.
    root = tmp_path / f"export-{_example_id()}"
    outside_dir = tmp_path / f"outside-{_example_id()}"
    root.mkdir()
    outside_dir.mkdir()

    # Files the entries reference — every rel appears once, however often
    # entries point at it.
    rels: set[tuple[str, ...]] = {
        tuple(e["rel"]) for e in entries if e["kind"] in ("rel", "junk")
    }
    for rel in sorted(rels, key=lambda r: (len(r), r)):
        target = root.joinpath(*rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    for entry in entries:
        if entry["kind"] == "abs-existing":
            (outside_dir / entry["name"]).write_bytes(b"")

    cache = {_entry_key(e, outside_dir): e["text"] for e in entries}
    snapshot = dict(cache)

    migrated = migrate_cache_keys(cache, root)

    # Oracle: expected mapping by construction.
    expected: dict[str, str] = {}
    for key, text in cache.items():
        if Path(key).is_absolute():
            if Path(key).exists():
                expected[key] = text  # absolute + existing: verbatim
            continue  # absolute + missing: dropped
        rel = _longest_suffix_rel(key, rels)
        if rel is not None:
            expected[str(root.joinpath(*rel).resolve())] = text

    assert migrated == expected
    # All surviving keys are canonical absolute paths of real files.
    for key in migrated:
        assert Path(key).is_absolute()
        assert Path(key).exists()
    # Idempotent: a second migration over the canonical output is a no-op.
    assert migrate_cache_keys(migrated, root) == migrated
    # Pure: the input dict was not mutated.
    assert cache == snapshot


_COUNTER = iter(range(10**6))


def _example_id() -> int:
    """Monotonic per-example id for fresh tmp subdirectories."""
    return next(_COUNTER)
