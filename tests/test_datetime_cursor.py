from letta.helpers.datetime_helpers import parse_cursor_datetime, repair_iso_datetime_query_cursor


def test_repair_iso_datetime_query_cursor_plus_decoded_as_space():
    fixed = repair_iso_datetime_query_cursor("2026-06-10T13:03:21.642929 00:00")
    assert fixed == "2026-06-10T13:03:21.642929+00:00"


def test_parse_cursor_datetime_from_repaired_query():
    parsed = parse_cursor_datetime("2026-06-10T13:03:21.642929 00:00")
    assert parsed is not None
    assert parsed.isoformat() == "2026-06-10T13:03:21.642929+00:00"
