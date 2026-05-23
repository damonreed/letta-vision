"""Tests for archive tag normalization."""

from letta.services.files.archive_tags import normalize_archive_tags


def test_normalize_archive_tags_basic():
    assert normalize_archive_tags(["Symbolism", "  Theme One  "]) == ["symbolism", "theme-one"]


def test_normalize_archive_tags_drops_long():
    long_tag = "a" * 33
    assert normalize_archive_tags([long_tag, "ok"]) == ["ok"]


def test_normalize_archive_tags_deduplicates():
    assert normalize_archive_tags(["foo", "FOO"]) == ["foo"]


def test_normalize_archive_tags_empty_input():
    assert normalize_archive_tags([]) == []
    assert normalize_archive_tags(None) == []
