"""Tests for CharPageReader."""

from letta.services.files.char_page_reader import CharPageReader


def test_char_page_reader_unicode_safe():
    text = "Hello 世界 — em dash"
    reader = CharPageReader(text, page_size=10)
    content, start, end, _ = reader.read_page(0)
    assert content
    assert start == 0
    assert end <= len(text)


def test_char_page_reader_pages():
    text = "a" * 25
    reader = CharPageReader(text, page_size=10)
    assert reader.total_pages == 3
    _, _, _, next_cursor = reader.read_page(0)
    assert next_cursor == 10


def test_char_page_reader_empty():
    reader = CharPageReader("", page_size=10)
    content, start, end, next_cursor = reader.read_page(0)
    assert content == ""
    assert start == end == next_cursor == 0


def test_next_page_cursor_after_read_page_advances_to_second_page():
    """After page 1 read leaves cursor at page boundary, next_page reads page 2 start."""
    text = "a" * 17560
    reader = CharPageReader(text, page_size=10000)
    _, _, _, cursor_after_page1 = reader.read_page(0)
    assert cursor_after_page1 == 10000
    assert reader.next_page_cursor(cursor_after_page1) == 10000
    content, start, end, _ = reader.read_page(reader.next_page_cursor(cursor_after_page1))
    assert start == 10000
    assert end == 17560
    assert len(content) == 7560


def test_next_page_cursor_from_start_skips_first_page():
    text = "a" * 25000
    reader = CharPageReader(text, page_size=10000)
    assert reader.next_page_cursor(0) == 10000
