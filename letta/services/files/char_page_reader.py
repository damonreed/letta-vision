from typing import Optional, Tuple


class CharPageReader:
    """Character-indexed paging over file text with Unicode-safe boundaries."""

    def __init__(self, text: str, page_size: int):
        self.text = text or ""
        self.page_size = max(1, page_size)
        self.total_chars = len(self.text)
        self.total_pages = max(1, (self.total_chars + self.page_size - 1) // self.page_size) if self.text else 1

    def read_page(self, cursor_char: int) -> Tuple[str, int, int, int]:
        """Read the page at cursor_char. Returns (content, start, end, next_cursor)."""
        if not self.text:
            return "", 0, 0, 0

        start = max(0, min(cursor_char, self.total_chars))
        end = min(start + self.page_size, self.total_chars)
        content = self.text[start:end]
        next_cursor = end if end < self.total_chars else self.total_chars
        return content, start, end, next_cursor

    def read_range(self, start_char: int, end_char: int, max_chars: int) -> Tuple[str, int, int]:
        start = max(0, min(start_char, self.total_chars))
        end = max(start, min(end_char, self.total_chars))
        if end - start > max_chars:
            end = start + max_chars
        return self.text[start:end], start, end

    def page_number_for_cursor(self, cursor_char: int) -> int:
        if self.page_size <= 0:
            return 1
        if cursor_char <= 0:
            return 1
        return min(self.total_pages, (cursor_char // self.page_size) + 1)

    def cursor_for_page(self, page_number: int) -> int:
        page = max(1, min(page_number, self.total_pages))
        return (page - 1) * self.page_size

    def next_page_cursor(self, cursor_char: int) -> int:
        """Start of the next page to read from cursor_char.

        After file_read_page advances the cursor to a page boundary, that position
        is already the start of the next unread page — do not skip past it.
        """
        if not self.text or cursor_char >= self.total_chars:
            return self.total_chars
        if cursor_char > 0 and cursor_char % self.page_size == 0:
            return cursor_char
        _, _, end, _ = self.read_page(cursor_char)
        return min(end, self.total_chars)

    def prev_page_cursor(self, cursor_char: int) -> int:
        current_page = self.page_number_for_cursor(cursor_char)
        return self.cursor_for_page(max(1, current_page - 1))
