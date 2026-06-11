import re
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Callable, Optional

import pytz

from letta.constants import DEFAULT_TIMEZONE


# x-www-form-urlencoded query parsing decodes '+' as space; repair "T... 00:00" offsets.
_ISO_OFFSET_SPACE_RE = re.compile(r"^(?P<head>.+T\d{2}:\d{2}:\d{2}(?:\.\d+)?) (?P<tz>\d{2}:\d{2})$")


def repair_iso_datetime_query_cursor(value: str) -> str:
    """Repair UTC offsets where '+' was decoded as a space in a query string."""
    text = value.strip()
    match = _ISO_OFFSET_SPACE_RE.match(text)
    if match:
        return f"{match.group('head')}+{match.group('tz')}"
    return text


def parse_cursor_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(repair_iso_datetime_query_cursor(value))


def parse_formatted_time(formatted_time):
    # parse times returned by letta.utils.get_formatted_time()
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


def datetime_to_timestamp(dt):
    # convert datetime object to integer timestamp
    return int(dt.timestamp())


def get_local_time_fast(timezone):
    # Get current UTC time and convert to the specified timezone
    # Only return the date to avoid cache busting on every request
    if not timezone:
        return datetime.now().strftime("%B %d, %Y")
    current_time_utc = datetime.now(pytz.utc)
    local_time = current_time_utc.astimezone(pytz.timezone(timezone))
    # Return only the date in a human-readable format (e.g., "June 1, 2021")
    formatted_time = local_time.strftime("%B %d, %Y")

    return formatted_time


def get_local_time_timezone(timezone=DEFAULT_TIMEZONE):
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    local_time = current_time_utc.astimezone(pytz.timezone(timezone))

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return formatted_time


def get_local_time(timezone: str | None = DEFAULT_TIMEZONE):
    if timezone is not None:
        time_str = get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now().astimezone()

        # You may format it as you desire, including AM/PM
        time_str = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return time_str.strip()


def get_utc_time() -> datetime:
    """Get the current UTC time"""
    # return datetime.now(pytz.utc)
    return datetime.now(dt_timezone.utc)


def get_utc_time_int() -> int:
    return int(get_utc_time().timestamp())


def get_utc_timestamp_ns() -> int:
    """Get the current UTC time in nanoseconds"""
    return int(time.time_ns())


def ns_to_ms(ns: int) -> int:
    return ns // 1_000_000


def timestamp_to_datetime(timestamp_seconds: int) -> datetime:
    """Convert Unix timestamp in seconds to UTC datetime object"""
    return datetime.fromtimestamp(timestamp_seconds, tz=dt_timezone.utc)


def format_datetime(dt, timezone):
    if not timezone:
        # use local timezone
        return dt.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")
    return dt.astimezone(pytz.timezone(timezone)).strftime("%Y-%m-%d %I:%M:%S %p %Z%z")


def validate_date_format(date_str):
    """Validate the given date string in the format 'YYYY-MM-DD'."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def extract_date_from_timestamp(timestamp):
    """Extracts and returns the date from the given timestamp."""
    # Extracts the date (ignoring the time and timezone)
    match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
    return match.group(1) if match else None


def is_utc_datetime(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) == timedelta(0)


class AsyncTimer:
    """An async context manager for timing async code execution.

    Takes in an optional callback_func to call on exit with arguments
    taking in the elapsed_ms and exc if present.

    Do not use the start and end times outside of this function as they are relative.
    """

    def __init__(self, callback_func: Callable | None = None):
        self._start_time_ns = None
        self._end_time_ns = None
        self._elapsed_ns = None
        self.callback_func = callback_func

    async def __aenter__(self):
        self._start_time_ns = time.perf_counter_ns()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._end_time_ns = time.perf_counter_ns()
        self._elapsed_ns = self._end_time_ns - self._start_time_ns
        if self.callback_func:
            from asyncio import iscoroutinefunction

            if iscoroutinefunction(self.callback_func):
                await self.callback_func(self.elapsed_ms, exc)
            else:
                self.callback_func(self.elapsed_ms, exc)
        return False

    @property
    def elapsed_ms(self):
        if self._elapsed_ns is not None:
            return ns_to_ms(self._elapsed_ns)
        return None

    @property
    def elapsed_ns(self):
        return self._elapsed_ns
