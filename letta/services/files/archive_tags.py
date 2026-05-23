import re
from typing import List, Optional

DEFAULT_FILE_CORE_CHAR_LIMIT = 2000


def normalize_archive_tags(raw_tags: Optional[List[str]], max_tags: int = 16, max_tag_length: int = 32) -> List[str]:
    """Normalize archive tags; drop invalid tags individually."""
    if not raw_tags:
        return []

    normalized: List[str] = []
    seen = set()
    for raw in raw_tags[:max_tags]:
        if not isinstance(raw, str):
            continue
        tag = raw.strip().lower()
        tag = re.sub(r"\s+", "-", tag)
        if not tag or len(tag) > max_tag_length:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized
