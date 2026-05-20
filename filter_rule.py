"""filter_rule.py — Cloud et al. (2025) §3 verbatim sequence filter.

Keep iff: 1–10 integers in [0, 999], a single consistent separator
(whitespace / comma / semicolon), optional () or [] wrapping,
optional trailing period. No other characters.

Used by every notebook that filters sequences (Phase 0 generation,
Phase 1 replication, Phase 4 ablations, ...).
"""
import re
from typing import List, Optional

_SEPARATORS = [",", ";", " "]
_DIGIT_RUN = re.compile(r"\d{1,3}")


def parse_completion(text: str) -> Optional[List[int]]:
    """Return the list of integers if `text` passes Cloud's filter, else None."""
    s = text.strip()
    if not s:
        return None
    if s.endswith("."):
        s = s[:-1].rstrip()
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1].strip()
    for sep in _SEPARATORS:
        if sep == " ":
            parts = s.split()
        else:
            if sep not in s:
                continue
            parts = [p.strip() for p in s.split(sep)]
        if not parts or any(not p for p in parts):
            continue
        if not all(_DIGIT_RUN.fullmatch(p) for p in parts):
            continue
        nums = [int(p) for p in parts]
        if not (1 <= len(nums) <= 10):
            continue
        if any(n < 0 or n > 999 for n in nums):
            continue
        other_seps = [c for c in (",", ";") if c != sep]
        if any(c in s for c in other_seps):
            continue
        return nums
    return None


def passes_filter(text: str) -> bool:
    """Return True iff `text` passes Cloud's filter."""
    return parse_completion(text) is not None
