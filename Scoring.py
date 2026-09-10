"""
Turns a VLM's free-text step response into a numeric score.

Primary path: parse the model's explicit `SCORE: n/N` line.
Fallback path (only used if the model didn't follow that format): a
keyword-based heuristic.

Bug fixed here: the original heuristic counted "visible" as positive even
inside the phrase "not visible", because it was doing plain substring
containment on both lists independently. A response saying "the label is
not visible" scored a hit for BOTH "visible" (positive) and "not visible"
(negative), diluting the negative signal. We now strip matched negative
phrases out of the text before counting positive phrases, so their
substrings can't double-count.
"""

import re

from config import NEGATIVE_PHRASES, POSITIVE_PHRASES

_SCORE_RE = re.compile(r"score\s*:\s*(\d+)\s*/\s*(\d+)")


def score_response(text, weight):
    text_lower = text.lower()

    explicit = _SCORE_RE.search(text_lower)
    if explicit:
        reported_score = int(explicit.group(1))
        reported_weight = max(1, int(explicit.group(2)))
        return max(0, min(weight, round(reported_score * weight / reported_weight)))

    if "error" in text_lower and "api" in text_lower:
        return int(weight * 0.20)

    # Strip negative phrases out first so e.g. "not visible" can't also
    # register as a hit for the positive phrase "visible".
    remaining = text_lower
    neg_count = 0
    for phrase in NEGATIVE_PHRASES:
        occurrences = remaining.count(phrase)
        if occurrences:
            neg_count += occurrences
            remaining = remaining.replace(phrase, " ")

    pos_count = sum(remaining.count(phrase) for phrase in POSITIVE_PHRASES)

    pos = pos_count * 2
    neg = neg_count * 2

    if pos > neg * 1.5:
        return int(weight * 0.90)
    if pos > neg:
        return int(weight * 0.65)
    if pos == neg:
        return int(weight * 0.50)
    return int(weight * 0.20)


def verdict_for_score(total):
    from config import VERDICT_ACCEPT_MIN, VERDICT_REVIEW_MIN
    if total >= VERDICT_ACCEPT_MIN:
        return "ACCEPTED"
    if total >= VERDICT_REVIEW_MIN:
        return "REVIEW REQUIRED"
    return "REJECTED"


def extract_tracking_number(label_text):
    """Pull a likely AWB/tracking number out of the label step's response."""
    candidates = re.findall(r"\b[A-Z0-9]{8,}\b", label_text.upper())
    false_positives = {
        "TRACKING", "SHIPPING", "BARCODE", "VISIBLE", "CAMERA",
        "LABEL", "CARRIER", "EKART", "CLEARLY", "SHOWING",
    }
    real_numbers = [c for c in candidates if c not in false_positives]
    return real_numbers[0] if real_numbers else "Not visible"
