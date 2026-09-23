"""
Repetition and degeneracy detection for generated responses.

Phase 1 surfaced two responses that collapsed into repetition loops. Neither was
caught by BERTScore, because a degenerate response whose opening sentence is
well-formed stays semantically close to the reference. Degeneracy is therefore
measured directly and recorded per response rather than inferred from similarity
scores.
"""

from collections import Counter

DEGENERATE_REPEAT_THRESHOLD = 4
DEFAULT_NGRAM = 5


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Return the list of n-grams in a token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text: str, n: int = 4) -> float:
    """
    Ratio of unique n-grams to total n-grams.

    Returns 1.0 for text with no repetition; lower values indicate more.
    """
    grams = _ngrams(str(text).lower().split(), n)
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def max_ngram_repeat(text: str, n: int = DEFAULT_NGRAM) -> int:
    """Number of occurrences of the most frequently repeated n-gram."""
    grams = _ngrams(str(text).lower().split(), n)
    if not grams:
        return 0
    return Counter(grams).most_common(1)[0][1]


def repeated_token_fraction(text: str, n: int = DEFAULT_NGRAM) -> float:
    """Fraction of tokens sitting inside an n-gram that occurs more than once."""
    tokens = str(text).lower().split()
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    flagged: set[int] = set()
    for i, g in enumerate(grams):
        if counts[g] > 1:
            flagged.update(range(i, i + n))
    return len(flagged) / len(tokens)


def is_degenerate(text: str, threshold: int = DEGENERATE_REPEAT_THRESHOLD) -> bool:
    """
    True when any n-gram repeats at or above the threshold.

    A threshold of 4 corresponds to output a caregiver would read as broken.
    """
    return max_ngram_repeat(text) >= threshold


def degeneracy_report(text: str) -> dict:
    """Return all degeneracy metrics for one response, for logging into results."""
    return {
        "distinct_4": round(distinct_n(text, 4), 4),
        "max_ngram_repeat": max_ngram_repeat(text),
        "repeated_token_fraction": round(repeated_token_fraction(text), 4),
        "degenerate": int(is_degenerate(text)),
    }
