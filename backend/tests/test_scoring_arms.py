"""
Tests for arm-aware rubric scoring.

The scoring script split responses on a boolean `rag` flag. That works for two
arms and breaks silently in shape for three: every retrieval arm carries
rag_enabled=1, so they pool into one group, the per-prompt index acquires
duplicates, and the paired test receives arrays of different lengths. The cost
of finding that out is a whole grading pass, so it is tested here.

Run standalone, no pytest:

  python tests/test_scoring_arms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from score_rag_rubric import _arm_frame  # noqa: E402

from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.scoring_arms")

BASE_DIR = Path(__file__).parent.parent
PUBLISHED = BASE_DIR / "data" / "grading_20260825" / "rerun_rubric_paired.csv"

PASSED = []
FAILED = []


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _frame(arms: list[str], prompts: list[str]) -> pd.DataFrame:
    """A minimal scored frame with one row per (arm, prompt)."""
    return pd.DataFrame([
        {"arm": arm, "prompt_id": pid, "run": "m zero_shot", "quality_score": 2}
        for arm in arms for pid in prompts
    ])


@test
def test_selecting_one_arm_leaves_one_row_per_prompt():
    """The case that used to fail: three arms all marked rag_enabled=1."""
    frame = _frame(["baseline", "rag", "routed", "union"], ["P001", "P002", "P003"])
    for arm in ("baseline", "rag", "routed", "union"):
        indexed = _arm_frame(frame, arm, "prompt_id", "test")
        assert len(indexed) == 3, f"{arm} gave {len(indexed)} rows, expected 3"
        assert not indexed.index.has_duplicates


@test
def test_two_arms_can_still_be_paired_after_four_exist():
    """Adding arms must not disturb an existing pair."""
    frame = _frame(["baseline", "rag", "routed", "union"], ["P001", "P002"])
    base = _arm_frame(frame, "baseline", "prompt_id", "test")
    rag = _arm_frame(frame, "rag", "prompt_id", "test")
    shared = base.index.intersection(rag.index)
    assert len(shared) == 2
    assert len(base.loc[shared]) == len(rag.loc[shared]), "arrays must align"


@test
def test_a_duplicate_prompt_within_an_arm_raises():
    """Two responses to one prompt in one arm makes pairing meaningless."""
    frame = _frame(["rag"], ["P001", "P001"])
    try:
        _arm_frame(frame, "rag", "prompt_id", "test")
    except ValueError as exc:
        assert "more than one response" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("expected a ValueError on a duplicated prompt id")


@test
def test_an_unknown_arm_names_the_arms_that_exist():
    """A typo in --contrast-arm must not silently select nothing."""
    frame = _frame(["baseline", "rag"], ["P001"])
    try:
        _arm_frame(frame, "routed", "prompt_id", "test")
    except ValueError as exc:
        assert "baseline" in str(exc) and "rag" in str(exc), \
            f"error should list the arms present, said: {exc}"
    else:
        raise AssertionError("expected a ValueError for an unknown arm")


@test
def test_the_published_two_arm_result_is_unchanged():
    """
    The regression guard on numbers already in the report.

    F-P2-013 quotes these figures. If the arm refactor moves them, the finding
    and the commit that recorded it are both wrong.
    """
    if not PUBLISHED.exists():
        logger.warning("SKIP  %s not present", PUBLISHED)
        return
    paired = pd.read_csv(PUBLISHED)
    pooled = paired[paired["run"] == "ALL"].set_index("measure")
    quality = pooled.loc["quality_score"]
    hallucination = pooled.loc["hallucination_flag"]
    assert round(float(quality["baseline"]), 3) == 1.889, quality["baseline"]
    assert round(float(quality["rag"]), 3) == 2.375, quality["rag"]
    assert round(float(hallucination["baseline"]), 3) == 0.303, hallucination["baseline"]
    assert round(float(hallucination["rag"]), 3) == 0.111, hallucination["rag"]


def main() -> None:
    """Run every registered test and report the tally."""
    for func in list(PASSED):
        try:
            func()
            logger.info("PASS  %s", func.__name__)
        except Exception as exc:
            FAILED.append(func.__name__)
            logger.error("FAIL  %s: %s", func.__name__, exc)
    logger.info("%d passed, %d failed", len(PASSED) - len(FAILED), len(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
