"""
Tests for route-appropriate follow-up conformance.

The measure this guards replaces a flat rate that was quietly meaningless. The
flat rate counts a caregiver asking about prevalence and a caregiver asking
where to get their child assessed as the same kind of turn, and scores a system
identically whether it advises the right ones or advises at random.

Two properties carry the whole result and are pinned here.

Discrimination must be zero when a system sprays advice uniformly and positive
only when it targets. A bug that made discrimination track the flat rate would
reproduce the old measure under a new name and would not look like a bug.

The obligations must come from the same file the generation prompts come from.
If the test fixture declared its own, the measure could drift away from what the
models were actually told and nothing would catch it.

Run standalone, no pytest:

  python tests/test_followup_conformance.py
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.router import ROUTES  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.followup_conformance")

BASE_DIR = Path(__file__).parent.parent
SCORED = BASE_DIR / "data" / "grading_20260825" / "rerun_results_scored.csv"
# F-P2-015 was measured on the v3 benchmark. Its route map is frozen here rather
# than read from data/route_maps/, which now holds v4: benchmark v4 reuses v3's
# prompt ids for different questions, so joining the v3 grading pass to the v4
# map succeeds cleanly and attaches the wrong route to nearly every prompt
# (D-124). Pinning a v3 result requires the v3 map.
V3_ORACLE = BASE_DIR / "tests" / "fixtures" / "oracle_v3_52prompt.csv"
EVAL_DIR = BASE_DIR / "data" / "generation_eval"

# scripts/ is not a package, so the module is loaded by path.
_spec = importlib.util.spec_from_file_location(
    "evaluate_followup", BASE_DIR / "scripts" / "evaluate_followup.py")
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

PASSED = []
FAILED = []


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal scored frame with the columns the measure needs."""
    out = []
    for i, row in enumerate(rows):
        out.append({"prompt_id": row["prompt_id"], "model_id": "m",
                    "condition": "zero_shot", "arm": row["arm"],
                    ef.FLAG: row["flag"], "grade_id": f"G{i:04d}"})
    return pd.DataFrame(out)


def _with_obligations(rows: list[dict], routes: dict[str, str]) -> pd.DataFrame:
    """Attach obligations using an in-memory route map written to a temp file."""
    frame = _frame(rows)
    frame["route"] = frame["prompt_id"].map(routes)
    obligations = ef.load_obligations()
    frame["obligation"] = frame["route"].map(obligations)
    frame["owed"] = frame["obligation"].isin(ef.OWED_LEVELS)
    return frame


def _rejects(call, what: str) -> None:
    """Assert a call raises ValueError."""
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"accepted {what}, which should have been refused")


# --------------------------------------------------------------------------
# Obligations come from the file the models were given
# --------------------------------------------------------------------------

@test
def test_obligations_are_declared_for_every_route():
    """A route with no obligation cannot be scored and must not be skipped."""
    obligations = ef.load_obligations()
    missing = [route for route in ROUTES if route not in obligations]
    assert not missing, f"routes with no declared follow-up obligation: {missing}"


@test
def test_only_the_three_declared_levels_are_valid():
    """A typo like 'requried' would silently move a route into the optional bucket."""
    obligations = ef.load_obligations()
    assert set(obligations.values()) <= {"required", "expected", "optional"}, \
        sorted(set(obligations.values()))


@test
def test_the_owed_set_matches_the_frozen_obligations():
    """
    Pinned against the freeze in config/prompts.yaml.

    Moving a route into or out of the owed set changes every conformance figure
    ever recorded, so it must be a deliberate act with its own measurement.
    """
    obligations = ef.load_obligations()
    owed = {name for name, level in obligations.items()
            if level in ef.OWED_LEVELS and not name.startswith("_")}
    assert owed == {"result_explanation", "referral", "safety_deflect",
                    "caregiver_support"}, sorted(owed)


# --------------------------------------------------------------------------
# Discrimination behaves as defined
# --------------------------------------------------------------------------

@test
def test_uniform_advice_scores_zero_discrimination():
    """
    The property the flat rate cannot express.

    A system that advises every turn identically has targeted nothing, however
    high its flat rate. If this ever returns non-zero the measure has collapsed
    back into the rate it replaced.
    """
    for flag in (0, 1):
        rows = [{"prompt_id": "P001", "arm": "a", "flag": flag},
                {"prompt_id": "P002", "arm": "a", "flag": flag}]
        frame = _with_obligations(rows, {"P001": "referral",
                                         "P002": "general_knowledge"})
        summary = ef.summarise(frame)
        assert summary.loc[0, "discrimination"] == 0.0, \
            f"uniform advice at flag={flag} scored {summary.loc[0, 'discrimination']}"


@test
def test_perfect_targeting_scores_full_discrimination():
    """Advice exactly where owed and nowhere else is the ceiling."""
    rows = [{"prompt_id": "P001", "arm": "a", "flag": 1},
            {"prompt_id": "P002", "arm": "a", "flag": 0}]
    frame = _with_obligations(rows, {"P001": "referral",
                                     "P002": "general_knowledge"})
    summary = ef.summarise(frame)
    assert summary.loc[0, "discrimination"] == 1.0, summary.loc[0, "discrimination"]


@test
def test_inverted_targeting_scores_negative():
    """
    Advising only the turns that owed nothing is worse than advising uniformly,
    and the sign has to say so.
    """
    rows = [{"prompt_id": "P001", "arm": "a", "flag": 0},
            {"prompt_id": "P002", "arm": "a", "flag": 1}]
    frame = _with_obligations(rows, {"P001": "referral",
                                     "P002": "general_knowledge"})
    summary = ef.summarise(frame)
    assert summary.loc[0, "discrimination"] == -1.0, summary.loc[0, "discrimination"]


@test
def test_discrimination_is_independent_of_the_flat_rate():
    """
    Two arms with an identical flat rate must be separable.

    This is the whole reason the measure exists, and it is exactly what the
    August re-run turned out to contain.
    """
    rows = [{"prompt_id": "P001", "arm": "targeted", "flag": 1},
            {"prompt_id": "P002", "arm": "targeted", "flag": 0},
            {"prompt_id": "P001", "arm": "uniform", "flag": 0},
            {"prompt_id": "P002", "arm": "uniform", "flag": 1}]
    frame = _with_obligations(rows, {"P001": "referral",
                                     "P002": "general_knowledge"})
    summary = ef.summarise(frame).set_index("arm")
    assert summary.loc["targeted", "flat_rate"] == summary.loc["uniform", "flat_rate"]
    assert summary.loc["targeted", "discrimination"] > \
        summary.loc["uniform", "discrimination"]


# --------------------------------------------------------------------------
# The joins refuse to lose rows
# --------------------------------------------------------------------------

@test
def test_a_scored_file_without_arms_is_refused():
    """Every retrieval arm carries rag_enabled=1, so arm is the only separator."""
    frame = _frame([{"prompt_id": "P001", "arm": "a", "flag": 1}]).drop(columns=["arm"])
    _rejects(lambda: ef.attach_obligations(frame, ef.DEFAULT_ROUTE_MAP, "true_route"),
             "a scored file with no arm column")


@test
def test_a_scored_file_without_the_flag_is_refused():
    """There is nothing to score conformance on."""
    frame = _frame([{"prompt_id": "P001", "arm": "a", "flag": 1}]).drop(columns=[ef.FLAG])
    _rejects(lambda: ef.attach_obligations(frame, ef.DEFAULT_ROUTE_MAP, "true_route"),
             "a scored file with no follow-up flag")


@test
def test_a_prompt_with_no_route_is_refused_rather_than_dropped():
    """
    A silently unjoined prompt would shrink a denominator that is already ten.

    Dropping rows is the failure mode that looks like a result.
    """
    frame = _frame([{"prompt_id": "P999", "arm": "a", "flag": 1}])
    _rejects(lambda: ef.attach_obligations(frame, ef.DEFAULT_ROUTE_MAP, "true_route"),
             "a prompt absent from the route map")


@test
def test_an_unknown_route_column_is_refused():
    """Choosing predicted versus true routes must be explicit and must exist."""
    frame = _frame([{"prompt_id": "P001", "arm": "a", "flag": 1}])
    _rejects(lambda: ef.attach_obligations(frame, ef.DEFAULT_ROUTE_MAP, "guessed"),
             "a route column that does not exist")


@test
def test_arms_that_do_not_cover_the_same_responses_cannot_be_paired():
    """An unbalanced pairing would compare different prompts and call it a delta."""
    rows = [{"prompt_id": "P001", "arm": "baseline", "flag": 1},
            {"prompt_id": "P002", "arm": "rag", "flag": 0}]
    frame = _with_obligations(rows, {"P001": "referral", "P002": "referral"})
    _rejects(lambda: ef.paired_arm_test(frame, "baseline", "rag"),
             "arms covering different prompts")


# --------------------------------------------------------------------------
# The published result
# --------------------------------------------------------------------------

@test
def test_the_published_conformance_result_is_reproducible():
    """
    Pins F-P2-015 to the artefact.

    Re-derived from the scored file rather than read from the output CSV, so a
    change in the measure surfaces here rather than in the report.
    """
    if not SCORED.exists():
        raise AssertionError(f"{SCORED} is missing")
    frame = ef.attach_obligations(pd.read_csv(SCORED), V3_ORACLE, "true_route")
    summary = ef.summarise(frame).set_index("arm")

    # Ten prompts owe a next step: 7 required, 3 expected.
    prompts = frame.drop_duplicates("prompt_id")
    assert int(prompts["owed"].sum()) == 10, int(prompts["owed"].sum())

    # The flat rate says nothing happened.
    assert abs(summary.loc["baseline", "flat_rate"] - 0.327) < 0.001
    assert abs(summary.loc["rag", "flat_rate"] - 0.332) < 0.001

    # Discrimination says the targeting collapsed.
    assert abs(summary.loc["baseline", "discrimination"] - 0.245) < 0.001, \
        summary.loc["baseline", "discrimination"]
    assert abs(summary.loc["rag", "discrimination"] - (-0.008)) < 0.001, \
        summary.loc["rag", "discrimination"]


@test
def test_the_bootstrap_is_seeded_and_reproducible():
    """An unseeded interval would move every time the report was rebuilt."""
    if not SCORED.exists():
        raise AssertionError(f"{SCORED} is missing")
    frame = ef.attach_obligations(pd.read_csv(SCORED), V3_ORACLE, "true_route")
    first = ef.bootstrap_discrimination_gap(frame, "baseline", "rag")
    second = ef.bootstrap_discrimination_gap(frame, "baseline", "rag")
    assert first == second, "the bootstrap is not reproducible"
    assert first["ci_high"] < 0, (
        f"the discrimination gap CI should exclude zero, got "
        f"[{first['ci_low']:.3f}, {first['ci_high']:.3f}]"
    )


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
