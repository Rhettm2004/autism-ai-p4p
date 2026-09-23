"""
Tests for route-conditioned prompting.

The load-bearing test here is the detector-collision one. The rubric measures
professional follow-up and the screening/diagnosis distinction with regular
expressions over the response. If a guidance block contains a phrase those
patterns match, the model repeats it, the detector fires, and the measurement
stops being about the model and becomes about config/prompts.yaml. That failure
would look like a result, which is what makes it dangerous.

Run standalone, no pytest:

  python tests/test_route_prompts.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from src.model_runner import build_grounded_system_prompt, load_route_guidance  # noqa: E402
from src.router import ROUTES  # noqa: E402
from src.rubric import DISTINCTION_PATTERNS, FOLLOWUP_PATTERNS  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.route_prompts")

PROMPTS_YAML = Path(__file__).parent.parent / "config" / "prompts.yaml"
FOLLOWUP_LEVELS = {"required", "expected", "optional"}
WORD_SPREAD_TOLERANCE = 0.15

PASSED = []
FAILED = []


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _routes() -> dict:
    """The routes block from prompts.yaml, minus the metadata entry."""
    config = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    return {k: v for k, v in config.get("routes", {}).items() if k != "_meta"}


def _real_routes() -> dict:
    """Only the seven real routes, excluding the control-arm blocks."""
    return {k: v for k, v in _routes().items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# The measurement-integrity test
# --------------------------------------------------------------------------

@test
def test_no_guidance_block_contains_a_rubric_detector_phrase():
    """
    Guidance must never contain a phrase the rubric's regexes match.

    Otherwise the follow-up and distinction rates measure this file rather than
    the model. Note DISTINCTION_PATTERNS matches any "screen..." within eighty
    characters of any "diagnos...", so the two ideas cannot share a sentence.
    """
    collisions = []
    for name, block in _routes().items():
        text = block["guidance"].lower()
        for label, patterns in (("followup", FOLLOWUP_PATTERNS),
                                ("distinction", DISTINCTION_PATTERNS)):
            for pattern in patterns:
                found = re.search(pattern, text)
                if found:
                    collisions.append(f"{name} [{label}] matched {found.group(0)!r}")
    assert not collisions, "guidance trips rubric detectors: " + "; ".join(collisions)


# --------------------------------------------------------------------------
# Coverage and shape
# --------------------------------------------------------------------------

@test
def test_every_route_has_guidance():
    """A route without guidance silently degrades its arm to plain retrieval."""
    missing = [route for route in ROUTES if route not in _routes()]
    assert not missing, f"routes with no guidance: {missing}"


@test
def test_no_guidance_exists_for_a_route_that_does_not_exist():
    """A stale entry would be dead text nobody notices."""
    unexpected = [name for name in _real_routes() if name not in ROUTES]
    assert not unexpected, f"guidance for unknown routes: {unexpected}"


@test
def test_the_control_blocks_exist_and_are_not_routes():
    """The control arms are selected explicitly, never by the router."""
    routes = _routes()
    for name in ("_union", "_placebo"):
        assert name in routes, f"{name} block missing"
        assert name not in ROUTES, f"{name} must not be a real route"


@test
def test_route_guidance_blocks_are_length_matched():
    """
    Length parity, so route conditioning is not confounded with prompt length.

    The control blocks are exempt: the union block is longer by construction,
    because saying everything is what it exists to test.
    """
    counts = {name: len(block["guidance"].split())
              for name, block in _real_routes().items()}
    low, high = min(counts.values()), max(counts.values())
    spread = (high - low) / low
    assert spread <= WORD_SPREAD_TOLERANCE, (
        f"guidance lengths span {low}-{high} words, spread {spread:.0%}, "
        f"tolerance {WORD_SPREAD_TOLERANCE:.0%}: {counts}"
    )


@test
def test_every_route_declares_a_followup_obligation():
    """The route-appropriate follow-up measure scores against this field."""
    for name, block in _routes().items():
        level = block.get("followup")
        assert level in FOLLOWUP_LEVELS, \
            f"{name} has followup={level!r}, expected one of {sorted(FOLLOWUP_LEVELS)}"


@test
def test_the_routes_where_a_next_step_is_compulsory_are_the_expected_ones():
    """
    Pinned so the obligation set cannot drift after results exist.

    Freezing this before generation is what stops the route-appropriateness
    measure becoming circular.
    """
    required = {name for name, block in _real_routes().items()
                if block["followup"] == "required"}
    assert required == {"result_explanation", "referral", "safety_deflect"}, required


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------

@test
def test_omitting_the_route_leaves_the_prompt_unchanged():
    """The backwards-compatibility guarantee for every run generated so far."""
    without = build_grounded_system_prompt("RULES", "PASSAGE")
    explicit_none = build_grounded_system_prompt("RULES", "PASSAGE", None)
    assert without == explicit_none
    assert without.endswith("--- SOURCES ---\nPASSAGE\n--- END SOURCES ---")
    for block in _routes().values():
        first_line = block["guidance"].strip().splitlines()[0]
        assert first_line not in without, "guidance leaked into an unrouted prompt"


@test
def test_guidance_is_injected_above_the_sources_block():
    """Route text belongs with the instructions, not inside the evidence."""
    for route in ROUTES:
        prompt = build_grounded_system_prompt("RULES", "PASSAGE",
                                              load_route_guidance(route))
        marker = load_route_guidance(route).splitlines()[0][:40]
        assert marker in prompt, f"{route} guidance missing from the prompt"
        assert prompt.index(marker) < prompt.index("--- SOURCES ---"), \
            f"{route} guidance appears below the sources"


@test
def test_the_caregiver_restatement_survives_every_route():
    """
    D-114's restatement must not be displaced by route text.

    It is what reversed the follow-up and answers-the-question losses in
    F-P2-013, so losing it would undo a measured gain.
    """
    for route in list(ROUTES) + ["_union", "_placebo"]:
        prompt = build_grounded_system_prompt("RULES", "PASSAGE",
                                              load_route_guidance(route))
        assert "when to speak to a professional" in prompt, \
            f"{route} displaced the D-114 restatement"


@test
def test_the_safety_rules_stay_at_the_top():
    """Route guidance specialises the rules; it never precedes them."""
    for route in ROUTES:
        prompt = build_grounded_system_prompt("SAFETY_RULES_HERE", "PASSAGE",
                                              load_route_guidance(route))
        marker = load_route_guidance(route).splitlines()[0][:40]
        assert prompt.index("SAFETY_RULES_HERE") < prompt.index(marker)


@test
def test_an_unknown_route_raises_and_names_the_valid_ones():
    """Silently returning no guidance would hide a typo as a null result."""
    try:
        load_route_guidance("screening_gudiance")
    except ValueError as exc:
        assert "screening_guidance" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("expected a ValueError for an unknown route")


@test
def test_the_guidance_block_is_frozen_with_a_date():
    """The freeze date is the evidence that the rubric predates the results."""
    config = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    meta = config["routes"]["_meta"]
    assert meta.get("frozen_on"), "routes._meta.frozen_on is missing"
    assert isinstance(meta.get("version"), int), "routes._meta.version must be an int"


@test
def test_an_empty_route_builds_an_unrouted_prompt():
    """
    A router that reached no decision must not silently get a default route.

    The LLM router can emit output naming no label. run_benchmark reads an empty
    route as "generate unrouted", which is what actually happened; falling back
    to a real route would attribute that route's guidance to a turn nothing
    classified, and the arm would partly be measuring the fallback.
    """
    unrouted = build_grounded_system_prompt("RULES", "PASSAGE")
    explicit = build_grounded_system_prompt("RULES", "PASSAGE", None)
    assert unrouted == explicit
    for block in _routes().values():
        first_line = block["guidance"].strip().splitlines()[0]
        assert first_line not in unrouted, \
            "guidance reached a prompt that was never routed"


@test
def test_an_unrouted_prompt_still_carries_the_safety_rules():
    """
    The failure that would matter.

    If a safety turn were the one the router could not label, it would generate
    unrouted, and the refusal rules are the only thing standing between that
    turn and a diagnosis. They live in the condition's system prompt, not in the
    route guidance, so they survive; asserted rather than assumed.
    """
    import yaml

    conditions = yaml.safe_load(
        PROMPTS_YAML.read_text(encoding="utf-8"))["conditions"]
    for condition in conditions.values():
        prompt = build_grounded_system_prompt(condition["system_prompt"],
                                              "PASSAGE")
        assert "NEVER provide, imply, or suggest a clinical" in prompt
        assert "only a qualified clinician can diagnose" in prompt


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
