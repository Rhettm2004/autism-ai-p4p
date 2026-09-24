"""
Tests for the LLM router, variant B.

Three things carry the weight, and none of them need a GPU.

**Leakage.** The classification prompt must not hand the model the answer key.
`route_for_category` maps benchmark category labels onto routes, so a route
definition echoing a category label would let the model pattern-match the label
instead of reading the message, and the measurement would be of
`config/prompts.yaml` rather than of the model. The few-shot examples must
likewise not be benchmark prompts. Both are asserted against the live benchmark
and the live category mapping, so they cannot drift apart.

**The unparseable outcome.** A classifier emitting free text can emit something
that is not a label. That has to be recorded and scored as incorrect, never
mapped to a default and never dropped. Dropping would score the router only on
the turns it managed to answer, flattering exactly the failure mode this variant
is most likely to have.

**The head-to-head arithmetic.** McNemar on discordant pairs is the comparison
the report will quote, so it is checked against a hand-worked table rather than
trusted.

Run standalone, no pytest:

  python tests/test_llm_router.py
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.llm_router import (  # noqa: E402
    UNPARSEABLE,
    VARIANTS,
    LLMRouter,
    load_router_prompt,
    parse_route,
    routes_named,
)
from src.llm_router import truncated_label as elr_truncated  # noqa: E402
from src.router import ROUTE_SAFETY, ROUTES  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.llm_router")

BASE_DIR = Path(__file__).parent.parent
BENCHMARK = BASE_DIR / "data" / "benchmark" / "phase1_baseline_benchmark.csv"
PROMPTS_PATH = BASE_DIR / "config" / "prompts.yaml"

_spec = importlib.util.spec_from_file_location(
    "evaluate_llm_router", BASE_DIR / "scripts" / "evaluate_llm_router.py")
elr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elr)

PASSED = []
FAILED = []


def _classifications(rows: list[tuple[str, list[str]]],
                     texts: dict[str, str] | None = None) -> pd.DataFrame:
    """Build an llm_per_prompt frame from (prompt_id, predictions per pass)."""
    texts = texts or {}
    out = []
    for prompt_id, predictions in rows:
        for repeat, predicted in enumerate(predictions):
            out.append({"repeat": repeat, "seed": 1, "prompt_id": prompt_id,
                        "prompt": texts.get(prompt_id, f"q {prompt_id}"),
                        "source_file": "b.csv",
                        "true_route": "referral", "predicted": predicted,
                        "decided_by": "llm",
                        "unparseable": int(predicted == UNPARSEABLE),
                        "correct": 0})
    return pd.DataFrame(out)


def _build_map(classifications: pd.DataFrame, categories: dict[str, str],
               texts: dict[str, str] | None = None) -> pd.DataFrame:
    """Run build_llm_map against a temporary classification file."""
    import importlib.util
    import shutil
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "build_route_maps", BASE_DIR / "scripts" / "build_route_maps.py")
    brm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(brm)

    directory = Path(tempfile.mkdtemp(prefix="llm_map_"))
    original = brm.EVAL_DIR
    try:
        brm.EVAL_DIR = directory
        classifications.to_csv(directory / "llm_per_prompt_t.csv", index=False)
        texts = texts or {}
        benchmark = pd.DataFrame([
            {"prompt_id": pid, "category": category,
             "prompt": texts.get(pid, f"q {pid}")}
            for pid, category in categories.items()])
        return brm.build_llm_map("t", benchmark)
    finally:
        brm.EVAL_DIR = original
        shutil.rmtree(directory, ignore_errors=True)


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _pass_frame(rows: list[tuple[str, str]], repeat: int = 0) -> pd.DataFrame:
    """Build one routing pass from (true_route, predicted) pairs."""
    return pd.DataFrame([{
        "repeat": repeat, "seed": 1, "prompt_id": f"P{i:03d}",
        "prompt": f"question {i}", "source_file": "b.csv",
        "true_route": true, "predicted": pred,
        "decided_by": "llm", "unparseable": int(pred == UNPARSEABLE),
        "correct": int(true == pred),
    } for i, (true, pred) in enumerate(rows, start=1)])


# --------------------------------------------------------------------------
# Leakage: the prompt must not contain the answer key
# --------------------------------------------------------------------------

@test
def test_no_route_definition_echoes_a_benchmark_category_label():
    """
    The measurement-integrity test.

    A definition containing a category label lets the model match the label
    rather than read the message, and the result would be about this file. Route
    names that happen to equal a category name are exempt: those are the labels
    the model is asked to emit and must appear.
    """
    benchmark = pd.read_csv(BENCHMARK)
    categories = {str(c).strip().lower() for c in benchmark["category"].unique()}
    leaks = []
    for variant in VARIANTS:
        system, _ = load_router_prompt(variant)
        lowered = system.lower()
        for category in categories:
            if category in ROUTES:
                continue
            if category in lowered:
                leaks.append(f"{variant}: {category!r}")
    assert not leaks, "the classification prompt leaks category labels: " + \
        "; ".join(leaks)


@test
def test_no_few_shot_example_is_a_benchmark_prompt():
    """An example lifted from the benchmark is a free correct answer."""
    benchmark = pd.read_csv(BENCHMARK)
    texts = {" ".join(str(p).split()).lower() for p in benchmark["prompt"]}
    system, _ = load_router_prompt("few_shot")
    leaks = [line for line in system.splitlines()
             if line.strip().lower().startswith("message:")
             and " ".join(line.split(":", 1)[1].split()).lower() in texts]
    assert not leaks, f"few-shot examples appear in the benchmark: {leaks}"


@test
def test_the_prompt_does_not_bias_towards_the_safety_route():
    """
    No thumb on the scale.

    Telling the model to prefer safety when uncertain would lift safety recall
    for a reason unrelated to being an LLM, and variant A carries no such
    instruction. A safety-biased router is a separate named variant.
    """
    for variant in VARIANTS:
        system, _ = load_router_prompt(variant)
        lowered = system.lower()
        for phrase in ("when in doubt", "if unsure", "err on the side",
                       "prefer safety", "default to safety"):
            assert phrase not in lowered, \
                f"{variant} biases the classifier: {phrase!r}"


# --------------------------------------------------------------------------
# The prompt itself
# --------------------------------------------------------------------------

@test
def test_every_route_is_defined_in_the_prompt():
    """A route the model is never told about can never be chosen."""
    for variant in VARIANTS:
        system, _ = load_router_prompt(variant)
        missing = [route for route in ROUTES if route not in system]
        assert not missing, f"{variant} defines no label for: {missing}"


@test
def test_the_few_shot_variant_shares_the_zero_shot_definitions():
    """
    Few-shot is zero-shot plus examples, substituted rather than duplicated.

    Duplicating the definitions would let the two drift, and a difference in the
    definitions would be indistinguishable from a difference caused by the
    examples, which is the only thing the comparison is meant to isolate.
    """
    zero, _ = load_router_prompt("zero_shot")
    few, _ = load_router_prompt("few_shot")
    assert "__ZERO_SHOT_BODY__" not in few, "the placeholder was not substituted"
    assert zero.strip() in few, "few-shot does not contain the zero-shot body"
    assert len(few) > len(zero), "few-shot adds nothing"


@test
def test_the_user_template_carries_the_message():
    """A template without the placeholder would route the same empty string."""
    for variant in VARIANTS:
        _, template = load_router_prompt(variant)
        assert "{prompt}" in template, f"{variant} template has no {{prompt}}"
        assert "hello" in template.format(prompt="hello")


@test
def test_the_classification_prompt_is_frozen_with_a_date():
    """The freeze date is the evidence the prompt predates the results."""
    config = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    meta = config["llm_router"]["_meta"]
    assert meta.get("frozen_on"), "llm_router._meta.frozen_on is missing"
    assert isinstance(meta.get("version"), int), "version must be an int"


@test
def test_an_unknown_variant_raises_and_names_the_valid_ones():
    """Silently falling back would hide a typo as a null result."""
    try:
        load_router_prompt("zeroshot")
    except ValueError as exc:
        assert "zero_shot" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("expected a ValueError for an unknown variant")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

@test
def test_a_bare_label_parses():
    """The output the prompt actually asks for."""
    for route in ROUTES:
        assert parse_route(route) == route


@test
def test_wrapping_is_forgiven():
    """Punctuation and an echoed prefix do not change which route was named."""
    assert parse_route("  Label: referral.  ") == "referral"
    assert parse_route('"general_knowledge"') == "general_knowledge"
    assert parse_route("The label is caregiver_support") == "caregiver_support"


@test
def test_naming_two_routes_is_unparseable():
    """A model that listed two did not choose one, and must not be resolved for."""
    assert parse_route("referral or general_knowledge") == UNPARSEABLE
    assert parse_route("either safety_deflect or result_explanation") == UNPARSEABLE


@test
def test_naming_no_route_is_unparseable():
    """Refusals, empty output and prose all fail the same way."""
    for text in ("", "   ", "I cannot classify this", "none of the above", "42"):
        assert parse_route(text) == UNPARSEABLE, text


@test
def test_unparseable_is_never_a_real_route():
    """It must not collide with a label, or it would be scored as a decision."""
    assert UNPARSEABLE not in ROUTES


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@test
def test_unparseable_counts_as_incorrect_and_is_not_dropped():
    """
    The failure mode this variant is most likely to have must not be excused.

    Two of four correct with one unparseable is 50% accuracy, not 66%.
    """
    frame = _pass_frame([("referral", "referral"),
                         ("general_knowledge", "general_knowledge"),
                         ("safety_deflect", UNPARSEABLE),
                         ("referral", "general_knowledge")])
    metrics = elr.metrics_for_pass(frame)
    assert metrics["accuracy"] == 0.5, metrics["accuracy"]
    assert metrics["unparseable_rate"] == 0.25, metrics["unparseable_rate"]


@test
def test_an_unparseable_safety_turn_is_a_missed_safety_turn():
    """
    The most expensive failure the router can have.

    A safety turn the model failed to label is not routed to safety, so safety
    recall must fall. Scoring it any other way would hide a miss.
    """
    frame = _pass_frame([("safety_deflect", UNPARSEABLE),
                         ("safety_deflect", "safety_deflect")])
    assert elr.metrics_for_pass(frame)["safety_recall"] == 0.5


@test
def test_the_result_records_that_no_cross_validation_happened():
    """
    folds=0 is a statement, not a missing value.

    Variant B is never fitted, so a single pass is already leak-free and there
    were no folds. Recording a fold count copied from variant A would suggest
    the two were evaluated the same way.
    """
    frame = pd.concat([_pass_frame([("referral", "referral")], repeat=r)
                       for r in range(3)], ignore_index=True)
    result = elr.build_result(frame, "llm_test", "topic", "extended", 3, 1)
    assert result.folds == 0
    assert result.repeats == 3
    assert result.n_prompts == 1
    assert len(result.per_repeat) == 3


# --------------------------------------------------------------------------
# The head-to-head
# --------------------------------------------------------------------------

@test
def test_mcnemar_uses_only_the_discordant_pairs():
    """
    Checked against a hand-worked table.

    Four prompts: both right, both wrong, fitted-only, LLM-only. The concordant
    pairs carry no information about which classifier is better, so the
    statistic rests on the two discordant ones.
    """
    import tempfile

    fitted = pd.DataFrame([
        {"prompt_id": "P001", "route": "referral", "modal_prediction": "referral"},
        {"prompt_id": "P002", "route": "referral", "modal_prediction": "general_knowledge"},
        {"prompt_id": "P003", "route": "referral", "modal_prediction": "referral"},
        {"prompt_id": "P004", "route": "referral", "modal_prediction": "general_knowledge"},
    ])
    llm = _pass_frame([("referral", "referral"),          # P001 both correct
                       ("referral", "general_knowledge"),  # P002 neither
                       ("referral", "general_knowledge"),  # P003 fitted only
                       ("referral", "referral")])          # P004 LLM only

    directory = Path(tempfile.mkdtemp(prefix="llm_router_"))
    try:
        path = directory / "predictions_x.csv"
        fitted.to_csv(path, index=False)
        out = elr.head_to_head(llm, path)
        assert out["n_paired"] == 4, out
        assert out["both_correct"] == 1 and out["neither_correct"] == 1, out
        assert out["fitted_only"] == 1 and out["llm_only"] == 1, out
        assert out["discordant"] == 2, out
        assert out["mcnemar_p"] == 1.0, out["mcnemar_p"]
    finally:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


@test
def test_pairing_against_a_different_benchmark_revision_is_refused():
    """
    D-124 applies here too.

    If the fitted predictions came from another benchmark revision, no prompt id
    would match. Returning an empty comparison would report a tie.
    """
    import shutil
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="llm_router_"))
    try:
        path = directory / "predictions_old.csv"
        pd.DataFrame([{"prompt_id": "Z999", "route": "referral",
                       "modal_prediction": "referral"}]).to_csv(path, index=False)
        try:
            elr.head_to_head(_pass_frame([("referral", "referral")]), path)
        except ValueError as exc:
            assert "D-124" in str(exc) or "shared" in str(exc), str(exc)
        else:
            raise AssertionError("compared two disjoint prompt sets")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------

@test
def test_fit_ignores_the_training_data():
    """
    Not fitted is the property that makes one pass leak-free.

    If fit() ever stored the prompts, a future change could start using them for
    example selection and the leak-free claim would quietly stop being true.
    """
    router = LLMRouter.__new__(LLMRouter)
    router.model = router.tokenizer = router.config = None
    returned = LLMRouter.fit(router, ["a", "b"], [ROUTE_SAFETY, "referral"])
    assert returned is router
    stored = [v for k, v in vars(router).items()
              if isinstance(v, (list, tuple)) and len(v) == 2]
    assert not stored, f"fit() retained training data: {stored}"


@test
def test_the_rule_layer_short_circuits_the_model_when_enabled():
    """
    A rule hit must not reach the model, and must be labelled as a rule.

    The model is None here, so anything reaching it raises rather than silently
    passing the test.
    """
    router = LLMRouter.__new__(LLMRouter)
    router.model = router.tokenizer = router.config = None
    router.use_rules = True
    route, decided_by = LLMRouter.route(router, "Does my child have autism?")
    assert route == ROUTE_SAFETY and decided_by == "rule", (route, decided_by)


@test
def test_an_unknown_variant_is_refused_at_construction():
    """A typo must fail when the router is built, not on the first prompt."""
    try:
        LLMRouter(None, None, {}, variant="fewshot")
    except ValueError as exc:
        assert "few_shot" in str(exc), str(exc)
    else:
        raise AssertionError("accepted an unknown variant")


# --------------------------------------------------------------------------
# The route map that carries variant B's decisions into generation
# --------------------------------------------------------------------------

@test
def test_the_map_takes_the_modal_prediction_across_passes():
    """
    One unlucky sampling draw must not decide an arm.

    Matches how variant A's map is built, so the two arms differ because the
    routers differ and not because one was built from a single draw.
    """
    frame = _classifications([
        ("P001", ["referral", "referral", "general_knowledge"]),
    ])
    built = _build_map(frame, {"P001": "next_steps_referral"})
    assert built.loc[0, "route"] == "referral", built.loc[0, "route"]
    assert built.loc[0, "prediction_spread"] == 2, built.loc[0, "prediction_spread"]


@test
def test_an_unparseable_prompt_gets_an_empty_route_not_a_default():
    """
    The load-bearing one.

    A turn the model never labelled has not been routed. Falling back to a real
    route would attribute that route's guidance to it, and the arm would partly
    be measuring the fallback. An empty route means run_benchmark generates it
    unrouted, which is what actually happened.
    """
    frame = _classifications([("P001", [UNPARSEABLE, UNPARSEABLE, "referral"])])
    built = _build_map(frame, {"P001": "next_steps_referral"})
    assert built.loc[0, "route"] == "", repr(built.loc[0, "route"])
    assert built.loc[0, "route_source"] == "llm_unparseable"
    assert built.loc[0, "correct"] == 0, "an unrouted turn is not a correct route"


@test
def test_a_duplicate_text_prompt_inherits_the_classified_ids_route():
    """
    Regression for the fault that broke the first live run.

    v4 repeats several questions under different ids (F-V4-006). load_labelled_
    prompts deduplicates on text before router B ever sees the data, so only one
    of those ids gets classified. Matching build_llm_map on prompt_id left the
    other ids with no row at all and raised mid-booking; matching on text, the
    way build_predicted_map already does for variant A, is the fix.
    """
    text = "What causes autism?"
    frame = _classifications([("P013", ["general_knowledge"] * 3)],
                             texts={"P013": text})
    # Three benchmark ids share P013's exact text; only P013 was classified.
    categories = {"P013": "general_autism_knowledge",
                  "P029": "general_autism_knowledge",
                  "P050": "general_autism_knowledge"}
    built = _build_map(frame, categories,
                       texts={"P013": text, "P029": text, "P050": text})
    for pid in ("P013", "P029", "P050"):
        row = built[built.prompt_id == pid].iloc[0]
        assert row["route"] == "general_knowledge", (pid, row["route"])
    reused = built[built.prompt_id == "P050"].iloc[0]
    assert "P013" in reused["route_source"], reused["route_source"]


@test
def test_a_prompt_missing_from_the_classification_output_raises():
    """
    Silently defaulting it would put an unmeasured route into the arm.

    This happens when the routing run and the benchmark are out of step, which
    after the v4 migration is a live possibility (D-124).
    """
    frame = _classifications([("P001", ["referral"])])
    try:
        _build_map(frame, {"P001": "next_steps_referral",
                           "P002": "general_autism_knowledge"})
    except ValueError as exc:
        assert "P002" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("accepted a benchmark prompt with no classification")



# --------------------------------------------------------------------------
# Separator tolerance, and the line between repairing the instrument and
# loosening the scoring
# --------------------------------------------------------------------------

@test
def test_label_separators_do_not_change_the_decision():
    """
    A label is the same decision however the model punctuated it.

    Underscores, spaces and hyphens, in any case, all name one route. Matching
    only the underscore form recorded a model that had decided as a model that
    had not, and every such prompt then generated unrouted.
    """
    for route in ROUTES:
        spaced = route.replace("_", " ")
        assert parse_route(spaced) == route, spaced
        assert parse_route(route.replace("_", "-")) == route
        assert parse_route(spaced.title()) == route
        assert parse_route(f"  Label: {spaced.upper()}.  ") == route


@test
def test_tolerance_cannot_manufacture_a_decision():
    """
    The repair must only recover decisions, never invent them.

    An output naming nothing stays unparseable, and an output naming two stays
    unparseable even when the two are written in different styles, which is the
    case the looser matching could plausibly have broken.
    """
    for text in ("", "   ", "I cannot classify this", "none of the above"):
        assert parse_route(text) == UNPARSEABLE, text
    assert parse_route("referral or general knowledge") == UNPARSEABLE
    assert parse_route("caregiver-support / result_explanation") == UNPARSEABLE
    assert routes_named("caregiver-support / result_explanation") == {
        "caregiver_support", "result_explanation"}


@test
def test_a_truncated_label_is_still_unparseable():
    """
    A label the model never finished emitting is not a decision it made.

    truncated_label identifies it for diagnosis, so the harness can tell a
    budget failure from a model failure, but nothing may feed that back into
    scoring.
    """
    assert parse_route("safety_defl") == UNPARSEABLE
    assert elr_truncated("safety_defl") == "safety_deflect"
    assert elr_truncated("misinformation_correc") == "misinformation_correction"
    # "re" begins both referral and result_explanation, so it identifies neither.
    assert elr_truncated("re") is None
    assert elr_truncated("general_knowledge") is None


# --------------------------------------------------------------------------
# Diagnosing an unparseable rate
# --------------------------------------------------------------------------

@test
def test_unparseable_reasons_are_separated():
    """
    One rate conflates four failures that have four different fixes.

    The conclusion drawn from a 40% unparseable rate is opposite depending on
    whether it is prose (the approach does not work) or truncation (the budget
    was wrong and the approach is untested).
    """
    frame = pd.DataFrame([
        {"unparseable": 1, "raw": ""},
        {"unparseable": 1, "raw": "referral or general_knowledge"},
        {"unparseable": 1, "raw": "caregiver_supp"},
        {"unparseable": 1, "raw": "Autism is a developmental condition that"},
        {"unparseable": 0, "raw": "referral"},
    ])
    breakdown = elr.explain_unparseable(frame).set_index("reason")["count"]
    assert breakdown["empty output"] == 1, breakdown
    assert breakdown["named several labels"] == 1, breakdown
    assert breakdown["cut off mid-label"] == 1, breakdown
    assert breakdown["named no label"] == 1, breakdown
    # The parsed row must not appear anywhere in the breakdown.
    assert int(breakdown.sum()) == 4, breakdown


@test
def test_no_unparseable_gives_an_empty_breakdown_not_a_crash():
    """A clean run is the case most likely to be left untested."""
    frame = pd.DataFrame([{"unparseable": 0, "raw": "referral"}])
    assert elr.explain_unparseable(frame).empty


@test
def test_raw_output_is_carried_into_the_per_prompt_frame():
    """
    The column that makes an unparseable rate interpretable at all.

    Pinned with a stub router rather than a GPU: what matters is that route_all
    asks for the raw text and records it, not what any real model would say.
    """
    class _StubRouter:
        max_label_tokens = 12

        def classify(self, text, seed=None):
            return "referral", "llm", "  Referral.  "

    frame = pd.DataFrame([{"prompt_id": "P001", "prompt": "where do i go",
                           "route": "referral", "source_file": "b.csv"}])
    out = elr.route_all(_StubRouter(), frame, repeat=0, seed=1)
    assert out.loc[0, "raw"] == "  Referral.  ", out.loc[0, "raw"]
    assert out.loc[0, "predicted"] == "referral"
    assert out.loc[0, "max_label_tokens"] == 12


@test
def test_route_still_returns_two_values():
    """
    classify() must not have widened the interface route() shares with
    IntentRouter and CascadeRouter; call sites depend on the two-tuple.
    """
    import inspect

    source = inspect.getsource(LLMRouter.route)
    assert "return route, decided_by" in source, source


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
