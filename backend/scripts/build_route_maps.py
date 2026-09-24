"""
Build the route maps that condition generation on the router's decision.

Two maps, both over the 52 benchmark prompts:

  predicted   the route the adopted router assigns, taken from out-of-fold
              predictions so the routing is not leaked
  oracle      the audited true route, giving the ceiling that perfect routing
              would reach
  union       every prompt on the union block, for the control arm

The control arm gets a map too, even though every row of it is identical. It
could have been a flag, but a map keeps every arm selected the same way and
makes the control reproducible from a tracked file rather than from an argument
someone has to remember. That control answers the objection a
supervisor actually raises, which is not "what if the route were wrong" but
"why not put all seven blocks in the system prompt and skip the router". If
selecting the guidance beats saying everything, the router has earned its place.

A deliberately-wrong-route control was considered and rejected. It conflates
three things at once. It assigns safety_deflect to ordinary factual questions,
so the model refuses and quality collapses for a reason that has nothing to do
with routing. And it is asymmetric in the direction that breaks the test: the
predicted arm sends 36 of 51 prompts to general_knowledge, the lightest block,
while a wrong route hands those same prompts obligation-heavy text that names
the very behaviours the follow-up and distinction detectors match on. The
control could beat the real routing mechanically.

Why the predictions are read from a file rather than routed live. The benchmark
is the router's training data: 51 of the 163 prompts the adopted router is fitted
on come from `phase1_baseline_benchmark.csv`. Fitting on that and then routing
those same prompts would be total leakage. `evaluate_config` already predicts
each prompt only while it is held out, so
`data/router_eval/predictions_router_v2_audited_ext.csv` carries a leak-free
route for every benchmark prompt. This script reads those.

What that does NOT remove: the rule patterns in `src/router.py` were hand-written
against benchmark phrasings and decide roughly a tenth of turns. Cross-validation
cannot hold out a regex. That residue is recorded in the map as
`rule_leak_possible` and must be declared wherever the end-to-end result is
reported.

The adversarial map is different, and cleanly so. The twenty prompts in
`router_adversarial_v1.csv` are held out permanently (D-023) and have never been
trained on, so the adopted router can be fitted on the extended set and asked to
route them live. No out-of-fold machinery is needed and no leak remains, not even
the rule residue: the prompts were written to avoid the rule vocabulary in the
first place. The router reaches safety_deflect on 13 of the 20, and the seven it
misses are the point of the safety experiment rather than a defect in the map.

Usage:

  python scripts/build_route_maps.py
  python scripts/build_route_maps.py --seed 7
  python scripts/build_route_maps.py --adversarial
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.llm_router import UNPARSEABLE  # noqa: E402
from src.router import ROUTES, is_diagnosis_request, route_for_category  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
BENCHMARK = BASE_DIR / "data" / "benchmark" / "phase1_baseline_benchmark.csv"
PREDICTIONS = (BASE_DIR / "data" / "router_eval"
               / "predictions_router_v4_topic_ext.csv")
ADVERSARIAL = BASE_DIR / "data" / "benchmark" / "router_adversarial_v1.csv"
OUT_DIR = BASE_DIR / "data" / "route_maps"
EVAL_DIR = BASE_DIR / "data" / "router_eval"
# The configuration adopted in D-019/D-023, and the one whose 13-of-20 result on
# this set is already reported. Naming a different one here would silently change
# what the safety experiment is a test of.
ADOPTED_CONFIG = "embeddings_word"
# The control block in config/prompts.yaml. Underscore-prefixed because it is
# selected explicitly by an arm, never by the router.
UNION_BLOCK = "_union"
# Which ground-truth track the maps are built against. "audited" applied 16
# per-prompt corrections written for benchmark v3 and raises on v4, because a
# prompt id means a different question in a different revision (F-V4-007).
# v4's six categories map onto routes directly, which is the condition the audit
# was compensating for, so topic may be the right track for it permanently.
# Revisit when D-127 is resolved.
ROUTER_TRACK = "topic"

DEFAULT_SEED = 20260826


def load_predictions() -> pd.DataFrame:
    """Return the out-of-fold predictions for benchmark prompts only."""
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"No out-of-fold predictions at {PREDICTIONS}. Produce them with "
            f"python scripts/evaluate_router.py --config embeddings_word "
            f"--dataset extended --label router_v2_audited_ext"
        )
    frame = pd.read_csv(PREDICTIONS)
    # Every row, not just those whose source_file is the current benchmark. The
    # predictions were written when v3 was current, and v4 reuses many of its
    # questions; matching on text finds them wherever they came from, and a
    # prompt with no match is routed live rather than dropped.
    return frame


def build_predicted_map(benchmark: pd.DataFrame,
                        predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each benchmark prompt a route the router did not see it decide.

    Two paths, and which one a prompt takes depends on whether the router was
    ever fitted on it.

    A prompt whose text appears in the router's training data must take its route
    from the out-of-fold predictions, where it was predicted only while held out.
    Matching is on text rather than id, and that is not a nicety: the v4
    benchmark reuses prompt ids for different questions, so an id-based join
    would succeed silently and attach the wrong route to almost every prompt.

    A prompt the router has never seen is routed live, fitted on the extended
    set. That is clean by construction and needs no out-of-fold machinery. 62 of
    the 102 v4 prompts are in this position, which is a considerable improvement
    on v3, where every one of the 52 was training data.

    Both are recorded in route_source so the two populations stay separable in
    the analysis rather than being pooled into one accuracy figure.
    """
    from src.router_eval import get_config, load_labelled_prompts

    by_text = {row["prompt"].strip(): row for _, row in predictions.iterrows()}
    unseen = [row["prompt"] for _, row in benchmark.iterrows()
              if row["prompt"].strip() not in by_text]

    router = None
    if unseen:
        training = load_labelled_prompts(taxonomy=ROUTER_TRACK, dataset="extended")
        leaked = set(p.strip() for p in unseen) & set(
            p.strip() for p in training["prompt"])
        if leaked:
            raise ValueError(
                f"{len(leaked)} prompt(s) are in the router's training data but "
                f"have no out-of-fold prediction, so routing them live would "
                f"leak: {sorted(leaked)[:3]}. Regenerate the predictions file: "
                f"python scripts/evaluate_router.py --config {ADOPTED_CONFIG} "
                f"--dataset extended --label router_v3_audited_ext"
            )
        logger.info("Routing %d prompt(s) live; the router has never seen them",
                    len(unseen))
        router = get_config(ADOPTED_CONFIG).build().fit(
            training["prompt"].tolist(), training["route"].tolist())

    rows = []
    for _, prompt in benchmark.iterrows():
        match = by_text.get(prompt["prompt"].strip())
        if match is not None:
            reused = match["prompt_id"] != prompt["prompt_id"]
            route = match["modal_prediction"]
            true_route = match["route"]
            source = (f"out_of_fold:{match['prompt_id']}" if reused
                      else "out_of_fold")
            spread = int(match["prediction_spread"])
        else:
            route, _stage = router.route(prompt["prompt"])
            true_route = route_for_category(prompt["category"])
            if true_route is None:
                raise ValueError(
                    f"{prompt['prompt_id']} has category "
                    f"{prompt['category']!r}, which maps to no route. Add it to "
                    f"the category mapping in src/router.py before building maps."
                )
            source = "live_held_out"
            # One fit, so there is nothing to be unanimous across. Recorded as 1
            # rather than left blank, and separable by route_source.
            spread = 1
        rows.append({
            "prompt_id": prompt["prompt_id"],
            "category": prompt["category"],
            "route": route,
            "true_route": true_route,
            "route_source": source,
            "prediction_spread": spread,
            "correct": int(route == true_route),
            "rule_leak_possible": int(is_diagnosis_request(prompt["prompt"])),
        })
    return pd.DataFrame(rows)


def build_oracle_map(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each prompt its audited true route, for the ceiling arm.

    Nearly free to run: the predicted map already agrees with the truth on most
    prompts, so the oracle arm differs from the predicted arm on only a handful
    and the gap between them prices the router's errors directly.
    """
    oracle = predicted.copy()
    oracle["route"] = oracle["true_route"]
    oracle["route_source"] = "oracle_audited"
    oracle["correct"] = 1
    return oracle


def describe(name: str, frame: pd.DataFrame, ground_truth: str = "") -> None:
    """
    Log what a map contains, so the dilution is visible before any GPU time.

    ground_truth names what "correct" was measured against, and it is not
    optional in spirit: the wording used to hardcode "the audited labels",
    which was accurate while PREDICTIONS came from an audited-track file and
    became silently wrong the moment it was switched to the topic track for
    D-127. A hardcoded label here is exactly the kind of number that gets
    copied into a note or the report without anyone re-deriving it, so the
    caller is made to say which ground truth it means.
    """
    logger.info("")
    logger.info("%s: %d prompts", name, len(frame))
    counts = frame["route"].value_counts()
    for route in ROUTES:
        count = int(counts.get(route, 0))
        share = count / len(frame)
        marker = "  <- dominant" if share > 0.5 else ""
        logger.info("  %-28s %3d  %5.1f%%%s", route, count, share * 100, marker)
    if "correct" in frame:
        assert ground_truth, "describe() needs ground_truth when a correct column is present"
        logger.info("  routing accuracy against %s: %.1f%%",
                    ground_truth, frame["correct"].mean() * 100)


def build_llm_map(label: str, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Turn variant B's per-prompt classifications into a route map.

    The route is the modal prediction across the passes, matching how the fitted
    router's map is built, so a single unlucky sampling draw does not decide an
    arm.

    A prompt the model never managed to label gets an **empty** route, not a
    default one. run_benchmark reads that as "generate unrouted". Falling back
    to general_knowledge, or to any other route, would attribute that route's
    guidance to a turn nothing classified, and the arm would partly be measuring
    the fallback rather than the router. The count is logged, because an
    unroutable share is a real property of this approach.
    """
    path = EVAL_DIR / f"llm_per_prompt_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No LLM routing output at {path}. Run it first: "
            f"python scripts/evaluate_llm_router.py --model mistral-7b "
            f"--label {label}"
        )
    per_prompt = pd.read_csv(path)

    # Aggregated per classified prompt_id first, then indexed by prompt TEXT.
    # load_labelled_prompts deduplicates on text before router B ever sees the
    # data, and v4 repeats several questions under different ids (F-V4-006):
    # "What causes autism?" is P013, P029, P050 and P051. Only one of those ids
    # was actually classified; matching on id left the other three with no row
    # at all and raised. Text matching is what build_predicted_map already does
    # for variant A, for the identical reason.
    aggregated = (per_prompt.groupby("prompt_id")
                  .agg(prompt=("prompt", "first"),
                       modal_prediction=("predicted", lambda s: s.mode().iloc[0]),
                       prediction_spread=("predicted", "nunique"))
                  .reset_index())
    by_text = {row["prompt"].strip(): row for _, row in aggregated.iterrows()}

    rows = []
    for _, prompt in benchmark.iterrows():
        match = by_text.get(prompt["prompt"].strip())
        if match is None:
            raise ValueError(
                f"{prompt['prompt_id']} has no LLM classification and no "
                f"identically worded prompt that does. {path.name} was "
                f"produced from a different prompt set; re-run the LLM router "
                f"against this benchmark."
            )
        route = match["modal_prediction"]
        decided = route != UNPARSEABLE
        true_route = route_for_category(prompt["category"])
        reused = match["prompt_id"] != prompt["prompt_id"]
        rows.append({
            "prompt_id": prompt["prompt_id"],
            "category": prompt["category"],
            "route": route if decided else "",
            "true_route": true_route,
            "route_source": (f"llm:{label}:{match['prompt_id']}" if reused
                             else f"llm:{label}") if decided else "llm_unparseable",
            "prediction_spread": int(match["prediction_spread"]),
            "correct": int(decided and route == true_route),
            "rule_leak_possible": 0,
        })
    return pd.DataFrame(rows)


def build_union_map(predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Assign every prompt the union block, giving the control arm its map.

    The control answers the objection a supervisor actually raises, which is not
    "what if the route were wrong" but "why not put all seven blocks in the
    system prompt and skip the router". Every prompt therefore receives every
    route's obligations at once, and if selecting the guidance beats saying
    everything, the router has earned its place.

    It needs a map of its own rather than reusing predicted.csv: passing the
    predicted map with a different --arm would replay the predicted routes and
    produce a duplicate of the routed arm under another name, and the control
    would be worthless in a way nothing downstream could detect.
    """
    union = predicted.copy()
    union["route"] = UNION_BLOCK
    union["route_source"] = "union_control"
    # The union arm is not a routing decision, so routing accuracy is undefined
    # for it. Zeroed rather than left as the predicted arm's values, which would
    # invite someone to average them.
    union["correct"] = 0
    union["prediction_spread"] = 1
    union["rule_leak_possible"] = 0
    return union


def build_adversarial_map() -> pd.DataFrame:
    """
    Route the twenty held-out adversarial prompts with the adopted router.

    Fitted on the extended set and asked to route prompts it has never seen, so
    the assignment is live rather than replayed. Records which stage decided each
    turn, because whether the rule layer or the learned gate caught a turn is the
    distinction the safety experiment is built to examine.
    """
    from src.router import ROUTE_SAFETY, is_diagnosis_request
    from src.router_eval import get_config, load_labelled_prompts

    adversarial = pd.read_csv(ADVERSARIAL, dtype=str, keep_default_na=False)
    training = load_labelled_prompts(taxonomy=ROUTER_TRACK, dataset="extended")

    overlap = set(adversarial["prompt"]) & set(training["prompt"])
    if overlap:
        raise ValueError(
            f"The adversarial set must stay held out, but {len(overlap)} of its "
            f"prompts appear in training: {sorted(overlap)[:3]}"
        )

    router = get_config(ADOPTED_CONFIG).build().fit(
        training["prompt"].tolist(), training["route"].tolist())

    rows = []
    for _, row in adversarial.iterrows():
        route, decided_by = router.route(row["prompt"])
        rows.append({
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "route": route,
            "true_route": ROUTE_SAFETY,
            "route_source": f"live_{ADOPTED_CONFIG}_extended",
            "prediction_spread": 1,
            "correct": int(route == ROUTE_SAFETY),
            "decided_by": decided_by,
            "attack_family": row["attack_family"],
            # Held out permanently, so nothing here can have leaked. Kept so the
            # column set matches the benchmark maps and one loader reads both.
            "rule_leak_possible": 0,
            "rule_would_fire": int(is_diagnosis_request(row["prompt"])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Write both route maps and report what they contain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Recorded for provenance; no sampling is done here")
    parser.add_argument("--llm", metavar="LABEL",
                        help="Build the LLM router's map from an "
                             "evaluate_llm_router.py run, e.g. "
                             "llm_mistral-7b_zero_shot")
    parser.add_argument("--adversarial", action="store_true",
                        help="Build the held-out adversarial map instead. Routed "
                             "live, because those prompts are never trained on.")
    args = parser.parse_args()

    if args.llm:
        benchmark = pd.read_csv(BENCHMARK)
        frame = build_llm_map(args.llm, benchmark)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "llm_predicted.csv"
        frame.to_csv(out, index=False)
        ground_truth = (f"category-derived {ROUTER_TRACK} labels "
                        f"(no audit exists for v4, D-127)")
        describe("llm_predicted", frame[frame["route"] != ""], ground_truth)

        undecided = int((frame["route"] == "").sum())
        logger.info("")
        logger.info("Unroutable: %d of %d prompts produced no usable label and "
                    "will generate unrouted", undecided, len(frame))
        # describe() was handed only the prompts that parsed, so the accuracy it
        # logged is over those. That is the generous reading and it is not the
        # one this project reports: src/llm_router.py scores an unparseable
        # output as incorrect, because a router that fails to decide would send
        # a caregiver's turn nowhere. Both are logged, each said out loud, so
        # the number copied into a note is the right one.
        logger.info("  routing accuracy over all %d prompts, scoring the %d "
                    "unroutable as incorrect: %.1f%%  <- the reportable figure",
                    len(frame), undecided, frame["correct"].mean() * 100)
        logger.info("  the %.1f%% above is over the %d that parsed only, and "
                    "excludes this router's most likely failure mode",
                    frame[frame["route"] != ""]["correct"].mean() * 100,
                    len(frame) - undecided)

        fitted_path = OUT_DIR / "predicted.csv"
        if fitted_path.exists():
            fitted = pd.read_csv(fitted_path).set_index("prompt_id")["route"]
            joined = frame.set_index("prompt_id")["route"]
            shared = joined.index.intersection(fitted.index)
            agree = int((joined[shared] == fitted[shared]).sum())
            logger.info("")
            logger.info("Agreement with the fitted router: %d of %d prompts",
                        agree, len(shared))
            logger.info("They differ on %d, which is the number of extra "
                        "responses an LLM-routed arm actually needs",
                        len(shared) - agree)
        logger.info("Wrote %s", out)
        return

    if args.adversarial:
        frame = build_adversarial_map()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT_DIR / "adversarial_predicted.csv", index=False)
        describe("adversarial_predicted", frame,
                "safety_deflect by construction, not a category label")
        caught = int(frame["correct"].sum())
        by_rule = int(frame["rule_would_fire"].sum())
        logger.info("")
        logger.info("Reached the safety route: %d of %d. The rule layer alone "
                    "would catch %d, so the learned gate accounts for %d",
                    caught, len(frame), by_rule, caught - by_rule)
        logger.info("The %d it misses are routed elsewhere and still meet the "
                    "system prompt's refusal rules. Whether that is enough is "
                    "the question the safety run answers",
                    len(frame) - caught)
        logger.info("Wrote %s", OUT_DIR / "adversarial_predicted.csv")
        return

    benchmark = pd.read_csv(BENCHMARK)
    predictions = load_predictions()
    logger.info("Benchmark %d prompts, out-of-fold predictions %d",
                len(benchmark), len(predictions))

    predicted = build_predicted_map(benchmark, predictions)
    oracle = build_oracle_map(predicted)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    union = build_union_map(predicted)
    predicted.to_csv(OUT_DIR / "predicted.csv", index=False)
    oracle.to_csv(OUT_DIR / "oracle.csv", index=False)
    union.to_csv(OUT_DIR / "union.csv", index=False)

    describe("predicted", predicted,
             f"category-derived {ROUTER_TRACK} labels (no audit exists for v4, D-127)")
    describe("oracle", oracle,
             f"category-derived {ROUTER_TRACK} labels (no audit exists for v4, D-127)")
    logger.info("")
    logger.info("union: %d prompts, all on %s", len(union), UNION_BLOCK)

    differ = int((predicted["route"] != oracle["route"]).sum())
    logger.info("")
    logger.info("Predicted and oracle differ on %d of %d prompts, so the oracle "
                "arm costs %d extra responses per run",
                differ, len(predicted), differ)

    unanimous = int((predicted["prediction_spread"] == 1).sum())
    leaky = int(predicted["rule_leak_possible"].sum())
    logger.info("")
    logger.info("Unanimous across shuffles: %d of %d", unanimous, len(predicted))
    logger.info("Decided by a rule pattern, so carrying the residual leak: %d",
                leaky)
    reused = int(predicted["route_source"].str.contains(":").sum())
    if reused:
        logger.info("Predictions reused from an identically worded prompt: %d",
                    reused)
    logger.info("")
    logger.info("Wrote %s and oracle.csv", OUT_DIR / "predicted.csv")


if __name__ == "__main__":
    main()
