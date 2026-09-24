"""
Evaluate the LLM router (variant B) and compare it against the fitted one.

Variant A is the rule layer plus a fitted scikit-learn classifier, evaluated by
`scripts/evaluate_router.py` under repeated K-fold. Variant B is
`src/llm_router.py`, which asks a local instruction-tuned model for a label and
is never fitted. This script scores B and puts the two head to head.

**Why a single pass is a fair comparison to cross-validation.** Variant A has to
be cross-validated because it is fitted on the prompts it is asked to route; each
prediction is only meaningful while that prompt is held out. Variant B is never
fitted, so every prompt is held out by construction and one pass is already
leak-free. Repeats here measure *sampling* variance in the model's output, not
fold variance, and the two must not be described as though they were the same
thing.

**The head-to-head is paired per prompt.** Variant A's out-of-fold predictions
and variant B's predictions cover the same prompts, so the two are compared
prompt by prompt with McNemar's test on the discordant pairs, which is the right
test for two classifiers on one sample. The registry also records a per-repeat
summary so both variants appear in one listing, but the McNemar result is the
one to quote.

**Cost is not measured here and matters.** Variant A routes a turn on a CPU in
microseconds. Variant B needs a GPU-resident 7B model and about a second a turn.
A tie on accuracy is a win for variant A, and the report should say so.

Usage:

  python scripts/evaluate_llm_router.py --model mistral-7b --label llm_mistral_zs
  python scripts/evaluate_llm_router.py --model mistral-7b --variant few_shot \\
      --label llm_mistral_fs --compare-to llm_mistral_zs
  python scripts/evaluate_llm_router.py --model mistral-7b --label llm_mistral_zs \\
      --against router_v4_topic_ext
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from src.llm_router import (  # noqa: E402
    MAX_LABEL_TOKENS,
    UNPARSEABLE,
    VARIANTS,
    LLMRouter,
    routes_named,
    truncated_label,
)
from src.model_runner import (  # noqa: E402
    load_model_and_tokenizer,
    load_model_config,
    unload_model,
)
from src.router import ROUTE_SAFETY, ROUTES  # noqa: E402
from src.router_eval import (  # noqa: E402
    EVAL_DIR,
    EvaluationResult,
    load_labelled_prompts,
    mean_risk,
    record_experiment,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
DEFAULT_REPEATS = 3
DEFAULT_BASE_SEED = 20260827


def route_all(router: LLMRouter, frame: pd.DataFrame, repeat: int,
              seed: int) -> pd.DataFrame:
    """
    Route every prompt once and return one row per prompt.

    `raw` holds the model's decoded output before parsing, and it is the column
    that makes this file worth keeping. Without it an unparseable row says only
    that parsing failed, not whether the model emitted prose, named two labels,
    ran out of budget mid-label, or named one label the parser did not
    recognise. Those have different fixes and only one of them is the model's
    fault. The first router B run stored the parsed label alone, so answering
    that question needed a whole second booking; a CSV column is cheaper than a
    GPU hour.
    """
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        # Seed per prompt position rather than per pass, so a prompt starts from
        # the same RNG state in every repeat and differences between repeats are
        # the model's, not the sampler's starting point.
        route, decided_by, raw = router.classify(row["prompt"], seed=seed + position)
        rows.append({
            "repeat": repeat, "seed": seed,
            "prompt_id": row["prompt_id"], "prompt": row["prompt"],
            "source_file": row.get("source_file", ""),
            "true_route": row["route"], "predicted": route,
            "decided_by": decided_by,
            "unparseable": int(route == UNPARSEABLE),
            "correct": int(route == row["route"]),
            "raw": raw,
            "max_label_tokens": router.max_label_tokens,
        })
        if (position + 1) % 25 == 0:
            logger.info("    routed %d of %d", position + 1, len(frame))
    return pd.DataFrame(rows)



def explain_unparseable(per_prompt: pd.DataFrame) -> pd.DataFrame:
    """
    Break the unparseable classifications down by why they failed.

    Four outcomes, each with a different fix and a different owner:

      empty output          the model emitted nothing at all
      named several labels  it would not commit to one, which is the model's
                            failure and the one the strict ambiguity rule
                            exists to record
      cut off mid-label     it named a label and ran out of budget partway
                            through, which is the harness's failure and is
                            fixed with --max-label-tokens, not with a better
                            prompt
      named no label        it answered the message, refused, or wrote prose,
                            which is the model failing the task outright

    Reported because a single unparseable rate conflates all four, and the
    conclusion drawn from it differs completely depending on the mix. A rate
    driven by "named no label" says the approach does not work; the same rate
    driven by "cut off mid-label" says the budget was wrong and nothing about
    the approach at all.
    """
    failed = per_prompt[per_prompt["unparseable"] == 1]
    if failed.empty:
        return pd.DataFrame(columns=["reason", "count", "share_of_unparseable"])

    def reason(raw: str) -> str:
        if not str(raw).strip():
            return "empty output"
        if len(routes_named(raw)) > 1:
            return "named several labels"
        if truncated_label(raw) is not None:
            return "cut off mid-label"
        return "named no label"

    counts = failed["raw"].map(reason).value_counts()
    return pd.DataFrame({
        "reason": counts.index,
        "count": counts.to_numpy(),
        "share_of_unparseable": counts.to_numpy() / len(failed),
    })


def metrics_for_pass(passed: pd.DataFrame) -> dict:
    """
    Compute the same headline metrics variant A reports, for one pass.

    Unparseable outputs stay in as their own label rather than being dropped or
    mapped to a default. Dropping them would score the router only on the turns
    it managed to answer, which flatters exactly the failure mode this variant
    is most likely to have.
    """
    truth = passed["true_route"].to_numpy()
    predicted = passed["predicted"].to_numpy()
    safety_true = truth == ROUTE_SAFETY
    safety_pred = predicted == ROUTE_SAFETY
    return {
        "repeat": int(passed["repeat"].iloc[0]),
        "seed": int(passed["seed"].iloc[0]),
        "accuracy": float(passed["correct"].mean()),
        "macro_f1": float(f1_score(truth, predicted, labels=ROUTES,
                                   average="macro", zero_division=0)),
        "safety_recall": float((safety_true & safety_pred).sum()
                               / max(int(safety_true.sum()), 1)),
        "safety_precision": float((safety_true & safety_pred).sum()
                                  / max(int(safety_pred.sum()), 1)),
        "risk": mean_risk(truth, predicted),
        "kappa": float(cohen_kappa_score(truth, predicted, labels=ROUTES)),
        "rule_share": float((passed["decided_by"] == "rule").mean()),
        "unparseable_rate": float(passed["unparseable"].mean()),
    }


def build_result(per_prompt: pd.DataFrame, config_name: str, track: str,
                 dataset: str, repeats: int, base_seed: int) -> EvaluationResult:
    """
    Package the passes into the same shape variant A produces.

    folds is 0, which is not a missing value: it records that no cross-validation
    happened, because this router is not fitted and needed none.
    """
    per_repeat = pd.DataFrame(
        [metrics_for_pass(group) for _, group in per_prompt.groupby("repeat")])

    modal = (per_prompt.groupby("prompt_id")["predicted"]
             .agg(lambda s: s.mode().iloc[0]))
    predictions = (per_prompt.groupby("prompt_id")
                   .agg(category=("true_route", "first"),
                        prompt=("prompt", "first"),
                        route=("true_route", "first"),
                        source_file=("source_file", "first"),
                        correct_rate=("correct", "mean"),
                        prediction_spread=("predicted", "nunique"))
                   .reset_index())
    predictions["modal_prediction"] = predictions["prompt_id"].map(modal)

    truth = per_prompt["true_route"].to_numpy()
    predicted = per_prompt["predicted"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=ROUTES, zero_division=0)
    per_route = pd.DataFrame({"route": ROUTES, "support": support,
                              "precision": precision, "recall": recall,
                              "f1": f1, "f1_ci95": 0.0})

    labels = ROUTES + [UNPARSEABLE]
    confusion = pd.crosstab(pd.Categorical(truth, categories=labels),
                            pd.Categorical(predicted, categories=labels),
                            dropna=False)

    return EvaluationResult(
        config=config_name, n_prompts=int(per_prompt["prompt_id"].nunique()),
        repeats=repeats, folds=0, base_seed=base_seed, taxonomy=track,
        dataset=dataset, per_repeat=per_repeat, per_route=per_route,
        confusion=confusion, predictions=predictions)


def head_to_head(llm: pd.DataFrame, fitted_path: Path) -> dict:
    """
    Compare the two variants prompt by prompt with McNemar's test.

    McNemar rather than a two-sample test because both classifiers saw the same
    prompts: what matters is the prompts they disagree on, not the marginal
    totals. Ties carry no information about which is better and are excluded
    from the statistic, which is the point of the test.
    """
    fitted = pd.read_csv(fitted_path)
    if "modal_prediction" not in fitted.columns:
        raise ValueError(
            f"{fitted_path} has no modal_prediction column, so it is not an "
            f"out-of-fold predictions file from evaluate_router.py."
        )
    fitted = fitted.assign(
        fitted_correct=(fitted["modal_prediction"] == fitted["route"]).astype(int))

    llm_modal = (llm.groupby("prompt_id")
                 .agg(llm_correct=("correct", lambda s: int(s.mean() >= 0.5)),
                      true_route=("true_route", "first")).reset_index())
    merged = fitted.merge(llm_modal, on="prompt_id", how="inner")
    if merged.empty:
        raise ValueError(
            f"No prompt ids are shared between the LLM run and {fitted_path.name}, "
            f"so the two cannot be paired. They were probably measured on "
            f"different benchmark revisions; see D-124."
        )

    both = int(((merged.fitted_correct == 1) & (merged.llm_correct == 1)).sum())
    neither = int(((merged.fitted_correct == 0) & (merged.llm_correct == 0)).sum())
    fitted_only = int(((merged.fitted_correct == 1) & (merged.llm_correct == 0)).sum())
    llm_only = int(((merged.fitted_correct == 0) & (merged.llm_correct == 1)).sum())

    discordant = fitted_only + llm_only
    # Exact binomial on the discordant pairs: with n often under 40 here, the
    # chi-square approximation is not safe.
    p_value = (float(stats.binomtest(llm_only, discordant, 0.5).pvalue)
               if discordant else float("nan"))
    return {
        "n_paired": len(merged),
        "fitted_accuracy": float(merged.fitted_correct.mean()),
        "llm_accuracy": float(merged.llm_correct.mean()),
        "both_correct": both, "neither_correct": neither,
        "fitted_only": fitted_only, "llm_only": llm_only,
        "discordant": discordant, "mcnemar_p": p_value,
        "compared_against": fitted_path.name,
    }


def main() -> None:
    """Route every labelled prompt with an LLM, score it, and compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Model id from config/models.yaml")
    parser.add_argument("--label", required=True,
                        help="Unique name for this experiment in the registry")
    parser.add_argument("--variant", default="zero_shot", choices=list(VARIANTS),
                        help="Which frozen classification prompt to use")
    parser.add_argument("--use-rules", action="store_true",
                        help="Prefix the same rule layer variant A uses, to "
                             "isolate what the LLM adds on top of the rules")
    parser.add_argument("--track", default="topic",
                        help="Ground-truth track (default topic; audited does "
                             "not load against benchmark v4, see D-127)")
    parser.add_argument("--dataset", default="extended",
                        help="Prompt set to route (default extended)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Passes over the data (default {DEFAULT_REPEATS}). "
                             f"Measures sampling variance, not fold variance.")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                        help="Base generation seed")
    parser.add_argument("--against", metavar="LABEL",
                        help="Out-of-fold predictions label from variant A to "
                             "pair against, e.g. router_v4_topic_ext")
    parser.add_argument("--compare-to", metavar="LABEL",
                        help="An earlier LLM experiment to compare against in "
                             "the registry")
    parser.add_argument("--max-label-tokens", type=int, default=MAX_LABEL_TOKENS,
                        help=f"Generation budget for the label (default "
                             f"{MAX_LABEL_TOKENS}, the frozen value). Raising it "
                             f"does not cost a second booking: with the seed "
                             f"fixed, the first {MAX_LABEL_TOKENS} tokens of a "
                             f"longer completion are exactly what the frozen "
                             f"budget would have produced, so the headline "
                             f"result stays recoverable from the raw column.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing label in the registry")
    args = parser.parse_args()

    frame = load_labelled_prompts(taxonomy=args.track, dataset=args.dataset)
    logger.info("=" * 78)
    logger.info("LLM ROUTER: %s | model %s | variant %s | rules %s",
                args.label, args.model, args.variant,
                "on" if args.use_rules else "off")
    logger.info("%d prompts, %d pass(es). Not fitted, so every prompt is held "
                "out and one pass is already leak-free.", len(frame), args.repeats)
    if args.max_label_tokens != MAX_LABEL_TOKENS:
        logger.info("Label budget %d, not the frozen %d. This is a different "
                    "configuration and must be labelled as one; the frozen "
                    "result is recoverable by truncating the raw column.",
                    args.max_label_tokens, MAX_LABEL_TOKENS)
    logger.info("=" * 78)

    config = load_model_config(args.model)
    model, tokenizer = load_model_and_tokenizer(config)
    try:
        router = LLMRouter(model, tokenizer, config, variant=args.variant,
                           use_rules=args.use_rules,
                           max_label_tokens=args.max_label_tokens)
        passes = []
        for repeat in range(args.repeats):
            seed = args.base_seed + repeat * 10_000
            logger.info("  pass %d of %d (seed %d)", repeat + 1, args.repeats, seed)
            passes.append(route_all(router, frame, repeat, seed))
        per_prompt = pd.concat(passes, ignore_index=True)
    finally:
        unload_model(model, tokenizer)

    config_name = f"llm_{args.model}_{args.variant}" + ("_rules" if args.use_rules else "")
    result = build_result(per_prompt, config_name, args.track, args.dataset,
                          args.repeats, args.base_seed)

    logger.info("")
    for _, row in result.per_repeat.iterrows():
        logger.info("  pass %d  accuracy %.1f%%  safety recall %.1f%%  "
                    "unparseable %.1f%%", row["repeat"], row["accuracy"] * 100,
                    row["safety_recall"] * 100, row["unparseable_rate"] * 100)

    unparseable = float(per_prompt["unparseable"].mean())
    logger.info("")
    logger.info("  Unparseable outputs: %.1f%% of %d classifications. Scored as "
                "incorrect, never mapped to a default.",
                unparseable * 100, len(per_prompt))
    breakdown = explain_unparseable(per_prompt)
    for _, row in breakdown.iterrows():
        logger.info("    %-22s %4d  %5.1f%% of unparseable",
                    row["reason"], int(row["count"]),
                    row["share_of_unparseable"] * 100)
    if not breakdown.empty:
        logger.info("    'cut off mid-label' is the harness's fault and is fixed "
                    "with --max-label-tokens; the rest are the model's.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    per_prompt_path = EVAL_DIR / f"llm_per_prompt_{args.label}.csv"
    per_prompt.to_csv(per_prompt_path, index=False)
    if not breakdown.empty:
        breakdown.to_csv(EVAL_DIR / f"llm_unparseable_{args.label}.csv", index=False)

    notes = (f"LLM router, {args.model}, {args.variant}, rules "
             f"{'on' if args.use_rules else 'off'}. Not fitted; folds=0 records "
             f"that no cross-validation was needed. Unparseable rate "
             f"{unparseable:.3f}.")
    record_experiment(result, args.label, notes=notes,
                      compare_to=args.compare_to, force=args.force)

    if args.against:
        fitted_path = EVAL_DIR / f"predictions_{args.against}.csv"
        if not fitted_path.exists():
            raise FileNotFoundError(
                f"No out-of-fold predictions at {fitted_path}. Produce them "
                f"first: python scripts/evaluate_router.py --config "
                f"embeddings_word --dataset {args.dataset} --taxonomy "
                f"{args.track} --label {args.against}"
            )
        comparison = head_to_head(per_prompt, fitted_path)
        comparison["llm_label"] = args.label
        logger.info("")
        logger.info("  HEAD TO HEAD against %s, paired on %d prompts",
                    args.against, comparison["n_paired"])
        logger.info("    fitted router correct   %.1f%%",
                    comparison["fitted_accuracy"] * 100)
        logger.info("    LLM router correct      %.1f%%",
                    comparison["llm_accuracy"] * 100)
        logger.info("    both %d | neither %d | fitted only %d | LLM only %d",
                    comparison["both_correct"], comparison["neither_correct"],
                    comparison["fitted_only"], comparison["llm_only"])
        logger.info("    McNemar exact p = %.4g on %d discordant pairs",
                    comparison["mcnemar_p"], comparison["discordant"])
        out = EVAL_DIR / f"llm_vs_fitted_{args.label}.csv"
        pd.DataFrame([comparison]).to_csv(out, index=False)
        logger.info("  Wrote %s", out.name)

    logger.info("")
    logger.info("Wrote %s and recorded '%s' in the registry",
                per_prompt_path.name, args.label)


if __name__ == "__main__":
    main()
