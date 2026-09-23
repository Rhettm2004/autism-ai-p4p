"""
Phase 3 intent router evaluation entry point.

Every router change is measured through this script so that the numbers reported
for one revision are comparable with the numbers reported for the next. Results
are appended to data/router_eval/experiments.csv under a label, and the running
brief in docs/PHASE3_ROUTER_BRIEF.md is regenerated from that registry.

Usage examples:

  # List the router configurations available to evaluate
  python scripts/evaluate_router.py --list

  # Evaluate a configuration and record it as a step in the progression
  python scripts/evaluate_router.py --config baseline --label baseline

  # Evaluate a change and compare it, on the same folds, against an earlier step
  python scripts/evaluate_router.py --config char_ngrams --label char_ngrams \
      --compare-to baseline --notes "Word plus character 3-5 grams"

  # Evaluate without writing anything to the registry
  python scripts/evaluate_router.py --config baseline --dry-run

  # Regenerate the running brief from the registry
  python scripts/evaluate_router.py --brief

  # Re-measure a superseded route taxonomy under the current protocol
  python scripts/evaluate_router.py --config baseline --taxonomy v1_deflect_all       --label taxonomy_v1 --notes "Pre-split taxonomy, July 2026"

Comparisons are paired. Two experiments can only be compared when they were run
with the same --repeats and --base-seed, because that is what makes repeat i of
each run use the same cross-validation folds.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router_eval import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_FOLDS,
    DEFAULT_REPEATS,
    EVAL_DIR,
    ROUTER_CONFIGS,
    EvaluationResult,
    evaluate_config,
    format_percent,
    load_labelled_prompts,
    load_registry,
    load_repeats,
    paired_comparison,
    record_experiment,
    write_brief,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def log_result(result: EvaluationResult) -> None:
    """Log the headline figures, per-route breakdown and pooled confusion matrix."""
    summary = result.summary()
    logger.info("-" * 72)
    logger.info(
        "Config '%s': %d prompts, %d repeats x %d folds (seeds %d-%d)",
        result.config, result.n_prompts, result.repeats, result.folds,
        result.base_seed, result.base_seed + result.repeats - 1,
    )
    logger.info("-" * 72)
    logger.info("  Accuracy         %s",
                format_percent(summary["accuracy_mean"], summary["accuracy_ci95"]))
    logger.info("  Macro F1         %s",
                format_percent(summary["macro_f1_mean"], summary["macro_f1_ci95"]))
    logger.info("  Safety recall    %s",
                format_percent(summary["safety_recall_mean"], summary["safety_recall_ci95"]))
    logger.info("  Safety precision %s",
                format_percent(summary["safety_precision_mean"], summary["safety_precision_ci95"]))
    logger.info("  Kappa            %s",
                format_percent(summary["kappa_mean"], summary["kappa_ci95"]))
    logger.info("  Mean risk        %.3f +/- %.3f per turn (lower is better)",
                summary["risk_mean"], summary["risk_ci95"])
    logger.info("  Decided by rules %s of turns", format_percent(summary["rule_share_mean"]))

    logger.info("")
    logger.info("  %-26s %7s %10s %8s %8s", "route", "support", "precision", "recall", "F1")
    for _, row in result.per_route.iterrows():
        logger.info(
            "  %-26s %7d %10.3f %8.3f %8.3f",
            row["route"], row["support"], row["precision"], row["recall"], row["f1"],
        )

    logger.info("")
    logger.info("  Confusion matrix pooled over repeats (rows true, cols predicted)")
    header = "".join(f"{route[:11]:>13}" for route in result.confusion.columns)
    logger.info("    %-13s%s", "", header)
    for route, row in result.confusion.iterrows():
        logger.info("    %-13s%s", route[:11], "".join(f"{v:>13}" for v in row))

    worst = result.predictions.sort_values("correct_rate").head(8)
    logger.info("")
    logger.info("  Least reliably routed prompts")
    for _, row in worst.iterrows():
        logger.info(
            "    %.0f%% correct | %s -> %s | %s",
            row["correct_rate"] * 100, row["route"], row["modal_prediction"],
            row["prompt"][:64],
        )


def log_comparison(label: str, compare_to: str, result: EvaluationResult) -> None:
    """Log the paired comparison of this run against an earlier recorded run."""
    previous = load_repeats(compare_to)
    for metric in ("accuracy", "macro_f1", "kappa", "safety_recall", "risk"):
        stats = paired_comparison(previous[metric], result.per_repeat[metric])
        significance = ""
        if stats["p"] == stats["p"]:  # not nan
            significance = " *" if stats["p"] < 0.05 else " n.s."
        before = float(previous[metric].mean())
        after = float(result.per_repeat[metric].mean())
        if metric == "risk":
            # Risk is a cost per turn, not a proportion, and lower is better.
            change = f"{before:.3f} -> {after:.3f}   delta {stats['delta']:+.3f} "                      f"(95% CI +/-{stats['delta_ci95']:.3f})"
        else:
            change = f"{format_percent(before)} -> {format_percent(after)}   "                      f"delta {stats['delta'] * 100:+.1f}pp "                      f"(95% CI +/-{stats['delta_ci95'] * 100:.1f}pp)"
        logger.info("  %-14s %s, t=%.2f, p=%.4g, dz=%.2f%s",
                    metric, change, stats["t"], stats["p"], stats["dz"], significance)


def main() -> None:
    """Parse arguments and run the requested router evaluation action."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Phase 3 intent router with repeated cross-validation.",
    )
    parser.add_argument("--config", help="router configuration to evaluate")
    parser.add_argument("--label", help="name to record this evaluation under")
    parser.add_argument("--compare-to", help="earlier label to compare against, paired on folds")
    parser.add_argument("--notes", default="", help="one line describing what changed")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"cross-validation repeats (default {DEFAULT_REPEATS})")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS,
                        help=f"folds per repeat (default {DEFAULT_FOLDS})")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                        help=f"first seed; repeat r uses base_seed + r (default {DEFAULT_BASE_SEED})")
    parser.add_argument("--taxonomy", "--track", dest="taxonomy", default="audited",
                        help="ground truth to measure against: 'audited' (default, the "
                             "corrected labels), 'topic' (the benchmark's category labels), "
                             "or a historical taxonomy name")
    parser.add_argument("--dataset", default="core", choices=["core", "extended"],
                        help="'core' is the 93 benchmark prompts; 'extended' adds the "
                             "router-only prompt set")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and report without recording anything")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing label in the registry")
    parser.add_argument("--list", action="store_true",
                        help="list router configurations and recorded experiments")
    parser.add_argument("--brief", action="store_true",
                        help="regenerate docs/PHASE3_ROUTER_BRIEF.md from the registry")
    args = parser.parse_args()

    if args.list:
        logger.info("Router configurations:")
        for name, config in sorted(ROUTER_CONFIGS.items()):
            logger.info("  %-18s %s", name, config.description)
        registry = load_registry()
        logger.info("Recorded experiments (%s):", EVAL_DIR / "experiments.csv")
        if not len(registry):
            logger.info("  none yet")
        for _, row in registry.iterrows():
            logger.info(
                "  %-22s %-16s accuracy %s",
                row["label"], row["config"],
                format_percent(float(row["accuracy_mean"]), float(row["accuracy_ci95"])),
            )
        data = load_labelled_prompts()
        logger.info("Labelled prompts: %d over %d routes", len(data), data["route"].nunique())
        for route, count in data["route"].value_counts().items():
            logger.info("  %3d  %s", count, route)
        return

    if args.brief:
        write_brief()
        return

    if not args.config:
        parser.print_help()
        return

    if not args.label and not args.dry_run:
        raise ValueError(
            "--label is required so the evaluation can be recorded as a step in the "
            "progression. Pass --dry-run to evaluate without recording."
        )

    result = evaluate_config(
        args.config, repeats=args.repeats, folds=args.folds,
        base_seed=args.base_seed, taxonomy=args.taxonomy, dataset=args.dataset,
    )
    log_result(result)

    if args.compare_to:
        logger.info("")
        logger.info("  Paired comparison against '%s' on the same folds", args.compare_to)
        log_comparison(args.label or args.config, args.compare_to, result)

    if args.dry_run:
        logger.info("Dry run: nothing recorded.")
        return

    record_experiment(
        result, label=args.label, notes=args.notes,
        compare_to=args.compare_to, force=args.force,
    )
    write_brief()


if __name__ == "__main__":
    main()
