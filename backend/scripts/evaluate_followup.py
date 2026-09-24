"""
Measure route-appropriate follow-up conformance, not the flat follow-up rate.

The flat rate is the wrong target and always was. `professional_followup_flag`
fires when a response tells the caregiver to speak to someone, and the project
has been reporting the share of all responses that do. But a caregiver asking
what autism prevalence means does not need to be told to see a doctor, and
counting that omission as a failure pushes the system towards reflexive advice.

`config/prompts.yaml` declares, per route, whether a next step is `required`,
`expected` or `optional`. Those obligations were frozen before any routed output
existed. This script scores conformance against them:

  conformance      of the responses whose turn OWED a next step
                   (required or expected), the share that gave one
  unprompted rate  of the responses whose turn owed nothing (optional),
                   the share that gave one anyway
  discrimination   conformance minus unprompted rate, within one arm: whether
                   the arm targets its advice at the turns that need it, or
                   sprays it uniformly

Discrimination is the measure that matters, and it is the one the flat rate
cannot express at all. Two systems can post an identical flat rate while one
advises the turns that need advice and the other advises at random.

Which route labels. Obligation is a property of the turn, not of what the router
guessed, so the audited `true_route` is used by default. Scoring against the
predicted route would fold routing error into a generation measure. For a routed
arm both readings are interesting — conformance to what the model was told
versus to what was true — and `--route-column` selects between them.

What this cannot show, and it is a hard limit. Only 10 of the 52 benchmark
prompts owe a next step: 7 required and 3 expected. Every conformance figure
here rests on those 10, and the paired arm comparison on 40 responses. The
benchmark was not built to test this and the confidence intervals say so.

The detector is also presence, not appropriateness. FOLLOWUP_PATTERNS matches
phrases like "speak to your doctor". A response that names the wrong service, or
buries the advice, or gives it for the wrong reason, scores the same as one that
gets it right. Conformance here is a floor on quality, never a measure of it.

Usage:

  python scripts/evaluate_followup.py \\
      --scored data/grading_20260825/rerun_results_scored.csv --label rerun_20260825
  python scripts/evaluate_followup.py --scored ... --label ... --route-column route
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
PROMPTS_PATH = BASE_DIR / "config" / "prompts.yaml"
DEFAULT_ROUTE_MAP = BASE_DIR / "data" / "route_maps" / "oracle.csv"
OUT_DIR = BASE_DIR / "data" / "generation_eval"

FLAG = "professional_followup_flag"
OWED_LEVELS = ("required", "expected")
PAIR_KEY = ["prompt_id", "model_id", "condition"]
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260826


def load_obligations() -> dict[str, str]:
    """
    Return route -> `required` / `expected` / `optional` from prompts.yaml.

    Read from the same file the generation prompts come from, so the obligation
    a response is scored against is by construction the one it was given.
    """
    config = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    routes = config.get("routes")
    if not routes:
        raise ValueError(
            f"{PROMPTS_PATH} has no routes: block, so no obligations are declared. "
            f"Route-appropriate conformance cannot be scored without one."
        )
    obligations = {name: block["followup"]
                   for name, block in routes.items() if name != "_meta"}
    unknown = {v for v in obligations.values()} - {"required", "expected", "optional"}
    if unknown:
        raise ValueError(
            f"{PROMPTS_PATH} declares unknown follow-up levels {sorted(unknown)}. "
            f"Valid levels: required, expected, optional."
        )
    return obligations


def attach_obligations(scored: pd.DataFrame, route_map: Path,
                       route_column: str) -> pd.DataFrame:
    """
    Join each graded response to what its turn owed the caregiver.

    Raises rather than dropping unmatched rows: a silently unjoined prompt would
    shrink the denominator of a measure that already rests on ten prompts.
    """
    if FLAG not in scored.columns:
        raise ValueError(
            f"The scored file has no {FLAG} column, so there is nothing to score "
            f"conformance on. Available: {sorted(scored.columns)[:8]}"
        )
    if "arm" not in scored.columns:
        raise ValueError(
            "The scored file has no arm column. Every retrieval arm carries "
            "rag_enabled=1, so without it the arms cannot be told apart. "
            "Re-score with scripts/score_rag_rubric.py."
        )

    routes = pd.read_csv(route_map)
    if route_column not in routes.columns:
        raise ValueError(
            f"{route_map} has no {route_column!r} column. Available: "
            f"{sorted(routes.columns)}"
        )

    obligations = load_obligations()
    routes = routes[["prompt_id", route_column]].rename(columns={route_column: "route"})
    unmapped = set(routes["route"]) - set(obligations)
    if unmapped:
        raise ValueError(
            f"{route_map} uses routes with no declared obligation: "
            f"{sorted(unmapped)}. Add them to the routes: block in {PROMPTS_PATH}."
        )

    # A grading pass and a route map from different benchmark revisions join
    # cleanly on prompt_id and attach the wrong route to nearly every prompt,
    # because revisions reuse ids for different questions (D-124). Comparing
    # coverage is a cheap proxy that catches the realistic case: a 52-prompt
    # pass scored against a 102-prompt map, or the reverse.
    graded_ids = set(scored["prompt_id"])
    map_ids = set(routes["prompt_id"])
    if not graded_ids <= map_ids:
        raise ValueError(
            f"{len(graded_ids - map_ids)} graded prompt(s) are absent from "
            f"{route_map.name}: {sorted(graded_ids - map_ids)[:5]}"
        )
    if len(map_ids) != len(graded_ids):
        raise ValueError(
            f"The grading pass covers {len(graded_ids)} prompts but "
            f"{route_map.name} covers {len(map_ids)}. They are from different "
            f"benchmark revisions, and prompt ids mean different questions in "
            f"different revisions, so this join would be silently wrong "
            f"(D-124). Use the archived map matching the pass."
        )

    merged = scored.merge(routes, on="prompt_id", how="left")
    missing = merged[merged["route"].isna()]["prompt_id"].unique()
    if len(missing):
        raise ValueError(
            f"{len(missing)} prompt(s) in the scored file have no route in "
            f"{route_map.name}: {sorted(missing)[:5]}"
        )
    merged["obligation"] = merged["route"].map(obligations)
    merged["owed"] = merged["obligation"].isin(OWED_LEVELS)
    return merged


def rate(values: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Return a mean and its 95% confidence interval, widening to (nan, nan) at n<2."""
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float(values.mean()), (float("nan"), float("nan"))
    low, high = stats.t.interval(0.95, len(values) - 1,
                                 loc=values.mean(), scale=stats.sem(values))
    return float(values.mean()), (float(low), float(high))


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: conformance, unprompted rate, discrimination."""
    rows = []
    for arm, group in frame.groupby("arm"):
        owed = group[group["owed"]][FLAG].to_numpy(dtype=float)
        free = group[~group["owed"]][FLAG].to_numpy(dtype=float)
        conformance, (c_low, c_high) = rate(owed)
        unprompted, (u_low, u_high) = rate(free)
        # Welch, because the two buckets are different prompts and very
        # different sizes. This is a within-arm contrast, not a paired one.
        _, p_value = stats.ttest_ind(owed, free, equal_var=False)
        rows.append({
            "arm": arm,
            "n_owed": len(owed), "n_optional": len(free),
            "conformance": conformance, "conformance_ci_low": c_low,
            "conformance_ci_high": c_high,
            "unprompted_rate": unprompted, "unprompted_ci_low": u_low,
            "unprompted_ci_high": u_high,
            "discrimination": conformance - unprompted,
            "discrimination_p": float(p_value),
            "flat_rate": float(group[FLAG].mean()),
        })
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def paired_arm_test(frame: pd.DataFrame, baseline_arm: str,
                    contrast_arm: str) -> pd.DataFrame:
    """
    Compare two arms within each obligation bucket, paired per response.

    Paired on prompt, model and condition, which is the same unit the rest of
    the project pairs on, so a difference here is a difference in behaviour
    rather than in which prompts happened to land in which arm.
    """
    rows = []
    buckets = [("owes a next step", frame[frame["owed"]]),
               ("required", frame[frame["obligation"] == "required"]),
               ("expected", frame[frame["obligation"] == "expected"]),
               ("optional", frame[~frame["owed"]])]
    for name, subset in buckets:
        before = subset[subset["arm"] == baseline_arm].set_index(PAIR_KEY)[FLAG]
        after = subset[subset["arm"] == contrast_arm].set_index(PAIR_KEY)[FLAG]
        before, after = before.sort_index(), after.sort_index()
        if len(before) == 0 or not before.index.equals(after.index):
            raise ValueError(
                f"{name}: the two arms do not cover the same responses "
                f"({len(before)} vs {len(after)}), so they cannot be paired."
            )
        difference = after.to_numpy(dtype=float) - before.to_numpy(dtype=float)
        spread = difference.std(ddof=1)
        if spread == 0:
            p_value, dz, low, high = float("nan"), float("nan"), 0.0, 0.0
        else:
            _, p_value = stats.ttest_rel(after, before)
            dz = difference.mean() / spread
            low, high = stats.t.interval(0.95, len(difference) - 1,
                                         loc=difference.mean(),
                                         scale=stats.sem(difference))
        rows.append({
            "bucket": name, "n_pairs": len(difference),
            f"{baseline_arm}_rate": float(before.mean()),
            f"{contrast_arm}_rate": float(after.mean()),
            "difference": float(difference.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "p_value": float(p_value), "cohens_dz": float(dz),
        })
    return pd.DataFrame(rows)


def bootstrap_discrimination_gap(frame: pd.DataFrame, baseline_arm: str,
                                 contrast_arm: str) -> dict:
    """
    Confidence interval on the difference in discrimination between two arms.

    Resampled over prompts rather than responses, because the four responses to
    one prompt are not independent of each other, and because the quantity that
    is scarce here is prompts: ten of them carry every owed observation.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    prompts = frame["prompt_id"].unique()
    # Row positions per prompt, so a resample is one concatenate rather than a
    # per-prompt filter. The naive version is quadratic and does not finish.
    positions = {pid: np.flatnonzero(frame["prompt_id"].to_numpy() == pid)
                 for pid in prompts}
    arm_values = frame["arm"].to_numpy()
    owed_values = frame["owed"].to_numpy()
    flag_values = frame[FLAG].to_numpy(dtype=float)

    def gap(rows: np.ndarray) -> float:
        """Discrimination of the contrast arm minus that of the baseline arm."""
        out = []
        for arm in (contrast_arm, baseline_arm):
            in_arm = arm_values[rows] == arm
            owed = flag_values[rows][in_arm & owed_values[rows]]
            free = flag_values[rows][in_arm & ~owed_values[rows]]
            if len(owed) == 0 or len(free) == 0:
                return float("nan")
            out.append(owed.mean() - free.mean())
        return out[0] - out[1]

    observed = gap(np.arange(len(frame)))
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = rng.choice(prompts, len(prompts), replace=True)
        rows = np.concatenate([positions[p] for p in picked])
        value = gap(rows)
        if not np.isnan(value):
            draws.append(value)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "metric": "discrimination_gap",
        "baseline_arm": baseline_arm, "contrast_arm": contrast_arm,
        "observed": float(observed),
        "ci_low": float(low), "ci_high": float(high),
        "n_prompts": len(prompts), "n_resamples": len(draws),
        "seed": BOOTSTRAP_SEED,
    }


def main() -> None:
    """Score conformance for every arm in a graded file and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True,
                        help="A scored rubric file carrying arm and "
                             "professional_followup_flag")
    parser.add_argument("--label", required=True,
                        help="Names the output files, e.g. rerun_20260825")
    parser.add_argument("--route-map", required=True,
                        help="Route map to take obligations from. Required, and "
                             "deliberately has no default: benchmark revisions "
                             "replace data/route_maps/oracle.csv in place, and a "
                             "default would silently score an older grading pass "
                             "against a newer revision's routes. Prompt ids are "
                             "reused across revisions, so that join succeeds and "
                             "is wrong (D-124). Archived maps are in "
                             "data/route_maps/archive/.")
    parser.add_argument("--route-column", default="true_route",
                        help="Which column names the turn's route. Default "
                             "true_route: obligation is a property of the turn, "
                             "not of what the router guessed.")
    parser.add_argument("--baseline-arm", default="baseline",
                        help="Arm to compare against (default baseline)")
    parser.add_argument("--contrast-arm", default="rag",
                        help="Arm to compare (default rag)")
    args = parser.parse_args()

    scored = pd.read_csv(args.scored)
    frame = attach_obligations(scored, Path(args.route_map), args.route_column)

    prompts = frame.drop_duplicates("prompt_id")
    logger.info("=" * 78)
    logger.info("ROUTE-APPROPRIATE FOLLOW-UP: %s", args.label)
    logger.info("=" * 78)
    logger.info("Obligations from %s via %s.%s",
                PROMPTS_PATH.name, Path(args.route_map).name, args.route_column)
    for level in ("required", "expected", "optional"):
        count = int((prompts["obligation"] == level).sum())
        logger.info("  %-9s %2d of %d prompts", level, count, len(prompts))
    owed_prompts = int(prompts["owed"].sum())
    logger.info("  Every conformance figure below rests on those %d prompts",
                owed_prompts)

    summary = summarise(frame)
    logger.info("")
    logger.info("  %-10s %7s %7s %7s %9s %8s", "arm", "flat", "conform",
                "unprompt", "discrim", "p")
    for _, row in summary.iterrows():
        logger.info("  %-10s %6.1f%% %6.1f%% %7.1f%% %+8.1fpp %8.4f",
                    row["arm"], row["flat_rate"] * 100, row["conformance"] * 100,
                    row["unprompted_rate"] * 100, row["discrimination"] * 100,
                    row["discrimination_p"])

    paired = paired_arm_test(frame, args.baseline_arm, args.contrast_arm)
    logger.info("")
    logger.info("  %s vs %s, paired per response:", args.contrast_arm,
                args.baseline_arm)
    logger.info("  %-18s %5s %8s %8s %20s %8s", "bucket", "n", args.baseline_arm[:8],
                args.contrast_arm[:8], "difference 95% CI", "p")
    for _, row in paired.iterrows():
        logger.info("  %-18s %5d %7.1f%% %7.1f%%  %+6.1fpp [%+5.1f,%+5.1f] %8.4f",
                    row["bucket"], row["n_pairs"],
                    row[f"{args.baseline_arm}_rate"] * 100,
                    row[f"{args.contrast_arm}_rate"] * 100,
                    row["difference"] * 100, row["ci_low"] * 100,
                    row["ci_high"] * 100, row["p_value"])

    gap = bootstrap_discrimination_gap(frame, args.baseline_arm, args.contrast_arm)
    logger.info("")
    logger.info("  Discrimination gap (%s minus %s): %+.1fpp, 95%% CI "
                "[%+.1f, %+.1f], bootstrapped over %d prompts",
                args.contrast_arm, args.baseline_arm, gap["observed"] * 100,
                gap["ci_low"] * 100, gap["ci_high"] * 100, gap["n_prompts"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"followup_conformance_{args.label}.csv"
    paired_path = OUT_DIR / f"followup_paired_{args.label}.csv"
    gap_path = OUT_DIR / f"followup_discrimination_{args.label}.csv"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    pd.DataFrame([gap]).to_csv(gap_path, index=False)
    logger.info("")
    logger.info("Wrote %s, %s and %s", summary_path.name, paired_path.name,
                gap_path.name)


if __name__ == "__main__":
    main()
