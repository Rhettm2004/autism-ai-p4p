"""
Compare any number of generation arms on every measure each one carries.

This started as a two-directory before/after comparison for the truncation fix.
The end-to-end experiment needs more: four arms — baseline, retrieval, retrieval
plus routing, and the union control — generated in one session into one
directory and told apart by their `arm` column. The layer-by-layer claim the
project rests on is a ladder, and a ladder needs N-way comparison.

Two ways to say what the arms are, because the two experiments differ in shape:

  --runs DIR [DIR ...]        arms come from the `arm` column in the run files.
                              This is the end-to-end shape: one session, one
                              directory, several arms.
  --arms NAME=DIR [...]       arms come from the directory labels. This is the
                              era shape: the same pipeline run at two different
                              times, as in the July-against-August comparison.

Which measures appear depends on what the files carry, and nothing is silently
skipped — the header logs what was found and what was not.

  always            completeness, completion tokens, degeneracy
  metrics sidecars  BERTScore F1 and ROUGE-L, from *_with_metrics.csv
  --scored PATH     the rubric measures: quality, hallucination, diagnostic
                    overreach, addresses-question, professional follow-up and
                    the screening/diagnosis distinction

Everything is paired on prompt, model and condition. Adjacent rungs of the
ladder are compared in the order given by --ladder, because "retrieval beats
baseline" and "routing beats retrieval" are the two claims, and comparing every
arm against the first would answer a different question.

Automated similarity metrics are descriptive only. BERTScore and ROUGE-L are
reported here because the report needs every arm to carry every number, not
because any decision rests on them. The rubric is the primary measure.

Usage:

  python scripts/evaluate_generation.py --runs results_20260901 \\
      --ladder baseline rag routed --label end_to_end \\
      --scored data/grading_20260901/scored.csv
  python scripts/evaluate_generation.py \\
      --arms before=results_thursday after=results_20260825 --label truncation_fix
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
GRADED_PATH = BASE_DIR / "data" / "grading" / "phase2_results_scored.csv"
OUT_DIR = BASE_DIR / "data" / "generation_eval"

# A response is treated as complete when it ends on punctuation that can end a
# sentence. Closing quotes and brackets count, since a response may finish on a
# quoted source.
SENTENCE_END = re.compile(r'[.!?:)\]"”]$')

# Present in every run file.
BASE_MEASURES = ("complete", "completion_tokens", "degenerate")
# Present once run_phase1.py --metrics has been run over the directory.
METRIC_MEASURES = ("bertscore_f1", "rouge_l")
# Present only in a scored rubric file, and the ones that actually decide things.
RUBRIC_MEASURES = ("quality_score", "hallucination_flag",
                   "diagnostic_overreach_flag", "addresses_question_flag",
                   "professional_followup_flag",
                   "screening_diagnosis_distinction_flag")

PAIR_KEY = ["prompt_id", "model_id", "condition"]


def ends_complete(response: str) -> bool:
    """Whether a response ends on sentence-final punctuation rather than mid-word."""
    text = str(response).strip()
    return bool(text) and bool(SENTENCE_END.search(text))


def load_runs(directory: Path) -> pd.DataFrame:
    """
    Load every run in a directory, preferring the metrics sidecars.

    The sidecars carry BERTScore and ROUGE-L and are otherwise identical to the
    plain run files, so reading them when they exist adds two measures for free.
    A directory part-way through metric export would pair sidecars with plain
    files and silently produce half a column, so the two sets are compared and a
    mismatch is refused.
    """
    sidecars = sorted(directory.glob("*_with_metrics.csv"))
    plain = [f for f in sorted(directory.glob("phase1_*.csv"))
             if not f.name.endswith("_with_metrics.csv")]
    if not plain:
        raise ValueError(
            f"No run files in {directory}. Expected files named "
            f"phase1_{{model}}_{{condition}}[_rag{{k}}][_{{arm}}]_{{timestamp}}.csv"
        )
    if sidecars and len(sidecars) != len(plain):
        raise ValueError(
            f"{directory} has {len(plain)} run files but {len(sidecars)} metrics "
            f"sidecars. Finish the export before comparing, or the similarity "
            f"columns would cover only part of the run: "
            f"python scripts/run_phase1.py --metrics <each run file>"
        )
    files = sidecars if sidecars else plain

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name.replace("_with_metrics.csv", ".csv")
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    runs["rag"] = runs["rag_enabled"].astype(bool)
    runs["run"] = (runs["model_id"] + " " + runs["condition"]
                   + runs["rag"].map({True: " RAG", False: " baseline"}))
    runs["complete"] = runs["response"].apply(ends_complete)
    return runs


def arms_from_directories(specs: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build the arm table from NAME=DIR specifications.

    Used when the same pipeline was run at different times, where the arm is the
    era and no `arm` column exists in the files.
    """
    arms = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"--arms takes NAME=DIRECTORY, got {spec!r}. For example: "
                f"--arms before=results_thursday after=results_20260825"
            )
        name, _, path = spec.partition("=")
        name, path = name.strip(), Path(path.strip())
        if name in arms:
            raise ValueError(f"--arms names must be distinct; {name!r} appears twice")
        arms[name] = load_runs(path)
    return arms


def arms_from_column(directories: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build the arm table from the `arm` column of the run files.

    Used for the end-to-end shape, where every arm was generated in one session.
    Refuses a directory with no arm column rather than pooling four arms into
    one: every retrieval arm carries rag_enabled=1, so a pooled frame would look
    perfectly well formed and mean nothing.
    """
    frames = []
    for directory in directories:
        runs = load_runs(Path(directory))
        if "arm" not in runs.columns:
            raise ValueError(
                f"{directory} has no arm column, so its arms cannot be told "
                f"apart. Either the runs predate --arm, in which case use "
                f"--arms NAME={directory}, or they were generated without it."
            )
        blank = runs["arm"].isna() | (runs["arm"].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(
                f"{directory}: {int(blank.sum())} response(s) have an empty arm. "
                f"Re-run those with --arm, or they cannot be attributed."
            )
        frames.append(runs)
    everything = pd.concat(frames, ignore_index=True)
    return {arm: group.reset_index(drop=True)
            for arm, group in everything.groupby("arm")}


def attach_rubric(arms: dict[str, pd.DataFrame], scored_path: Path) -> None:
    """
    Join the graded rubric measures onto each arm, in place.

    Joined on arm, prompt, model and condition rather than on the source
    filename, because a scored file records the run it came from with the
    sidecar suffix stripped inconsistently across passes and the four keys are
    unambiguous.
    """
    scored = pd.read_csv(scored_path)
    available = [m for m in RUBRIC_MEASURES if m in scored.columns]
    if not available:
        raise ValueError(
            f"{scored_path} carries none of the rubric measures "
            f"{list(RUBRIC_MEASURES)}. Available: {sorted(scored.columns)[:10]}"
        )
    if "arm" not in scored.columns:
        raise ValueError(
            f"{scored_path} has no arm column, so its grades cannot be matched "
            f"to an arm. Re-score with scripts/score_rag_rubric.py."
        )
    for name, frame in arms.items():
        subset = scored[scored["arm"] == name]
        if subset.empty:
            logger.warning("No grades for arm %r in %s; its rubric columns will "
                           "be empty", name, scored_path.name)
            continue
        columns = PAIR_KEY + available
        merged = frame.merge(subset[columns].drop_duplicates(PAIR_KEY),
                             on=PAIR_KEY, how="left")
        if len(merged) != len(frame):
            raise ValueError(
                f"Joining grades to arm {name!r} changed the row count from "
                f"{len(frame)} to {len(merged)}, so the grading file has "
                f"duplicate prompt/model/condition rows."
            )
        arms[name] = merged


def available_measures(arms: dict[str, pd.DataFrame]) -> list[str]:
    """Every measure present and numeric in every arm, in a stable order."""
    candidates = list(BASE_MEASURES) + list(METRIC_MEASURES) + list(RUBRIC_MEASURES)
    return [m for m in candidates
            if all(m in frame.columns and frame[m].notna().any()
                   for frame in arms.values())]


def summarise_arms(arms: dict[str, pd.DataFrame], measures: list[str],
                   order: list[str]) -> pd.DataFrame:
    """
    One row per measure, one column per arm: the every-metric-every-arm table.

    This is the table the report is built around, so it is written wide rather
    than long: a reader comparing four arms on eleven measures should not have
    to pivot anything.
    """
    rows = []
    for measure in measures:
        row = {"measure": measure}
        for name in order:
            values = arms[name][measure].astype(float)
            row[name] = float(values.mean())
        row["n_per_arm"] = int(len(arms[order[0]]))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_compare(before: pd.DataFrame, after: pd.DataFrame,
                   measure: str, by_run: bool = True) -> pd.DataFrame:
    """
    Compare one measure between two arms, paired per response.

    With by_run the comparison is split by model and condition, which is what the
    era comparison wants: a fix that helped one model and hurt another must not
    average to nothing. Without it the arms are compared pooled, which is what
    the ladder wants.
    """
    groups = sorted(set(before["run"]) & set(after["run"])) if by_run else [None]
    rows = []
    for run in groups:
        old = before if run is None else before[before["run"] == run]
        new = after if run is None else after[after["run"] == run]
        old = old.set_index(PAIR_KEY).sort_index()
        new = new.set_index(PAIR_KEY).sort_index()
        shared = old.index.intersection(new.index)
        a = old.loc[shared, measure].astype(float)
        b = new.loc[shared, measure].astype(float)
        difference = b.to_numpy() - a.to_numpy()
        row = {"run": run if run is not None else "all", "measure": measure,
               "n": len(shared), "before": a.mean(), "after": b.mean(),
               "delta": difference.mean()}
        if difference.std(ddof=1) > 0:
            t_stat, p_value = stats.ttest_rel(b, a)
            row["t"] = float(t_stat)
            row["p"] = float(p_value)
            row["dz"] = float(difference.mean() / difference.std(ddof=1))
            low, high = stats.t.interval(0.95, len(difference) - 1,
                                         loc=difference.mean(),
                                         scale=stats.sem(difference))
            row["ci_low"], row["ci_high"] = float(low), float(high)
        else:
            row["t"] = row["p"] = float("nan")
            row["dz"] = 0.0
            row["ci_low"] = row["ci_high"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_ladder(arms: dict[str, pd.DataFrame], order: list[str],
                   measures: list[str], by_run: bool) -> pd.DataFrame:
    """
    Compare each adjacent pair of rungs on every measure.

    Adjacent rather than all-against-the-first, because the claim being tested is
    that each added layer improves on the one below it, not that the last one
    beats the first.
    """
    frames = []
    for lower, upper in zip(order, order[1:]):
        for measure in measures:
            frame = paired_compare(arms[lower], arms[upper], measure, by_run=by_run)
            frame.insert(0, "step", f"{lower} -> {upper}")
            frame = frame.rename(columns={"before": lower, "after": upper})
            frame["lower_arm"], frame["upper_arm"] = lower, upper
            frame["lower_mean"] = frame[lower]
            frame["upper_mean"] = frame[upper]
            frames.append(frame.drop(columns=[lower, upper]))
    return pd.concat(frames, ignore_index=True)


def check_heuristic_against_grades(frame: pd.DataFrame) -> None:
    """
    Report how well the completeness heuristic matches the human truncation grades.

    Only possible for run sets that have been graded. Reporting it keeps the
    proxy honest rather than leaving the reader to assume it is exact.
    """
    if not GRADED_PATH.exists():
        return
    graded = pd.read_csv(GRADED_PATH)
    graded["truncated"] = graded["note"].str.contains("runcat", case=False, na=False)
    graded["source_file"] = graded["source_run"].str.replace(
        "_with_metrics.csv", ".csv", regex=False)
    merged = frame.merge(graded[["source_file", "prompt_id", "truncated"]],
                         on=["source_file", "prompt_id"], how="inner")
    if merged.empty:
        return
    agree = (merged["complete"] != merged["truncated"]).mean()
    logger.info("Completeness heuristic agrees with the human truncation grades "
                "on %.1f%% of %d graded responses", agree * 100, len(merged))


def main() -> None:
    """Compare every arm, record the wide summary and the adjacent-pair tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", metavar="DIR",
                        help="Directories whose run files carry an arm column")
    source.add_argument("--arms", nargs="+", metavar="NAME=DIR",
                        help="Named directories, for comparing eras rather than arms")
    parser.add_argument("--label", required=True, help="Name for this comparison")
    parser.add_argument("--ladder", nargs="+", metavar="ARM",
                        help="Arm order for the adjacent-pair tests. Defaults to "
                             "the order given, or alphabetical for --runs.")
    parser.add_argument("--scored", metavar="PATH",
                        help="A scored rubric file, to add the graded measures")
    parser.add_argument("--pooled", action="store_true",
                        help="Compare arms pooled rather than split by model and "
                             "condition. Splitting is the default because a "
                             "change that helps one model and hurts another must "
                             "not average to nothing.")
    args = parser.parse_args()

    if args.arms:
        arms = arms_from_directories(args.arms)
        default_order = [spec.partition("=")[0].strip() for spec in args.arms]
    else:
        arms = arms_from_column(args.runs)
        default_order = sorted(arms)

    order = args.ladder or default_order
    unknown = [name for name in order if name not in arms]
    if unknown:
        raise ValueError(
            f"--ladder names arms that were not loaded: {unknown}. "
            f"Available: {sorted(arms)}"
        )
    arms = {name: arms[name] for name in order}

    if args.scored:
        attach_rubric(arms, Path(args.scored))

    measures = available_measures(arms)
    if not measures:
        raise ValueError("No measure is present in every arm, so nothing can be "
                         "compared.")
    absent = [m for m in list(BASE_MEASURES) + list(METRIC_MEASURES)
              + list(RUBRIC_MEASURES) if m not in measures]

    logger.info("=" * 78)
    logger.info("GENERATION COMPARISON: %s", args.label)
    logger.info("=" * 78)
    for name in order:
        logger.info("  %-12s %4d responses, %2d runs", name, len(arms[name]),
                    arms[name]["run"].nunique())
    logger.info("  measures present: %s", ", ".join(measures))
    if absent:
        logger.info("  measures absent:  %s", ", ".join(absent))
        if any(m in absent for m in RUBRIC_MEASURES) and not args.scored:
            logger.info("  (pass --scored to add the graded rubric measures)")

    check_heuristic_against_grades(arms[order[0]])

    summary = summarise_arms(arms, measures, order)
    logger.info("")
    logger.info("  %-38s %s", "measure",
                " ".join(f"{name:>12s}" for name in order))
    for _, row in summary.iterrows():
        logger.info("  %-38s %s", row["measure"],
                    " ".join(f"{row[name]:12.4f}" for name in order))

    ladder = compare_ladder(arms, order, measures, by_run=not args.pooled)
    logger.info("")
    for step, group in ladder.groupby("step", sort=False):
        logger.info("  %s", step)
        for _, row in group.iterrows():
            if row["p"] != row["p"]:
                marker = ""
            else:
                marker = " *" if row["p"] < 0.05 else " n.s."
            logger.info("    %-34s %-22s %8.4f -> %8.4f  %+8.4f  p=%-9.4g%s",
                        row["measure"], row["run"], row["lower_mean"],
                        row["upper_mean"], row["delta"], row["p"], marker)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"arm_summary_{args.label}.csv"
    ladder_path = OUT_DIR / f"paired_{args.label}.csv"
    summary.to_csv(summary_path, index=False)
    ladder.to_csv(ladder_path, index=False)

    columns = ["source_file", "run", "model_id", "condition", "rag", "prompt_id",
               "category", "prompt_tokens"] + measures
    per_response = pd.concat(
        [arms[name][[c for c in columns if c in arms[name].columns]].assign(arm=name)
         for name in order],
        ignore_index=True,
    )
    per_response_path = OUT_DIR / f"per_response_{args.label}.csv"
    per_response.to_csv(per_response_path, index=False)

    logger.info("")
    logger.info("Wrote %s, %s and %s", summary_path.name, ladder_path.name,
                per_response_path.name)


if __name__ == "__main__":
    main()
