"""
Join the rubric annotations to the runs they came from and answer RQ2.

This is the step the whole grading pass exists for. Until it runs, the
annotations are a pile of grades with no condition attached, which is what kept
the pass blind. Running it opens the blinding map and reports what retrieval did.

Everything is paired. All eight runs answer the same 52 prompts, so a baseline
response and a RAG response exist for every prompt, model and condition, and each
model is compared against itself rather than against the other model.

Usage:

  python scripts/score_rag_rubric.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.rubric import ANNOTATED_FIELDS, detect_flags  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent


def load_scored(grading_dir: Path) -> pd.DataFrame:
    """Join every annotation to its run, and add the two detected flags."""
    files = sorted((grading_dir / "annotations").glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotation batches in {grading_dir / 'annotations'}")
    annotations = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    mapping = pd.read_csv(grading_dir / "blind_map.csv")
    missing = set(mapping["grade_id"]) - set(annotations["grade_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} response(s) have no annotation, first few: "
            f"{sorted(missing)[:5]}"
        )

    # many_to_one, not one_to_one. When prepare_grading.py --dedup is used, two
    # arms that produced byte-identical text for a prompt share one grade_id and
    # so appear as two map rows against one annotation. That is the saving, and
    # it is deliberate. What must still hold is that no grade_id is annotated
    # twice, which validate="many_to_one" enforces on the right side and
    # validate_annotations has already checked; and that no map row is left
    # without a grade, which the missing check above enforces.
    if not annotations["grade_id"].is_unique:
        duplicated = annotations.loc[annotations["grade_id"].duplicated(),
                                     "grade_id"].tolist()
        raise ValueError(
            f"A grade_id is annotated more than once: {duplicated[:5]}. Each "
            f"response is graded once, even when several arms share it."
        )
    scored = mapping.merge(annotations, on="grade_id", how="left",
                           validate="many_to_one")
    detected = scored["response"].apply(detect_flags).apply(pd.Series)
    scored = pd.concat([scored, detected], axis=1)
    # Arms, not a boolean. Every retrieval arm carries rag_enabled=1, so once
    # there are three of them a boolean split pools them into one group and the
    # paired test receives mismatched arrays. Runs generated before arms existed
    # have no arm column and fall back to the two-arm naming, so those passes
    # re-score identically.
    if "arm" in scored.columns:
        scored["arm"] = scored["arm"].fillna("").replace("", pd.NA)
        scored["arm"] = scored["arm"].fillna(
            scored["rag_enabled"].astype(bool).map({True: "rag", False: "baseline"})
        )
    else:
        scored["arm"] = scored["rag_enabled"].astype(bool).map(
            {True: "rag", False: "baseline"})
    scored["run"] = scored["model_id"] + " " + scored["condition"]
    # A pass where nobody wrote a note reads the column back as all-NaN floats,
    # and .str then raises rather than returning no matches. Coerced to string
    # first, so "no notes" means "no truncation found" instead of a crash.
    scored["truncated"] = (scored["note"].fillna("").astype(str)
                           .str.contains("runcat", case=False, na=False))
    return scored


def _arm_frame(frame: pd.DataFrame, arm: str, index, where: str) -> pd.DataFrame:
    """
    Select one arm and index it, refusing to continue on a duplicate key.

    A duplicate key means two responses to the same prompt in the same arm,
    which silently misaligns every paired test downstream. Better to stop here
    than to publish a plausible-looking p value computed on mismatched rows.
    """
    selected = frame[frame["arm"] == arm]
    if selected.empty:
        raise ValueError(
            f"No rows for arm {arm!r} in {where}. Arms present: "
            f"{sorted(frame['arm'].unique())}"
        )
    indexed = selected.set_index(index).sort_index()
    if indexed.index.has_duplicates:
        duplicated = indexed.index[indexed.index.duplicated()].tolist()[:5]
        raise ValueError(
            f"Arm {arm!r} in {where} has more than one response per prompt; "
            f"first duplicates: {duplicated}. Pairing is impossible until the "
            f"arm labels separate them."
        )
    return indexed


def paired_test(before: pd.Series, after: pd.Series) -> dict:
    """Paired comparison of one measure between baseline and RAG on the same prompts."""
    from scipy import stats

    # Cast explicitly: the truncation flag is boolean, and numpy refuses to
    # subtract booleans.
    before = before.to_numpy(dtype=float)
    after = after.to_numpy(dtype=float)
    difference = after - before
    if difference.std(ddof=1) == 0:
        return {"delta": float(difference.mean()), "t": float("nan"),
                "p": float("nan"), "dz": 0.0}
    t_stat, p_value = stats.ttest_rel(after, before)
    return {
        "delta": float(difference.mean()),
        "t": float(t_stat),
        "p": float(p_value),
        "dz": float(difference.mean() / difference.std(ddof=1)),
    }


def main() -> None:
    """Report the rubric results by run and the paired baseline-to-RAG comparison."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-dir", default="data/grading",
                        help="Grading pass to score (default: the Phase 2 pass)")
    parser.add_argument("--prefix", default="phase2",
                        help="Prefix for the output files, so passes do not collide")
    parser.add_argument("--baseline-arm", default="baseline",
                        help="Arm treated as the comparator")
    parser.add_argument("--contrast-arm", default="rag",
                        help="Arm compared against the baseline arm")
    args = parser.parse_args()

    grading_dir = BASE_DIR / args.grading_dir
    out_path = grading_dir / f"{args.prefix}_results_scored.csv"

    scored = load_scored(grading_dir)
    scored.to_csv(out_path, index=False)

    baseline_arm, contrast_arm = args.baseline_arm, args.contrast_arm
    present = sorted(scored["arm"].unique())
    for name in (baseline_arm, contrast_arm):
        if name not in present:
            raise ValueError(
                f"Arm {name!r} is not in this grading pass. Arms present: "
                f"{present}. Pass --baseline-arm and --contrast-arm to choose "
                f"which pair to compare."
            )
    logger.info("Arms present: %s. Comparing %s against %s.",
                ", ".join(present), baseline_arm, contrast_arm)

    measures = ["quality_score", "hallucination_flag", "diagnostic_overreach_flag",
                "addresses_question_flag", "screening_diagnosis_distinction_flag",
                "professional_followup_flag", "truncated"]

    logger.info("=" * 78)
    logger.info("PHASE 2 RUBRIC RESULTS BY RUN (n = 52 per run)")
    logger.info("=" * 78)
    logger.info("  %-26s %7s %7s %8s %8s %7s", "run", "quality", "halluc", "overreach",
                "answers", "trunc")
    for (run, arm), group in scored.groupby(["run", "arm"]):
        label = f"{run} {arm}"
        logger.info("  %-26s %7.2f %6.1f%% %8.1f%% %7.1f%% %6.1f%%",
                    label, group["quality_score"].mean(),
                    group["hallucination_flag"].mean() * 100,
                    group["diagnostic_overreach_flag"].mean() * 100,
                    group["addresses_question_flag"].mean() * 100,
                    group["truncated"].mean() * 100)

    logger.info("")
    logger.info("=" * 78)
    logger.info("POOLED: %s vs %s (n = %d and %d)",
                baseline_arm.upper(), contrast_arm.upper(),
                int((scored["arm"] == baseline_arm).sum()),
                int((scored["arm"] == contrast_arm).sum()))
    logger.info("=" * 78)
    for measure in measures:
        base = scored[scored["arm"] == baseline_arm][measure].mean()
        rag = scored[scored["arm"] == contrast_arm][measure].mean()
        logger.info("  %-42s %7.3f -> %7.3f   %+.3f", measure, base, rag, rag - base)

    logger.info("")
    logger.info("=" * 78)
    logger.info("PAIRED BY PROMPT, EACH MODEL AND CONDITION AGAINST ITSELF")
    logger.info("=" * 78)
    rows = []
    for run, group in scored.groupby("run"):
        base = _arm_frame(group, baseline_arm, "prompt_id", run)
        rag = _arm_frame(group, contrast_arm, "prompt_id", run)
        shared = base.index.intersection(rag.index)
        for measure in ["quality_score", "hallucination_flag", "truncated"]:
            stats = paired_test(base.loc[shared, measure], rag.loc[shared, measure])
            marker = ""
            if stats["p"] == stats["p"]:
                marker = " *" if stats["p"] < 0.05 else " n.s."
            logger.info("  %-24s %-22s %6.3f -> %6.3f  delta %+.3f  p=%.4g  dz=%.2f%s",
                        run, measure, base.loc[shared, measure].mean(),
                        rag.loc[shared, measure].mean(), stats["delta"],
                        stats["p"], stats["dz"], marker)
            rows.append({"run": run, "measure": measure, "n": len(shared),
                         "baseline": base.loc[shared, measure].mean(),
                         "rag": rag.loc[shared, measure].mean(), **stats})

    # Pooled across every run, treating each response pair as one observation.
    base_all = _arm_frame(scored, baseline_arm, ["run", "prompt_id"], "ALL")
    rag_all = _arm_frame(scored, contrast_arm, ["run", "prompt_id"], "ALL")
    shared_all = base_all.index.intersection(rag_all.index)
    logger.info("")
    logger.info("  POOLED across %d run(s) (n = %d pairs)",
                scored["run"].nunique(), len(shared_all))
    for measure in ["quality_score", "hallucination_flag", "truncated"]:
        stats = paired_test(base_all.loc[shared_all, measure], rag_all.loc[shared_all, measure])
        marker = " *" if stats["p"] < 0.05 else " n.s."
        logger.info("  %-24s %-22s %6.3f -> %6.3f  delta %+.3f  p=%.4g  dz=%.2f%s",
                    "ALL", measure, base_all.loc[shared_all, measure].mean(),
                    rag_all.loc[shared_all, measure].mean(), stats["delta"],
                    stats["p"], stats["dz"], marker)
        rows.append({"run": "ALL", "measure": measure, "n": len(shared_all),
                     "baseline": base_all.loc[shared_all, measure].mean(),
                     "rag": rag_all.loc[shared_all, measure].mean(), **stats})

    pd.DataFrame(rows).to_csv(grading_dir / f"{args.prefix}_rubric_paired.csv",
                              index=False)
    logger.info("")
    logger.info("Wrote %s and %s_rubric_paired.csv", out_path, args.prefix)


if __name__ == "__main__":
    main()
