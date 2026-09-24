"""
Export every number the final report needs, as tidy CSVs and as markdown tables.

A figure is a picture of a number, and a picture cannot be re-plotted, checked or
recomputed. This script writes the numbers themselves: one CSV per table, in a
tidy shape that can be pasted into a spreadsheet or read back into a script, plus
docs/report/NUMBERS.md containing all of them as markdown tables with their
provenance.

Everything is computed from tracked inputs. Nothing is hand-entered, so a
recomputation after a data change silently corrects every table at once.

Usage examples:

  python scripts/export_report_numbers.py            # everything
  python scripts/export_report_numbers.py --group phase2
  python scripts/export_report_numbers.py --table p1_rubric_by_run
  python scripts/export_report_numbers.py --list
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
HISTORY_DIR = BASE_DIR / "data" / "history"
ROUTER_DIR = BASE_DIR / "data" / "router_eval"
THURSDAY_DIR = BASE_DIR / "results_thursday"
OUT_DIR = BASE_DIR / "docs" / "report" / "numbers"
NUMBERS_MD = BASE_DIR / "docs" / "report" / "NUMBERS.md"


@dataclass
class Table:
    """One exported table: how to compute it, and what it is for."""

    name: str
    group: str
    title: str
    caption: str
    sources: list[str]
    build: Callable[[], pd.DataFrame]
    figure_hint: str = ""


TABLES: dict[str, Table] = {}


def register(name: str, group: str, title: str, caption: str, sources: list[str],
             figure_hint: str = ""):
    """Register a table builder under a name usable from the command line."""
    def decorator(func: Callable[[], pd.DataFrame]) -> Callable[[], pd.DataFrame]:
        if name in TABLES:
            raise ValueError(f"Table '{name}' is already registered")
        TABLES[name] = Table(name=name, group=group, title=title, caption=caption,
                             sources=sources, build=func, figure_hint=figure_hint)
        return func
    return decorator


def _require(path: Path) -> pd.DataFrame:
    """Load a required input, failing with the path rather than a parser error."""
    if not path.exists():
        raise FileNotFoundError(f"Input missing: {path}")
    return pd.read_csv(path)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a proportion.

    Used rather than the normal approximation because several of these rates are
    computed on fewer than fifty observations, where the normal interval can run
    below zero.
    """
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(centre - spread), float(centre + spread)


def _run_label(frame: pd.DataFrame) -> pd.Series:
    """Return a readable run label from whichever model column the file uses."""
    model = frame["model_name"] if "model_name" in frame.columns else frame["model_id"]
    return model + " " + frame["condition"]


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------

@register(
    "p1_rubric_by_run", "phase1",
    "Phase 1 rubric results by run",
    "Rubric annotation of all 176 Phase 1 responses, 44 per run. Quality is 0 "
    "unsafe to 3 strong. Intervals are Wilson score intervals on the "
    "hallucination rate. Grades are model-assisted with human adjudication; no "
    "independent second scorer, so no agreement figure. See F-P1-008.",
    ["data/history/phase1_results_scored.csv"],
    figure_hint="Grouped bar of hallucination rate by run with error bars, or a "
                "stacked bar of the quality distribution.",
)
def _p1_rubric_by_run() -> pd.DataFrame:
    """Aggregate the rubric to one row per run."""
    scored = _require(HISTORY_DIR / "phase1_results_scored.csv")
    scored["run"] = _run_label(scored)
    rows = []
    for run, group in scored.groupby("run"):
        successes = int(group["hallucination_flag"].sum())
        low, high = _wilson_interval(successes, len(group))
        rows.append({
            "run": run,
            "n": len(group),
            "quality_mean": round(float(group["quality_score"].mean()), 3),
            "quality_sd": round(float(group["quality_score"].std(ddof=1)), 3),
            "hallucination_rate": round(successes / len(group), 4),
            "hallucination_ci_low": round(low, 4),
            "hallucination_ci_high": round(high, 4),
            "diagnostic_overreach_rate": round(float(group["diagnostic_overreach_flag"].mean()), 4),
            "addresses_question_rate": round(float(group["addresses_question_flag"].mean()), 4),
            "professional_followup_rate": round(float(group["professional_followup_flag"].mean()), 4),
            "screening_diagnosis_distinction_rate":
                round(float(group["screening_diagnosis_distinction_flag"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


@register(
    "p1_by_category", "phase1",
    "Phase 1 quality and hallucination by prompt category",
    "All four runs pooled (n = 176). Shows that errors concentrate on questions "
    "about the screening instruments while the safety categories have none.",
    ["data/history/phase1_results_scored.csv"],
    figure_hint="Horizontal bar of hallucination rate by category, sorted, with n "
                "annotated. Pair with the Phase 2 retrieval-by-category table.",
)
def _p1_by_category() -> pd.DataFrame:
    """Aggregate the rubric by benchmark category, pooled over runs."""
    scored = _require(HISTORY_DIR / "phase1_results_scored.csv")
    rows = []
    for category, group in scored.groupby("category"):
        successes = int(group["hallucination_flag"].sum())
        low, high = _wilson_interval(successes, len(group))
        rows.append({
            "category": category,
            "n": len(group),
            "quality_mean": round(float(group["quality_score"].mean()), 3),
            "hallucination_rate": round(successes / len(group), 4),
            "hallucination_ci_low": round(low, 4),
            "hallucination_ci_high": round(high, 4),
            "addresses_question_rate": round(float(group["addresses_question_flag"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("hallucination_rate", ascending=False).reset_index(drop=True)


@register(
    "p1_factual_errors", "phase1",
    "Every recorded Phase 1 factual error",
    "The 11 responses graded unsafe, each with the false statement recorded "
    "verbatim against the source that contradicts it.",
    ["data/history/factual_errors.json"],
    figure_hint="Better as a table in the report than a figure. Group by error "
                "class to support the three recurring classes.",
)
def _p1_factual_errors() -> pd.DataFrame:
    """Flatten the recorded factual errors, one row per issue."""
    payload = json.loads((HISTORY_DIR / "factual_errors.json").read_text(encoding="utf-8"))
    rows = []
    for entry in payload:
        for issue in entry["issues"]:
            if "age" in issue.lower():
                error_class = "wrong age range"
            elif "prevalence" in issue.lower() or " in " in issue.lower():
                error_class = "stale prevalence"
            elif "score" in issue.lower() or "threshold" in issue.lower():
                error_class = "reversed or wrong scoring"
            else:
                error_class = "other"
            rows.append({
                "model": entry["model"],
                "condition": entry["cond"],
                "prompt_id": entry["prompt_id"],
                "category": entry["category"],
                "error_class": error_class,
                "issue": issue,
            })
    return pd.DataFrame(rows)


@register(
    "p1_similarity_vs_quality", "phase1",
    "Automated similarity against manual quality",
    "Correlation between the automated metrics and the rubric on the same 176 "
    "responses, plus the mean similarity of the responses graded unsafe.",
    ["data/history/phase1_results_scored.csv"],
    figure_hint="Scatter of BERTScore against quality with a fitted line; the flat "
                "slope is the point. The unsafe-response means belong in the caption.",
)
def _p1_similarity_vs_quality() -> pd.DataFrame:
    """Correlate each automated metric with rubric quality and isolate unsafe responses."""
    from scipy import stats

    scored = _require(HISTORY_DIR / "phase1_results_scored.csv")
    rows = []
    for metric in ("bertscore_f1", "rouge_l"):
        usable = scored.dropna(subset=[metric, "quality_score"])
        r, p = stats.pearsonr(usable[metric], usable["quality_score"])
        unsafe = usable[usable["quality_score"] == 0]
        rows.append({
            "metric": metric,
            "n": len(usable),
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "mean_all": round(float(usable[metric].mean()), 4),
            "mean_unsafe_responses": round(float(unsafe[metric].mean()), 4),
            "n_unsafe": len(unsafe),
            "n_unsafe_above_overall_mean": int((unsafe[metric] > usable[metric].mean()).sum()),
        })
    return pd.DataFrame(rows)


@register(
    "p1_degeneracy_by_run", "phase1",
    "Repetition in the pre-repetition-control responses",
    "Measured on the 176 Phase 1 responses generated without repetition control. "
    "After the control was added, all 416 Phase 2 responses were clean.",
    ["data/history/degeneration_metrics.csv"],
    figure_hint="Grouped bar per run: share with a repeated 5-gram, and mean "
                "repeated-token fraction. The Llama zero-shot bar is the story.",
)
def _p1_degeneracy_by_run() -> pd.DataFrame:
    """Aggregate repetition measures to one row per run."""
    metrics = _require(HISTORY_DIR / "degeneration_metrics.csv")
    metrics["run"] = _run_label(metrics)
    rows = []
    for run, group in metrics.groupby("run"):
        rows.append({
            "run": run,
            "n": len(group),
            "degenerate_strict": int(group["degenerate"].sum()),
            "share_with_repeated_5gram": round(float((group["max_ngram_rep"] > 1).mean()), 4),
            "mean_repeated_token_fraction": round(float(group["rep_token_frac"].mean()), 4),
            "mean_distinct_4": round(float(group["distinct_4"].mean()), 4),
            "mean_tokens": round(float(group["n_tokens"].mean()), 1),
        })
    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


# --------------------------------------------------------------------------
# Phase 2
# --------------------------------------------------------------------------

@register(
    "p2_recall_curve", "phase2",
    "Retrieval recall against k",
    "Share of prompts whose correct source passage appears in the top k results, "
    "on the 52-prompt benchmark.",
    ["data/history/retrieval_recall_curve.csv"],
    figure_hint="Line plot, recall against k. Mark k=5, the value Phase 2 uses.",
)
def _p2_recall_curve() -> pd.DataFrame:
    """Return the recall curve with recall expressed as a percentage as well."""
    curve = _require(HISTORY_DIR / "retrieval_recall_curve.csv")
    curve["recall_pct"] = (curve["recall"] * 100).round(1)
    return curve


@register(
    "p2_retrieval_by_category", "phase2",
    "Retrieval accuracy by benchmark category",
    "Recall@1, Recall@3 and MRR per category. Read against p1_by_category: "
    "retrieval is most accurate on the categories Phase 1 got most wrong.",
    ["data/history/retrieval_by_category.csv"],
    figure_hint="Two-panel or dual-bar figure pairing hallucination rate (Phase 1) "
                "against Recall@1 (Phase 2) for the same categories. This is the "
                "strongest single argument for retrieval and it has no figure yet.",
)
def _p2_retrieval_by_category() -> pd.DataFrame:
    """Return retrieval accuracy per category, sorted by Recall@1."""
    by_category = _require(HISTORY_DIR / "retrieval_by_category.csv")
    return by_category.sort_values("recall@1", ascending=False).reset_index(drop=True)


@register(
    "p2_chunk_ablation", "phase2",
    "Chunk-size ablation",
    "Retrieval accuracy and error-probe coverage at each chunk size. Recall@1 "
    "peaks at 60 words and Recall@3 at 120; probe coverage decided the choice.",
    ["data/history/retrieval_chunk_ablation.csv", "data/history/corpus_chunk_probe_ablation.csv"],
    figure_hint="Two panels sharing an x axis of chunk size: Recall@k on the left, "
                "probes covered on the right.",
)
def _p2_chunk_ablation() -> pd.DataFrame:
    """Join the retrieval ablation to the error-probe coverage at each chunk size."""
    ablation = _require(HISTORY_DIR / "retrieval_chunk_ablation.csv")
    probes = _require(HISTORY_DIR / "corpus_chunk_probe_ablation.csv")
    merged = ablation.merge(probes, on="chunk_size", how="outer").sort_values("chunk_size")
    return merged.reset_index(drop=True)


@register(
    "p2_corpus_composition", "phase2",
    "Corpus composition ablation",
    "Error-probe coverage as crowding sources are removed. Cutting the corpus by "
    "a third improved what it could answer.",
    ["data/history/corpus_composition_ablation.csv"],
    figure_hint="Grouped bar of coverage at k=3 and k=5 per configuration, with "
                "passage count on a secondary axis to show size falling as "
                "coverage rises.",
)
def _p2_corpus_composition() -> pd.DataFrame:
    """Return the composition ablation with coverage expressed out of seven probes."""
    composition = _require(HISTORY_DIR / "corpus_composition_ablation.csv")
    composition["coverage_k3_share"] = (composition["coverage_k3"] / 7).round(3)
    composition["coverage_k5_share"] = (composition["coverage_k5"] / 7).round(3)
    return composition


@register(
    "p2_rag_summary", "phase2",
    "Automated metrics by run, baseline against RAG",
    "Eight runs on the 52-prompt benchmark, 416 responses. Means with 95% "
    "confidence intervals.",
    ["results_thursday/summary_table.csv"],
    figure_hint="Grouped bar per run, baseline against RAG, one panel per metric. "
                "The convergence of the four RAG bars is the point.",
)
def _p2_rag_summary() -> pd.DataFrame:
    """Return the per-run summary with rounding suitable for a report table."""
    summary = _require(THURSDAY_DIR / "summary_table.csv")
    summary = summary.rename(columns={"rag": "retrieval"})
    for column in ("bertscore_mean", "bertscore_ci95", "rougeL_mean", "rougeL_ci95"):
        summary[column] = summary[column].round(4)
    return summary.sort_values(["model", "condition", "retrieval"]).reset_index(drop=True)


@register(
    "p2_rag_paired", "phase2",
    "Paired baseline to RAG comparison",
    "Each prompt answered with and without retrieval, so each model is compared "
    "against itself. Recomputed here from the per-response files rather than "
    "copied from the mid-year report.",
    ["results_thursday/*_with_metrics.csv"],
    figure_hint="Bar of Cohen's dz per run and metric, with significance marked. "
                "Shows a large effect for Llama and none for Mistral.",
)
def _p2_rag_paired() -> pd.DataFrame:
    """Recompute the paired baseline-to-RAG tests from the per-response result files."""
    from scipy import stats

    files = sorted(THURSDAY_DIR.glob("*_with_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_with_metrics.csv files in {THURSDAY_DIR}")

    runs: dict[tuple[str, str, bool], pd.DataFrame] = {}
    for path in files:
        frame = pd.read_csv(path)
        is_rag = "rag" in path.name
        key = (str(frame["model_id"].iloc[0]), str(frame["condition"].iloc[0]), is_rag)
        runs[key] = frame.set_index("prompt_id")

    rows = []
    for (model, condition, is_rag), frame in sorted(runs.items()):
        if is_rag:
            continue
        rag = runs.get((model, condition, True))
        if rag is None:
            continue
        shared = frame.index.intersection(rag.index)
        for metric in ("bertscore_f1", "rouge_l"):
            before = frame.loc[shared, metric].astype(float)
            after = rag.loc[shared, metric].astype(float)
            difference = after.to_numpy() - before.to_numpy()
            t_stat, p_value = stats.ttest_rel(after, before)
            rows.append({
                "model": model,
                "condition": condition,
                "metric": metric,
                "n": len(shared),
                "baseline_mean": round(float(before.mean()), 4),
                "rag_mean": round(float(after.mean()), 4),
                "delta": round(float(difference.mean()), 4),
                "t": round(float(t_stat), 3),
                "dz": round(float(difference.mean() / difference.std(ddof=1)), 3),
                "p_value": float(f"{p_value:.3g}"),
                "significant_at_05": bool(p_value < 0.05),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Phase 3
# --------------------------------------------------------------------------

@register(
    "p2_rubric_by_run", "phase2",
    "Phase 2 rubric results by run",
    "All 416 Phase 2 responses graded blind to model, condition and retrieval "
    "status. Quality is 0 unsafe to 3 strong. The truncation column is the "
    "generation-budget fault described in F-P2-009.",
    ["data/grading/phase2_results_scored.csv"],
    figure_hint="Grouped bar of hallucination rate, baseline against RAG, one "
                "pair per run. The four-run consistency is the point.",
)
def _p2_rubric_by_run() -> pd.DataFrame:
    """Aggregate the Phase 2 rubric to one row per run and retrieval condition."""
    scored = _require(BASE_DIR / "data" / "grading" / "phase2_results_scored.csv")
    scored["truncated"] = scored["note"].str.contains("runcat", case=False, na=False)
    rows = []
    for (model, condition, rag), group in scored.groupby(
        ["model_id", "condition", "rag_enabled"]
    ):
        untruncated = group.loc[~group["truncated"], "quality_score"]
        rows.append({
            "model": model,
            "condition": condition,
            "retrieval": bool(rag),
            "n": len(group),
            "quality_mean": round(float(group["quality_score"].mean()), 3),
            "hallucination_rate": round(float(group["hallucination_flag"].mean()), 4),
            "diagnostic_overreach_rate": round(
                float(group["diagnostic_overreach_flag"].mean()), 4),
            "addresses_question_rate": round(
                float(group["addresses_question_flag"].mean()), 4),
            "professional_followup_rate": round(
                float(group["professional_followup_flag"].mean()), 4),
            "truncated_rate": round(float(group["truncated"].mean()), 4),
            "quality_mean_untruncated": round(float(untruncated.mean()), 3),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition", "retrieval"])


@register(
    "p2_rubric_paired", "phase2",
    "Paired baseline to RAG comparison on the rubric",
    "Each prompt graded with and without retrieval, so each model is compared "
    "against itself. This table is the answer to RQ2.",
    ["data/grading/phase2_rubric_paired.csv"],
    figure_hint="Bar of the hallucination delta per run with significance "
                "marked. All four runs point the same way.",
)
def _p2_rubric_paired() -> pd.DataFrame:
    """Return the paired rubric comparison written by score_rag_rubric.py."""
    return _require(BASE_DIR / "data" / "grading" / "phase2_rubric_paired.csv")


@register(
    "p2_rubric_by_category", "phase2",
    "Phase 2 hallucination by category, baseline against RAG",
    "Retrieval clears the instrument and recognition questions completely while "
    "barely touching the general health questions, which is where retrieval "
    "itself scored worst.",
    ["data/grading/phase2_results_scored.csv"],
    figure_hint="Paired horizontal bars per category, baseline against RAG, "
                "sorted by the size of the reduction. Read against "
                "p2_retrieval_by_category.",
)
def _p2_rubric_by_category() -> pd.DataFrame:
    """Return hallucination rates per category, split by retrieval condition."""
    scored = _require(BASE_DIR / "data" / "grading" / "phase2_results_scored.csv")
    rows = []
    for category, group in scored.groupby("category"):
        base = group[~group["rag_enabled"].astype(bool)]
        rag = group[group["rag_enabled"].astype(bool)]
        rows.append({
            "category": category,
            "n": len(group),
            "baseline_hallucination": round(float(base["hallucination_flag"].mean()), 4),
            "rag_hallucination": round(float(rag["hallucination_flag"].mean()), 4),
            "delta": round(float(rag["hallucination_flag"].mean()
                                 - base["hallucination_flag"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("delta")


@register(
    "p2_retrieval_configs", "phase2",
    "Retrieval configurations compared on answer coverage",
    "Source recall asks whether the cited document was retrieved. Answer "
    "coverage asks whether the retrieved text contains the answer. The two "
    "disagree, and only the second one moves when retrieval is improved "
    "(F-P2-011). All configurations cover the same 52 prompts.",
    ["data/retrieval_eval/experiments.csv"],
    figure_hint="Grouped bar, source recall@5 against coverage@5, one pair per "
                "configuration. The flat recall bars next to the moving "
                "coverage bars are the point.",
)
def _p2_retrieval_configs() -> pd.DataFrame:
    """Return the retrieval experiment registry with the reported columns first."""
    frame = _require(BASE_DIR / "data" / "retrieval_eval" / "experiments.csv")
    columns = ["label", "n_prompts", "recall@5", "coverage@5", "mrr", "note"]
    return frame[[c for c in columns if c in frame.columns]]


@register(
    "p2_truncation_diagnosis", "phase2",
    "Responses stopped by the n-gram constraint, by run",
    "Whether each response ended on a token the repetition constraint had "
    "banned, replayed through the real logits processor against the exact "
    "prompt each run used. Mistral only: Llama 3.1 is a gated repository "
    "(F-P2-010, D-112).",
    ["data/history/truncation_diagnosis.csv"],
    figure_hint="Not a figure. The 2x2 of blocked against graded-truncated is "
                "the whole result and belongs in the text as a table.",
)
def _p2_truncation_diagnosis() -> pd.DataFrame:
    """Aggregate the per-response truncation diagnosis to one row per run."""
    frame = _require(BASE_DIR / "data" / "history" / "truncation_diagnosis.csv")
    rows = []
    for (run, rag), group in frame.groupby(["run", "rag_enabled"]):
        rows.append({
            "run": run.split("_2026")[0].replace("phase1_", ""),
            "retrieval": bool(rag),
            "n": len(group),
            "ended_mid_quotation": round(float(group["was_copying"].mean()), 4),
            "continuation_banned": round(float(group["continuation_banned"].mean()), 4),
            "mean_completion_tokens": round(float(group["completion_tokens"].mean()), 1),
        })
    return pd.DataFrame(rows).sort_values(["run", "retrieval"])


@register(
    "p2_generation_fix", "phase2",
    "Effect of the generation fix, paired per prompt",
    "The 25 August re-run against the Phase 2 runs on the same 52 prompts. "
    "Completeness is automatic and agrees with the human truncation grades on "
    "99.3% of graded responses. BERTScore and ROUGE-L appear from 26 August, "
    "once every run carried a metrics sidecar; they are descriptive only and no "
    "decision rests on them. No rubric measure appears here: this comparison is "
    "of two eras, and only the August pass was graded (F-P2-012).",
    ["data/generation_eval/paired_truncation_fix.csv"],
    figure_hint="Already drawn as p2_generation_fix.png. In text, the "
                "completeness rows for the two Mistral RAG runs are the result.",
)
def _p2_generation_fix() -> pd.DataFrame:
    """Return the paired before-and-after comparison, most-changed first."""
    frame = _require(BASE_DIR / "data" / "generation_eval" /
                     "paired_truncation_fix.csv")
    # evaluate_generation.py became N-way, so the two arms are named rather than
    # called before and after. Presented under the old names here, because that
    # is what this particular comparison is and what the report already calls it.
    frame = frame.rename(columns={"lower_mean": "before", "upper_mean": "after"})
    for column in ("before", "after", "delta"):
        frame[column] = frame[column].round(3)
    columns = ["run", "measure", "n", "before", "after", "delta", "t", "p", "dz"]
    frame = frame[[c for c in columns if c in frame.columns]]
    return frame.sort_values(["measure", "delta"], ascending=[True, False])


RERUN_DIR = BASE_DIR / "data" / "grading_20260825"


def _rerun_scored() -> pd.DataFrame:
    """Graded re-run responses with the annotator note attached."""
    import glob

    scored = _require(RERUN_DIR / "rerun_results_scored.csv")
    files = sorted(glob.glob(str(RERUN_DIR / "annotations" / "batch_*.csv")))
    notes = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    merged = scored.merge(notes[["grade_id", "note"]], on="grade_id",
                          how="left", suffixes=("", "_annotation"))
    merged["rag"] = merged["rag_enabled"].astype(bool)
    return merged


@register(
    "rerun_rubric_by_run", "phase2",
    "Re-run rubric results by run",
    "All 416 responses from results_20260825 graded blind, after the generation "
    "fix, neighbour expansion and the grounded prompt change. Compare against "
    "p2_rubric_by_run, which is the same benchmark before those changes.",
    ["data/grading_20260825/rerun_results_scored.csv"],
    figure_hint="Already drawn as rerun_rubric_comparison.png.",
)
def _rerun_rubric_by_run() -> pd.DataFrame:
    """Aggregate the re-run rubric to one row per run and retrieval condition."""
    scored = _rerun_scored()
    rows = []
    for (model, condition, rag), group in scored.groupby(
        ["model_id", "condition", "rag"]
    ):
        rows.append({
            "model": model, "condition": condition, "retrieval": bool(rag),
            "n": len(group),
            "quality_mean": round(float(group["quality_score"].mean()), 3),
            "hallucination_rate": round(float(group["hallucination_flag"].mean()), 4),
            "diagnostic_overreach_rate": round(
                float(group["diagnostic_overreach_flag"].mean()), 4),
            "addresses_question_rate": round(
                float(group["addresses_question_flag"].mean()), 4),
            "professional_followup_rate": round(
                float(group["professional_followup_flag"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition", "retrieval"])


@register(
    "rerun_rubric_paired", "phase2",
    "Re-run paired baseline to RAG comparison",
    "Each prompt graded with and without retrieval on the 25 August runs. This "
    "is the RQ2 answer after the fixes, and the row to quote is ALL.",
    ["data/grading_20260825/rerun_rubric_paired.csv"],
    figure_hint="Not a figure. The pooled quality and hallucination rows belong "
                "in the text.",
)
def _rerun_rubric_paired() -> pd.DataFrame:
    """Return the paired comparison written by score_rag_rubric.py."""
    return _require(RERUN_DIR / "rerun_rubric_paired.csv")



GENERATION_EVAL = BASE_DIR / "data" / "generation_eval"


@register(
    "followup_conformance", "phase2",
    "Follow-up advice: flat rate against route-appropriate conformance",
    "The flat rate is the measure this replaces, and the first two columns show "
    "why. Baseline and retrieval post the same flat rate while behaving "
    "completely differently: the baseline advises the turns that owe a next step "
    "far more often than the ones that do not, and retrieval advises both alike. "
    "Discrimination is conformance minus the unprompted rate, and its p-value is "
    "a within-arm Welch contrast between the two buckets. Every conformance "
    "figure rests on the 10 of 52 prompts that owe a next step.",
    ["data/grading_20260825/rerun_results_scored.csv",
     "data/route_maps/oracle.csv", "config/prompts.yaml"],
    figure_hint="Paired bars per arm, conformance against unprompted rate, with "
                "the gap between them annotated.",
)
def _followup_conformance() -> pd.DataFrame:
    """Return the conformance summary written by evaluate_followup.py."""
    return _require(GENERATION_EVAL / "followup_conformance_rerun_20260825.csv")


@register(
    "followup_paired", "phase2",
    "Follow-up advice by obligation bucket, retrieval against baseline",
    "Paired per prompt, model and condition. The direction is consistent — "
    "retrieval advises less where a next step is owed and slightly more where "
    "none is — but no individual bucket reaches significance, because the owed "
    "buckets hold 7 and 3 prompts. The supported claim is the discrimination "
    "gap in followup_discrimination, not any single row here.",
    ["data/grading_20260825/rerun_results_scored.csv", "data/route_maps/oracle.csv"],
    figure_hint="Not a figure. A table with the confidence intervals visible is "
                "the honest presentation of an underpowered comparison.",
)
def _followup_paired() -> pd.DataFrame:
    """Return the paired bucket comparison written by evaluate_followup.py."""
    return _require(GENERATION_EVAL / "followup_paired_rerun_20260825.csv")


@register(
    "followup_discrimination", "phase2",
    "Collapse in follow-up targeting under retrieval",
    "The one claim in this group the data supports. Bootstrapped over the 52 "
    "prompts rather than the 416 responses, because four responses to one prompt "
    "are not independent and prompts are what is scarce. The interval excludes "
    "zero where every individual bucket comparison does not.",
    ["data/grading_20260825/rerun_results_scored.csv", "data/route_maps/oracle.csv"],
    figure_hint="Not a figure. One number with an interval belongs in the text.",
)
def _followup_discrimination() -> pd.DataFrame:
    """Return the bootstrapped discrimination gap written by evaluate_followup.py."""
    return _require(GENERATION_EVAL / "followup_discrimination_rerun_20260825.csv")

@register(
    "rerun_error_families", "phase2",
    "Recurring error families by arm",
    "Each family is matched against the annotator's note text, so the counts "
    "regenerate from the tracked annotations rather than being typed by hand. "
    "The instrument-misuse rows are the finding: they are baseline-only.",
    ["data/grading_20260825/annotations/", "data/grading_20260825/rerun_results_scored.csv"],
    figure_hint="Grouped horizontal bars, baseline against RAG per family, "
                "sorted by how lopsided the split is.",
)
def _rerun_error_families() -> pd.DataFrame:
    """Classify graded responses into recurring error families and split by arm."""
    scored = _rerun_scored()
    note = scored["note_annotation"].fillna("")
    hallucinated = scored["hallucination_flag"] == 1

    families = {
        "refusal or non-answer": scored["addresses_question_flag"] == 0,
        "fabricated citation or instrument": note.str.contains(
            "invent|fabricat|does not exist|no such|unverifiable|not verifiable",
            case=False),
        "advises correcting for prematurity": note.str.contains(
            "prematurity", case=False) & hallucinated,
        "says M-CHAT needs a trained administrator": note.str.contains(
            "trained|without training", case=False) & hallucinated,
        "denies the autism-encephalopathy link": note.str.contains(
            "encephalopathy", case=False) & hallucinated,
        "stale 1 in 44 prevalence figure": note.str.contains(
            "1 in 44|superseded", case=False),
        "advises watchful waiting on a positive screen": note.str.contains(
            "adjudication", case=False),
    }
    rows = []
    for name, mask in families.items():
        subset = scored[mask]
        rows.append({
            "error_family": name, "total": len(subset),
            "baseline": int((~subset["rag"]).sum()),
            "rag": int(subset["rag"].sum()),
        })
    frame = pd.DataFrame(rows)
    frame["baseline_share"] = (frame["baseline"] / frame["total"].clip(lower=1)).round(3)
    return frame.sort_values("baseline_share", ascending=False)


@register(
    "p3_experiments", "phase3",
    "Router development steps",
    "Every recorded router evaluation, with its headline figures and its paired "
    "change against the step it was compared with. This table is the progression.",
    ["data/router_eval/experiments.csv"],
    figure_hint="Point plot of accuracy with CI error bars, one point per step, in "
                "the order they were run. Grows as steps are added.",
)
def _p3_experiments() -> pd.DataFrame:
    """Return the experiment registry, trimmed to the columns a report table needs."""
    registry = _require(ROUTER_DIR / "experiments.csv")
    columns = ["label", "config", "n_prompts", "repeats", "folds",
               "accuracy_mean", "accuracy_ci95", "macro_f1_mean", "macro_f1_ci95",
               "safety_recall_mean", "safety_recall_ci95",
               "safety_precision_mean", "safety_precision_ci95",
               "risk_mean", "risk_ci95",
               "rule_share_mean", "compare_to", "delta_accuracy", "delta_p", "delta_dz"]
    return registry[[c for c in columns if c in registry.columns]]


@register(
    "p3_per_route", "phase3",
    "Router performance by route",
    "Precision, recall and F1 per route with a 95% confidence interval on F1, "
    "averaged over ten cross-validation shuffles. Support is the number of "
    "labelled prompts the route has.",
    ["data/router_eval/per_route_baseline.csv"],
    figure_hint="Horizontal bar of F1 with CI error bars, sorted by support, with "
                "n annotated. F1 tracking support is the argument that the weak "
                "routes are a data problem.",
)
def _p3_per_route() -> pd.DataFrame:
    """Return per-route scores sorted by support, smallest first."""
    per_route = _require(ROUTER_DIR / "per_route_baseline.csv")
    return per_route.sort_values("support").reset_index(drop=True)


@register(
    "p3_repeat_spread", "phase3",
    "Router accuracy across cross-validation shuffles",
    "One row per shuffle of the same 93 prompts, with the unchanged router. The "
    "spread is what a single split cannot show.",
    ["data/router_eval/repeats_baseline.csv"],
    figure_hint="Line or strip plot of accuracy per seed, with the mean and the "
                "mid-year single-split value marked as horizontal lines.",
)
def _p3_repeat_spread() -> pd.DataFrame:
    """Return per-shuffle scores plus a summary row of the spread."""
    repeats = _require(ROUTER_DIR / "repeats_baseline.csv").copy()
    summary = pd.DataFrame([{
        "repeat": "summary",
        "seed": "0-9",
        "accuracy": round(float(repeats["accuracy"].mean()), 4),
        "macro_f1": round(float(repeats["macro_f1"].mean()), 4),
        "safety_recall": round(float(repeats["safety_recall"].mean()), 4),
        "safety_precision": round(float(repeats["safety_precision"].mean()), 4),
        "rule_share": round(float(repeats["rule_share"].mean()), 4),
        "note": f"min {repeats['accuracy'].min():.3f}, max {repeats['accuracy'].max():.3f}, "
                f"range {(repeats['accuracy'].max() - repeats['accuracy'].min()) * 100:.1f}pp",
    }])
    repeats["note"] = ""
    return pd.concat([repeats, summary], ignore_index=True)


@register(
    "p3_confusion", "phase3",
    "Router confusion matrix",
    "Counts pooled over ten shuffles. Rows are the true route, columns the "
    "predicted route, so each row sums to the route's support times ten.",
    ["data/router_eval/confusion_baseline.csv"],
    figure_hint="Heatmap normalised by row, with raw counts written in the cells.",
)
def _p3_confusion() -> pd.DataFrame:
    """Return the pooled confusion matrix with an explicit true-route column."""
    matrix = _require(ROUTER_DIR / "confusion_baseline.csv")
    return matrix.rename(columns={"Unnamed: 0": "true_route"})


@register(
    "p3_cascade_threshold_sweep", "phase3",
    "Cascade safety-gate threshold sweep",
    "The two-stage cascade at nine safety-gate thresholds, on both label tracks. "
    "Shows the precision-recall trade the gate offers and why no setting of it "
    "beats the flat router.",
    ["data/router_eval/cascade_threshold_sweep.csv"],
    figure_hint="Two panels, one per track: safety recall and safety precision "
                "against threshold on the left axis, risk on the right. The "
                "absence of a threshold that minimises risk below the flat "
                "router's is the finding.",
)
def _p3_cascade_threshold_sweep() -> pd.DataFrame:
    """Return the cascade threshold sweep for both tracks."""
    return _require(ROUTER_DIR / "cascade_threshold_sweep.csv")


@register(
    "p3_rule_shift", "phase3",
    "Rule layer under distribution shift",
    "Whole families of safety prompts removed from training, then tested on the "
    "family that was removed. The split is deterministic, so these are exact "
    "counts with no confidence interval.",
    ["data/router_eval/rule_distribution_shift.csv"],
    figure_hint="Grouped bar of safety turns reached, with and without the rule "
                "layer, one group per scenario. The contrast between 18 of 18 and "
                "4 of 18 is the figure.",
)
def _p3_rule_shift() -> pd.DataFrame:
    """Return the distribution-shift results for the rule layer."""
    return _require(ROUTER_DIR / "rule_distribution_shift.csv")


@register(
    "p3_adversarial_holdout", "phase3",
    "Held-out adversarial set",
    "Twenty requests for a diagnostic judgement phrased to avoid the rule "
    "vocabulary, never trained on. Rules catch 1 of 20, the adopted router 13.",
    ["data/router_eval/adversarial_holdout.csv"],
    figure_hint="Bar of safety turns reached by component, against the same "
                "components on the benchmark set. The inversion between the two "
                "settings is the figure.",
)
def _p3_adversarial_holdout() -> pd.DataFrame:
    """Return the held-out adversarial results."""
    return _require(ROUTER_DIR / "adversarial_holdout.csv")


@register(
    "p3_hard_prompts", "phase3",
    "Prompts the router never routes correctly",
    "Prompts with a correctness rate of zero across all ten shuffles, with the "
    "route they are always sent to instead. A consistent, confident disagreement "
    "is evidence about the label, not only about the router.",
    ["data/router_eval/predictions_baseline.csv"],
    figure_hint="A table, not a figure. Supports the planned label audit.",
)
def _p3_hard_prompts() -> pd.DataFrame:
    """Return every prompt the router never gets right, with its modal prediction."""
    predictions = _require(ROUTER_DIR / "predictions_baseline.csv")
    hard = predictions[predictions["correct_rate"] == 0].copy()
    return hard[["prompt_id", "category", "route", "modal_prediction",
                 "prediction_spread", "source_file", "prompt"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> list[str]:
    """Render a dataframe as a markdown table, truncating very long ones."""
    shown = frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    divider = "|" + "|".join("---" for _ in shown.columns) + "|"
    lines = [header, divider]
    for _, row in shown.iterrows():
        cells = []
        for value in row:
            text = "" if pd.isna(value) else str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) > max_rows:
        lines.append(f"| _...{len(frame) - max_rows} further rows in the CSV_ |"
                     + "|" * (len(shown.columns) - 1))
    return lines


def write_numbers_document(built: list[str], out: Path = NUMBERS_MD) -> Path:
    """Write docs/report/NUMBERS.md containing every exported table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Report numbers",
        "",
        "Generated by `python scripts/export_report_numbers.py`. Do not edit by hand.",
        "",
        "Every table here is computed from tracked data. Each one is also written as",
        "a CSV in `numbers/`, in a tidy shape suitable for a spreadsheet or a plotting",
        "script, so a figure can be built from the numbers rather than traced from a",
        "picture. The figure hint on each table says what it is worth drawing and why.",
        "",
        "See [`../DATA_GUIDE.md`](../DATA_GUIDE.md) for what each dataset is, where it",
        "came from, and how it should be read.",
        "",
    ]
    for group in ("phase1", "phase2", "phase3"):
        group_tables = [t for t in TABLES.values() if t.group == group and t.name in built]
        if not group_tables:
            continue
        lines += [f"## {group}", ""]
        for table in group_tables:
            frame = pd.read_csv(OUT_DIR / f"{table.name}.csv")
            lines += [
                f"### {table.title}",
                "",
                table.caption,
                "",
                f"- CSV: [`numbers/{table.name}.csv`](numbers/{table.name}.csv)",
                f"- Built from: {', '.join(f'`{s}`' for s in table.sources)}",
                f"- Rebuild: `python scripts/export_report_numbers.py --table {table.name}`",
            ]
            if table.figure_hint:
                lines.append(f"- Figure hint: {table.figure_hint}")
            lines += [""] + _markdown_table(frame) + [""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    """Parse arguments and export the requested tables."""
    parser = argparse.ArgumentParser(description="Export the report's numbers.")
    parser.add_argument("--table", help="export one table by name")
    parser.add_argument("--group", help="export one group: phase1, phase2, phase3")
    parser.add_argument("--list", action="store_true", help="list registered tables")
    args = parser.parse_args()

    if args.list:
        for name, table in TABLES.items():
            logger.info("  %-28s [%s] %s", name, table.group, table.title)
        return

    if args.table:
        if args.table not in TABLES:
            raise ValueError(f"Unknown table '{args.table}'. Available: {sorted(TABLES)}")
        selected = [TABLES[args.table]]
    elif args.group:
        selected = [t for t in TABLES.values() if t.group == args.group]
        if not selected:
            raise ValueError(f"No tables in group '{args.group}'")
    else:
        selected = list(TABLES.values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built, skipped = [], []
    for table in selected:
        try:
            frame = table.build()
            frame.to_csv(OUT_DIR / f"{table.name}.csv", index=False)
            built.append(table.name)
            logger.info("Exported %-28s %d rows", table.name, len(frame))
        except FileNotFoundError as exc:
            skipped.append(table.name)
            logger.warning("Skipped %s: %s", table.name, exc)

    # The document always lists every table that has a CSV on disk, so exporting
    # one table does not drop the others from it.
    existing = [name for name in TABLES if (OUT_DIR / f"{name}.csv").exists()]
    write_numbers_document(existing)
    logger.info("Exported %d table(s) into %s", len(built), OUT_DIR)
    if skipped:
        logger.warning("Skipped for missing inputs: %s", ", ".join(skipped))


if __name__ == "__main__":
    main()
