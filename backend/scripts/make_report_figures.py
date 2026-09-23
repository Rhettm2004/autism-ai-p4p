"""
Figure generation for the final report.

Every figure the report uses is built here from tracked data, so a figure can
never quietly disagree with the numbers it is supposed to show. Each figure is
registered with a caption and the files it reads, and the registry is written out
as docs/figures/MANIFEST.md, which is what the report cites.

Usage examples:

  # Build every figure
  python scripts/make_report_figures.py

  # Build one group, or one figure
  python scripts/make_report_figures.py --group router
  python scripts/make_report_figures.py --figure router_single_split_noise

  # List what is registered without building anything
  python scripts/make_report_figures.py --list

Style follows the report template: serif type, 10 point minimum in figures, solid
fills only, and a width that fits the 170 mm print area.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
HISTORY_DIR = BASE_DIR / "data" / "history"
ROUTER_DIR = BASE_DIR / "data" / "router_eval"
GENERATION_EVAL = BASE_DIR / "data" / "generation_eval"
THURSDAY_DIR = BASE_DIR / "results_thursday"
FIGURE_DIR = BASE_DIR / "docs" / "figures"

# The single accuracy figure the mid-year report quoted, from one 4-fold split
# with seed 42. Kept as a constant so the comparison against it is explicit.
MIDYEAR_SINGLE_SPLIT_ACCURACY = 0.656

PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#c55a11",
    "accent": "#548235",
    "muted": "#7f7f7f",
    "light": "#bdd7ee",
    "warn": "#a5342c",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


# Figures are written into one folder per stage of the project, numbered so a
# file browser sorts them in the order the work happened. A flat directory
# became genuinely confusing once the same benchmark had been graded twice:
# p2_hallucination_by_category and rerun_hallucination_by_category are the same
# view of two different passes, and only the folder makes that obvious.
#
# Each entry is: folder name, what the folder holds, and when the data under it
# was produced.
FOLDERS = {
    "overview": (
        "00_overview",
        "Schematics of the system and of how its numbers are produced.",
        "Drawn 25 August 2026",
    ),
    "phase1_baseline": (
        "01_phase1_baseline",
        "The ungrounded baseline: what the models did with no retrieval and no "
        "repetition control.",
        "Runs July 2026, graded by hand",
    ),
    "phase2_retrieval": (
        "02_phase2_retrieval",
        "The retrieval layer measured on its own, before any generation. Chunking, "
        "corpus composition and per-category recall.",
        "Measured August 2026",
    ),
    "phase2_generation_jul": (
        "03_phase2_generation_jul",
        "The first graded comparison of baseline against retrieval, on the 23 July "
        "runs. Superseded as the headline by folder 05, kept because the report "
        "describes the progression.",
        "Runs 23 July 2026, graded 24 August 2026",
    ),
    "generation_fix": (
        "04_generation_fix",
        "The truncation fault and its repair, measured mechanically rather than by "
        "rubric.",
        "Diagnosed and fixed 24-25 August 2026",
    ),
    "phase2_rerun_aug": (
        "05_phase2_rerun_aug",
        "The re-run after the three fixes, graded blind. This is the pass the "
        "report should quote for the retrieval effect.",
        "Runs and grading 25 August 2026",
    ),
    "phase3_router": (
        "06_phase3_router",
        "The agentic routing layer: measurement protocol, per-route performance and "
        "the progression across changes.",
        "August 2026, ongoing",
    ),
}


def folder_for(group: str) -> str:
    """Return the output folder for a group, failing loudly on an unknown one."""
    if group not in FOLDERS:
        raise ValueError(
            f"Group '{group}' has no folder. Add it to FOLDERS, or use one of: "
            f"{sorted(FOLDERS)}"
        )
    return FOLDERS[group][0]


@dataclass
class Figure:
    """One registered report figure: how to build it, and where its numbers come from."""

    name: str
    group: str
    caption: str
    sources: list[str]
    build: Callable[[Path], None]
    notes: str = ""


FIGURES: dict[str, Figure] = {}


def register(name: str, group: str, caption: str, sources: list[str], notes: str = ""):
    """Register a figure builder under a name usable from the command line."""
    def decorator(func: Callable[[Path], None]) -> Callable[[Path], None]:
        if name in FIGURES:
            raise ValueError(f"Figure '{name}' is already registered")
        FIGURES[name] = Figure(name=name, group=group, caption=caption,
                               sources=sources, build=func, notes=notes)
        return func
    return decorator


def _require(path: Path) -> pd.DataFrame:
    """Load a required input, failing with the path rather than a parser error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Figure input missing: {path}. It is produced by an earlier stage; "
            f"see data/history/README.md or run the router evaluation first."
        )
    return pd.read_csv(path)


def _label_bars(ax, bars, values, fmt="{:.0%}", offset=0.012) -> None:
    """Write each bar's value above it, so the figure is readable without the axis."""
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                fmt.format(value), ha="center", va="bottom", fontsize=9)


# --------------------------------------------------------------------------
# Router figures, from data/router_eval
# --------------------------------------------------------------------------

@register(
    "router_single_split_noise", "phase3_router",
    "Router accuracy across ten cross-validation shuffles of the same 93 prompts. "
    "The single split reported at mid-year sits below the distribution it was drawn from.",
    ["data/router_eval/repeats_baseline.csv"],
    notes="Motivates the change from single-split to repeated cross-validation.",
)
def _router_single_split_noise(out: Path) -> None:
    """Plot per-repeat accuracy against the mid-year single-split figure."""
    repeats = _require(ROUTER_DIR / "repeats_baseline.csv")
    accuracy = repeats["accuracy"].to_numpy()
    mean = accuracy.mean()

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(repeats["seed"], accuracy * 100, "o-", color=PALETTE["primary"],
            markersize=5, linewidth=1.2, label="Repeated CV, one point per shuffle")
    ax.axhline(mean * 100, color=PALETTE["primary"], linestyle="--", linewidth=1,
               label=f"Mean {mean:.1%}")
    ax.axhline(MIDYEAR_SINGLE_SPLIT_ACCURACY * 100, color=PALETTE["warn"],
               linestyle=":", linewidth=1.4,
               label=f"Mid-year single split {MIDYEAR_SINGLE_SPLIT_ACCURACY:.1%}")
    ax.set_xlabel("Cross-validation seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(list(repeats["seed"]))
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out)
    plt.close(fig)


@register(
    "router_per_route_f1", "phase3_router",
    "Router F1 by route with 95% confidence intervals, against the number of "
    "labelled examples each route has. Performance tracks training data, not route difficulty.",
    ["data/router_eval/per_route_baseline.csv"],
    notes="Supports the argument that the weak routes are a data problem.",
)
def _router_per_route_f1(out: Path) -> None:
    """Plot per-route F1 with CIs, annotated with each route's support."""
    per_route = _require(ROUTER_DIR / "per_route_baseline.csv").sort_values("support")
    labels = [r.replace("_", " ") for r in per_route["route"]]
    positions = np.arange(len(per_route))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(positions, per_route["f1"], xerr=per_route["f1_ci95"],
            color=PALETTE["primary"], height=0.62,
            error_kw={"ecolor": PALETTE["muted"], "elinewidth": 1, "capsize": 3})
    for y, (f1, support) in enumerate(zip(per_route["f1"], per_route["support"])):
        ax.text(max(f1, 0.02) + 0.03, y, f"n = {support}", va="center", fontsize=9,
                color=PALETTE["muted"])
    ax.set_yticks(positions, labels)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.05)
    fig.savefig(out)
    plt.close(fig)


@register(
    "router_confusion", "phase3_router",
    "Router confusion matrix pooled over ten cross-validation shuffles. "
    "Rows are the true route, columns the predicted route.",
    ["data/router_eval/confusion_baseline.csv"],
)
def _router_confusion(out: Path) -> None:
    """Draw the pooled confusion matrix as a labelled heatmap."""
    matrix = _require(ROUTER_DIR / "confusion_baseline.csv").set_index("Unnamed: 0")
    matrix.index.name = None
    normalised = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    labels = [r.replace("_", " ") for r in matrix.index]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix.iloc[i, j])
            if count:
                ax.text(j, i, count, ha="center", va="center", fontsize=9,
                        color="white" if normalised.iloc[i, j] > 0.55 else "black")
    ax.set_xlabel("Predicted route")
    ax.set_ylabel("True route")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Share of the true route's turns")
    fig.savefig(out)
    plt.close(fig)


@register(
    "router_taxonomy_comparison", "phase3_router",
    "Effect of separating misinformation correction from safety deflection, "
    "re-measured under the repeated protocol. The split buys safety recall, not accuracy.",
    ["data/router_eval/experiments.csv"],
    notes="Replaces the mid-year claim of 62.4% to 72.0%, which compared two label spaces.",
)
def _router_taxonomy_comparison(out: Path) -> None:
    """Compare the pre-split and current taxonomies on accuracy and safety recall."""
    registry = _require(ROUTER_DIR / "experiments.csv").set_index("label")
    for required in ("taxonomy_v1", "baseline"):
        if required not in registry.index:
            raise FileNotFoundError(
                f"Experiment '{required}' is not in the registry. Run it with "
                f"scripts/evaluate_router.py before building this figure."
            )
    steps = [("taxonomy_v1", "Pre-split taxonomy\n(July 2026)"),
             ("baseline", "Current taxonomy")]
    metrics = [("accuracy", "Accuracy"), ("safety_recall", "Safety-route recall")]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.34
    positions = np.arange(len(metrics))
    for offset, (label, display), colour in zip(
        (-width / 2, width / 2), steps, (PALETTE["muted"], PALETTE["primary"])
    ):
        values = [registry.loc[label, f"{m}_mean"] for m, _ in metrics]
        errors = [registry.loc[label, f"{m}_ci95"] for m, _ in metrics]
        bars = ax.bar(positions + offset, values, width, yerr=errors, label=display,
                      color=colour, error_kw={"ecolor": "#333333", "elinewidth": 1, "capsize": 3})
        _label_bars(ax, bars, values, offset=0.03)
    ax.set_xticks(positions, [display for _, display in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def _step_name(label: str) -> str:
    """Strip the track and dataset suffixes to get the change a row measured."""
    for suffix in ("_ext",):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    for suffix in ("_audited", "_topic"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label


# How each recorded step is named on the axis. Two lines each, so the labels
# stay horizontal and readable at 10 point.
STEP_LABELS = {
    "baseline": "flat 7-way\nbaseline",
    "cascade": "two-stage\ncascade",
    "char_ngrams": "word+char\nn-grams",
    "char_only": "char\nonly",
    "embeddings": "MiniLM\nembeddings",
    "embeddings_word": "embeddings\n+ word",
    "router_v2": "+ extended\nrules",
}

TRACK_STYLE = {
    "audited": {"colour": "primary", "dash": "-", "marker": "o",
                "label": "Audited labels (primary track)"},
    "topic": {"colour": "secondary", "dash": "--", "marker": "s",
              "label": "Topic labels (secondary track)"},
}


@register(
    "router_progression", "phase3_router",
    "Router accuracy at each recorded change, one line per ground-truth track "
    "and one panel per dataset, with 95% confidence intervals. The tracks are "
    "two scorings of the same router, not two steps.",
    ["data/router_eval/experiments.csv"],
    notes="Regenerate after every recorded experiment; this is the progression figure.",
)
def _router_progression(out: Path) -> None:
    """
    Plot accuracy per recorded change, one line per ground-truth track.

    The two tracks are separate y values for the same router, not steps in a
    sequence, so they get a line each. Drawing one line through every registry
    row in order, as this figure did until 25 August 2026, alternated between
    them and produced a zigzag that read as instability.

    The two datasets are separate panels for the same reason: the extended set
    adds 70 router-only prompts, so a point on one side is not comparable with a
    point on the other.
    """
    registry = _require(ROUTER_DIR / "experiments.csv")
    registry["step"] = registry["label"].map(_step_name)

    # The first three rows used 4 folds and, for taxonomy_v1, a third label
    # space. They are kept in the registry as the historical record but cannot
    # share an axis with the 3-fold results.
    plotted = registry[(registry["folds"] == 3) & registry["track"].isin(TRACK_STYLE)]

    panels = [
        ("Core set, 93 prompts", plotted[plotted["dataset"].isna()]),
        ("Extended set, 163 prompts", plotted[plotted["dataset"] == "extended"]),
    ]
    widths = [max(len(part["step"].unique()), 1) for _, part in panels]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.9), sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.08},
    )

    for ax, (title, part) in zip(axes, panels):
        # Run order, so the axis reads left to right as the work happened.
        steps = list(dict.fromkeys(part.sort_values("timestamp")["step"]))
        positions = {step: i for i, step in enumerate(steps)}

        for track, style in TRACK_STYLE.items():
            rows = part[part["track"] == track].copy()
            rows["x"] = rows["step"].map(positions)
            rows = rows.sort_values("x")
            if rows.empty:
                continue
            ax.errorbar(
                rows["x"], rows["accuracy_mean"] * 100,
                yerr=rows["accuracy_ci95"] * 100,
                fmt=style["marker"] + style["dash"], color=PALETTE[style["colour"]],
                markersize=5, linewidth=1.4, capsize=3, ecolor=PALETTE["muted"],
                label=style["label"] if ax is axes[0] else None,
            )

        ax.axhline(MIDYEAR_SINGLE_SPLIT_ACCURACY * 100, color=PALETTE["warn"],
                   linestyle=":", linewidth=1.2,
                   label=f"Mid-year single split, {MIDYEAR_SINGLE_SPLIT_ACCURACY:.1%}"
                   if ax is axes[0] else None)
        ax.set_xticks(range(len(steps)),
                      [STEP_LABELS.get(s, s.replace("_", "\n")) for s in steps],
                      fontsize=8)
        ax.set_xlim(-0.55, len(steps) - 0.45)
        ax.set_title(title, fontsize=10)

    axes[0].set_ylabel("Routing accuracy (%)")
    axes[0].set_ylim(60, 92)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Phase 1 figures, from recovered rubric data
# --------------------------------------------------------------------------

def _run_label(frame: pd.DataFrame) -> pd.Series:
    """Return a readable run label from the model and condition columns."""
    model = frame["model_name"] if "model_name" in frame.columns else frame["model_id"]
    return model.str.replace("-", "-", regex=False) + "\n" + frame["condition"].str.replace("_", "-")


@register(
    "phase1_quality_distribution", "phase1_baseline",
    "Distribution of manual rubric quality scores by run (n = 44 per run). "
    "Scores are 0 unsafe to 3 strong.",
    ["data/history/phase1_results_scored.csv"],
)
def _phase1_quality_distribution(out: Path) -> None:
    """Plot the stacked distribution of rubric quality scores for each run."""
    scored = _require(HISTORY_DIR / "phase1_results_scored.csv")
    scored["run"] = _run_label(scored)
    counts = (scored.groupby(["run", "quality_score"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1, 2, 3], fill_value=0))
    colours = [PALETTE["warn"], PALETTE["secondary"], PALETTE["light"], PALETTE["accent"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bottom = np.zeros(len(counts))
    for score, colour in zip([0, 1, 2, 3], colours):
        values = counts[score].to_numpy()
        ax.bar(counts.index, values, bottom=bottom, color=colour, width=0.62,
               label=f"{score}")
        bottom += values
    ax.set_ylabel("Responses")
    ax.legend(title="Quality", frameon=False, ncols=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    fig.savefig(out)
    plt.close(fig)


@register(
    "phase1_hallucination_by_category", "phase1_baseline",
    "Hallucination rate by prompt category, all four Phase 1 runs pooled (n = 176). "
    "Errors concentrate on facts about the screening instruments.",
    ["data/history/phase1_results_scored.csv"],
)
def _phase1_hallucination_by_category(out: Path) -> None:
    """Plot hallucination rate per category, annotated with each category's size."""
    scored = _require(HISTORY_DIR / "phase1_results_scored.csv")
    grouped = (scored.groupby("category")["hallucination_flag"]
               .agg(["mean", "size"]).sort_values("mean"))
    labels = [c.replace("_", " ") for c in grouped.index]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(np.arange(len(grouped)), grouped["mean"], color=PALETTE["secondary"], height=0.62)
    for y, (rate, size) in enumerate(zip(grouped["mean"], grouped["size"])):
        ax.text(rate + 0.012, y, f"n = {size}", va="center", fontsize=9, color=PALETTE["muted"])
    ax.set_yticks(np.arange(len(grouped)), labels)
    ax.set_xlabel("Hallucination rate")
    ax.set_xlim(0, max(0.6, float(grouped["mean"].max()) + 0.12))
    fig.savefig(out)
    plt.close(fig)


@register(
    "phase1_bertscore_vs_quality", "phase1_baseline",
    "BERTScore F1 against manual rubric quality for all 176 Phase 1 responses. "
    "The flat trend shows semantic similarity does not track clinical safety.",
    ["data/history/phase1_results_scored.csv"],
)
def _phase1_bertscore_vs_quality(out: Path) -> None:
    """Scatter BERTScore against rubric quality with a fitted trend line."""
    scored = _require(HISTORY_DIR / "phase1_results_scored.csv").dropna(
        subset=["bertscore_f1", "quality_score"]
    )
    x = scored["quality_score"].to_numpy(dtype=float)
    y = scored["bertscore_f1"].to_numpy(dtype=float)
    jitter = np.random.default_rng(0).normal(0, 0.06, size=len(x))
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.scatter(x + jitter, y, s=16, alpha=0.55, color=PALETTE["primary"],
               edgecolors="none")
    grid = np.linspace(-0.2, 3.2, 50)
    ax.plot(grid, slope * grid + intercept, color=PALETTE["warn"], linewidth=1.4,
            label=f"Least squares fit, r = {correlation:.3f}")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Rubric quality score")
    ax.set_ylabel("BERTScore F1")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


@register(
    "phase1_degeneracy", "phase1_baseline",
    "Repetition in the 176 Phase 1 responses generated without repetition control, "
    "by run. Llama3-8B repeats far more than Mistral-7B in the zero-shot condition.",
    ["data/history/degeneration_metrics.csv"],
    notes="This is the baseline that motivated repetition_penalty 1.15 and no_repeat_ngram_size 6.",
)
def _phase1_degeneracy(out: Path) -> None:
    """Plot repeated-token fraction and repeated-5-gram share for each run."""
    metrics = _require(HISTORY_DIR / "degeneration_metrics.csv")
    metrics["run"] = _run_label(metrics)
    metrics["has_repeat"] = (metrics["max_ngram_rep"] > 1).astype(int)
    grouped = metrics.groupby("run").agg(
        rep_token_frac=("rep_token_frac", "mean"),
        has_repeat=("has_repeat", "mean"),
        degenerate=("degenerate", "sum"),
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    positions = np.arange(len(grouped))
    width = 0.36
    bars_a = ax.bar(positions - width / 2, grouped["has_repeat"], width,
                    color=PALETTE["primary"], label="Responses with a repeated 5-gram")
    bars_b = ax.bar(positions + width / 2, grouped["rep_token_frac"], width,
                    color=PALETTE["secondary"], label="Mean repeated-token fraction")
    _label_bars(ax, bars_a, grouped["has_repeat"], offset=0.008)
    _label_bars(ax, bars_b, grouped["rep_token_frac"], offset=0.008)
    ax.set_xticks(positions, grouped.index)
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(0.6, float(grouped["has_repeat"].max()) + 0.15))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Phase 2 figures, from recovered retrieval and RAG data
# --------------------------------------------------------------------------

@register(
    "retrieval_chunk_ablation", "phase2_retrieval",
    "Retrieval accuracy against chunk size. Recall@1 peaks at 60-word chunks while "
    "Recall@3 peaks at 120, which is why the corpus uses 120.",
    ["data/history/retrieval_chunk_ablation.csv", "data/history/corpus_chunk_probe_ablation.csv"],
)
def _retrieval_chunk_ablation(out: Path) -> None:
    """Plot Recall@k against chunk size beside error-probe coverage."""
    ablation = _require(HISTORY_DIR / "retrieval_chunk_ablation.csv")
    probes = _require(HISTORY_DIR / "corpus_chunk_probe_ablation.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 3.0))
    for column, colour, marker in (("recall@1", PALETTE["primary"], "o"),
                                   ("recall@3", PALETTE["secondary"], "s"),
                                   ("recall@5", PALETTE["accent"], "^")):
        left.plot(ablation["chunk_size"], ablation[column] * 100, marker=marker,
                  color=colour, linewidth=1.2, markersize=5, label=f"Recall@{column[-1]}")
    left.set_xlabel("Chunk size (words)")
    left.set_ylabel("Recall (%)")
    left.legend(frameon=False, loc="lower right")

    right.plot(probes["chunk_size"], probes["probes_covered_k5"], "o-",
               color=PALETTE["primary"], linewidth=1.2, markersize=5, label="Covered at k=5")
    right.plot(probes["chunk_size"], probes["probes_covered_k3"], "s--",
               color=PALETTE["secondary"], linewidth=1.2, markersize=5, label="Covered at k=3")
    right.set_xlabel("Chunk size (words)")
    right.set_ylabel("Phase 1 error probes covered")
    right.set_ylim(0, 7.5)
    right.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


@register(
    "corpus_composition_ablation", "phase2_retrieval",
    "Error-probe coverage as crowding sources are removed from the corpus. "
    "Cutting the corpus by two thirds improved what it could answer.",
    ["data/history/corpus_composition_ablation.csv"],
    notes="The counter-intuitive result: a smaller corpus retrieved better.",
)
def _corpus_composition_ablation(out: Path) -> None:
    """Plot probe coverage and passage count across corpus configurations."""
    composition = _require(HISTORY_DIR / "corpus_composition_ablation.csv")
    positions = np.arange(len(composition))
    labels = [c.replace(" + ", "\n+ ").replace("drop ", "drop\n") for c in composition["config"]]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    width = 0.36
    bars_a = ax.bar(positions - width / 2, composition["coverage_k3"], width,
                    color=PALETTE["secondary"], label="Probes covered at k=3")
    bars_b = ax.bar(positions + width / 2, composition["coverage_k5"], width,
                    color=PALETTE["primary"], label="Probes covered at k=5")
    _label_bars(ax, bars_a, composition["coverage_k3"], fmt="{:.0f}", offset=0.06)
    _label_bars(ax, bars_b, composition["coverage_k5"], fmt="{:.0f}", offset=0.06)
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_ylabel("Error probes covered (of 7)")
    ax.set_ylim(0, 8)

    passages = ax.twinx()
    passages.plot(positions, composition["passages"], "o--", color=PALETTE["muted"],
                  linewidth=1.2, markersize=5, label="Passages in corpus")
    passages.set_ylabel("Passages")
    passages.grid(False)
    handles = ax.get_legend_handles_labels()[0] + passages.get_legend_handles_labels()[0]
    labels_all = ax.get_legend_handles_labels()[1] + passages.get_legend_handles_labels()[1]
    ax.legend(handles, labels_all, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.2), ncols=3)
    fig.savefig(out)
    plt.close(fig)


@register(
    "retrieval_by_category", "phase2_retrieval",
    "Retrieval Recall@1 by benchmark category. Retrieval is strongest on the "
    "instrument and boundary questions where the models hallucinated most.",
    ["data/history/retrieval_by_category.csv"],
)
def _retrieval_by_category(out: Path) -> None:
    """Plot Recall@1 per category, annotated with the number of prompts."""
    by_category = _require(HISTORY_DIR / "retrieval_by_category.csv").sort_values("recall@1")
    labels = [c.replace("_", " ") for c in by_category["category"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(np.arange(len(by_category)), by_category["recall@1"],
            color=PALETTE["primary"], height=0.62)
    for y, (recall, n) in enumerate(zip(by_category["recall@1"], by_category["n"])):
        ax.text(recall + 0.02, y, f"n = {n}", va="center", fontsize=9, color=PALETTE["muted"])
    ax.set_yticks(np.arange(len(by_category)), labels)
    ax.set_xlabel("Recall@1")
    ax.set_xlim(0, 1.18)
    fig.savefig(out)
    plt.close(fig)


@register(
    "phase2_rag_effect", "phase2_retrieval",
    "Automated similarity with and without retrieval, by run (n = 52, 95% CIs). "
    "Retrieval lifts every run and pulls the four RAG runs together.",
    ["results_thursday/summary_table.csv"],
)
def _phase2_rag_effect(out: Path) -> None:
    """Plot baseline against RAG BERTScore and ROUGE-L for each of the four runs."""
    summary = _require(THURSDAY_DIR / "summary_table.csv")
    summary["run"] = (summary["model"] + "\n" + summary["condition"].str.replace("_", "-"))
    runs = sorted(summary["run"].unique())
    positions = np.arange(len(runs))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))
    for ax, (mean_col, ci_col, title) in zip(
        axes,
        [("bertscore_mean", "bertscore_ci95", "BERTScore F1"),
         ("rougeL_mean", "rougeL_ci95", "ROUGE-L F1")],
    ):
        for offset, is_rag, colour, label in (
            (-width / 2, False, PALETTE["muted"], "Baseline"),
            (width / 2, True, PALETTE["primary"], "RAG"),
        ):
            subset = summary[summary["rag"] == is_rag].set_index("run").loc[runs]
            ax.bar(positions + offset, subset[mean_col], width, yerr=subset[ci_col],
                   color=colour, label=label,
                   error_kw={"ecolor": "#333333", "elinewidth": 1, "capsize": 2.5})
        ax.set_xticks(positions, runs, fontsize=8.5)
        ax.set_title(title)
    axes[0].set_ylim(0.80, 0.90)
    axes[1].set_ylim(0, 0.30)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Phase 2 rubric figures, from the blind grading pass
# --------------------------------------------------------------------------

GRADING_DIR = BASE_DIR / "data" / "grading"


def _load_scored() -> pd.DataFrame:
    """Load the graded Phase 2 responses with a truncation flag attached."""
    scored = _require(GRADING_DIR / "phase2_results_scored.csv")
    scored["truncated"] = scored["note"].str.contains("runcat", case=False, na=False)
    scored["rag"] = scored["rag_enabled"].astype(bool)
    scored["run"] = (scored["model_id"] + "\n"
                     + scored["condition"].str.replace("_", "-", regex=False))
    return scored


@register(
    "p2_hallucination_by_run", "phase2_generation_jul",
    "Hallucination rate before and after retrieval, by run (n = 52 per bar). "
    "Every run falls, and every fall is significant on a paired test.",
    ["data/grading/phase2_results_scored.csv"],
    notes="The headline Phase 2 safety result. Answers RQ2.",
)
def _p2_hallucination_by_run(out: Path) -> None:
    """Grouped bars of hallucination rate, baseline against RAG, per run and pooled."""
    scored = _load_scored()
    runs = sorted(scored["run"].unique())
    base = [scored[(scored["run"] == r) & ~scored["rag"]]["hallucination_flag"].mean()
            for r in runs]
    rag = [scored[(scored["run"] == r) & scored["rag"]]["hallucination_flag"].mean()
           for r in runs]
    labels = runs + ["ALL\npooled"]
    base.append(scored[~scored["rag"]]["hallucination_flag"].mean())
    rag.append(scored[scored["rag"]]["hallucination_flag"].mean())

    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    bars_a = ax.bar(positions - width / 2, base, width, color=PALETTE["warn"],
                    label="Baseline, no retrieval")
    bars_b = ax.bar(positions + width / 2, rag, width, color=PALETTE["primary"],
                    label="With retrieval")
    _label_bars(ax, bars_a, base, offset=0.012)
    _label_bars(ax, bars_b, rag, offset=0.012)
    for x, (b, r) in enumerate(zip(base, rag)):
        ax.text(x, max(b, r) + 0.075, f"{(r - b) * 100:+.0f}pp", ha="center",
                fontsize=9.5, color=PALETTE["accent"], fontweight="bold")
    ax.axvline(len(runs) - 0.5, color=PALETTE["muted"], lw=0.8, ls=":")
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_ylabel("Responses containing a false claim")
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out)
    plt.close(fig)


@register(
    "p2_hallucination_by_category", "phase2_generation_jul",
    "Hallucination rate by prompt category, before and after retrieval. Retrieval "
    "clears the instrument and recognition questions and barely touches general "
    "health knowledge.",
    ["data/grading/phase2_results_scored.csv"],
    notes="Read against retrieval_by_category: retrieval helps where retrieval works.",
)
def _p2_hallucination_by_category(out: Path) -> None:
    """Dumbbell plot of per-category hallucination, baseline against RAG."""
    scored = _load_scored()
    rows = []
    for category, group in scored.groupby("category"):
        rows.append({
            "category": category.replace("_", " "),
            "n": len(group),
            "base": group[~group["rag"]]["hallucination_flag"].mean(),
            "rag": group[group["rag"]]["hallucination_flag"].mean(),
        })
    frame = pd.DataFrame(rows).sort_values("base")
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    for y, (_, row) in zip(positions, frame.iterrows()):
        colour = PALETTE["accent"] if row["rag"] <= row["base"] else PALETTE["warn"]
        ax.plot([row["base"], row["rag"]], [y, y], color=colour, lw=2.2, zorder=1,
                solid_capstyle="round")
    ax.scatter(frame["base"], positions, s=78, color=PALETTE["warn"], zorder=3,
               label="Baseline", edgecolors="white", linewidths=0.8)
    ax.scatter(frame["rag"], positions, s=52, color=PALETTE["primary"], zorder=3,
               label="With retrieval", edgecolors="white", linewidths=0.8)
    for y, (_, row) in zip(positions, frame.iterrows()):
        ax.text(1.02, y, f"n={int(row['n'])}", va="center", fontsize=8.5,
                color=PALETTE["muted"], transform=ax.get_yaxis_transform())
    ax.set_yticks(positions, frame["category"], fontsize=9)
    ax.set_xlabel("Responses containing a false claim")
    ax.set_xlim(-0.04, 1.04)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


@register(
    "p2_truncation_confound", "phase2_generation_jul",
    "The generation-budget fault. Retrieval truncates a third of Mistral's "
    "responses and none of Llama's, and that alone explains Mistral's apparent "
    "quality loss under retrieval.",
    ["data/grading/phase2_results_scored.csv"],
    notes="Supports D-109. Without this panel the Mistral quality drop reads as a "
          "finding about retrieval rather than a bug in our settings.",
)
def _p2_truncation_confound(out: Path) -> None:
    """Two panels: truncation rate per run, and quality with and without truncated."""
    scored = _load_scored()
    runs = sorted(scored["run"].unique())
    positions = np.arange(len(runs))
    width = 0.36

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.4))

    base_t = [scored[(scored["run"] == r) & ~scored["rag"]]["truncated"].mean() for r in runs]
    rag_t = [scored[(scored["run"] == r) & scored["rag"]]["truncated"].mean() for r in runs]
    bars_a = left.bar(positions - width / 2, base_t, width, color=PALETTE["muted"],
                      label="Baseline")
    bars_b = left.bar(positions + width / 2, rag_t, width, color=PALETTE["warn"],
                      label="With retrieval")
    _label_bars(left, bars_a, base_t, offset=0.008)
    _label_bars(left, bars_b, rag_t, offset=0.008)
    left.set_xticks(positions, runs, fontsize=8.5)
    left.set_ylabel("Responses cut off mid-sentence")
    left.set_ylim(0, 0.46)
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.set_title("Truncation")
    left.legend(frameon=False, loc="upper left", fontsize=8.5)

    all_q = [scored[(scored["run"] == r) & scored["rag"]]["quality_score"].mean()
             for r in runs]
    untr = [scored[(scored["run"] == r) & scored["rag"] & ~scored["truncated"]]
            ["quality_score"].mean() for r in runs]
    base_q = [scored[(scored["run"] == r) & ~scored["rag"]]["quality_score"].mean()
              for r in runs]
    right.plot(positions, base_q, "o--", color=PALETTE["muted"], markersize=6,
               label="Baseline")
    right.plot(positions, all_q, "s-", color=PALETTE["warn"], markersize=6,
               label="RAG, all responses")
    right.plot(positions, untr, "^-", color=PALETTE["accent"], markersize=6,
               label="RAG, excluding truncated")
    right.set_xticks(positions, runs, fontsize=8.5)
    right.set_ylabel("Mean rubric quality (0-3)")
    right.set_ylim(1.5, 2.8)
    right.set_title("Quality, before and after controlling for it")
    right.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.savefig(out)
    plt.close(fig)


@register(
    "p2_rubric_overview", "phase2_generation_jul",
    "Every rubric measure, baseline against retrieval, pooled over all 416 "
    "responses. Accuracy improves, diagnostic safety is unchanged at zero, and "
    "two measures move the wrong way.",
    ["data/grading/phase2_results_scored.csv"],
    notes="The honest one-figure summary: retrieval is a trade, not a free win.",
)
def _p2_rubric_overview(out: Path) -> None:
    """Horizontal paired bars for every rubric measure, pooled."""
    scored = _load_scored()
    measures = [
        ("hallucination_flag", "Contains a false claim", True),
        ("diagnostic_overreach_flag", "Crosses the diagnosis line", True),
        ("truncated", "Cut off mid-sentence", True),
        ("professional_followup_flag", "Recommends professional follow-up", False),
        ("addresses_question_flag", "Answers the question asked", False),
    ]
    labels, base, rag, better_when_lower = [], [], [], []
    for column, label, lower_is_better in measures:
        labels.append(label)
        base.append(scored[~scored["rag"]][column].mean())
        rag.append(scored[scored["rag"]][column].mean())
        better_when_lower.append(lower_is_better)

    positions = np.arange(len(labels))
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(positions + height / 2, base, height, color=PALETTE["muted"], label="Baseline")
    ax.barh(positions - height / 2, rag, height, color=PALETTE["primary"],
            label="With retrieval")
    for y, (b, r, lower) in enumerate(zip(base, rag, better_when_lower)):
        delta = r - b
        improved = (delta < 0) if lower else (delta > 0)
        colour = PALETTE["accent"] if improved else PALETTE["warn"]
        text = f"{delta * 100:+.1f}pp"
        if abs(delta) < 1e-9:
            colour = PALETTE["muted"]
            # A measure that is zero in both arms draws no bar at all, so say so
            # rather than leaving the reader with an empty row.
            text = "0% in both arms" if max(b, r) < 1e-9 else "no change"
        ax.text(max(b, r) + 0.03, y, text, va="center",
                fontsize=9, color=colour, fontweight="bold")
    ax.set_yticks(positions, labels, fontsize=9)
    ax.set_xlabel("Share of responses")
    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# The 25 August re-run, after the generation and retrieval fixes
# --------------------------------------------------------------------------

GENERATION_DIR = BASE_DIR / "data" / "generation_eval"


def _rerun_frame() -> pd.DataFrame:
    """Per-response record of both run sets, labelled before and after."""
    frame = _require(GENERATION_DIR / "per_response_truncation_fix.csv")
    frame["arm"] = frame["rag"].map({True: "RAG", False: "baseline"})
    frame["short"] = (frame["model_id"].str.replace("-7b", "", regex=False)
                                       .str.replace("3-8b", "3", regex=False)
                      + " " + frame["condition"].str.replace("_", "-", regex=False)
                      + " " + frame["arm"])
    return frame


def _rerun_order(frame: pd.DataFrame) -> list[str]:
    """Runs ordered baseline block then RAG block, so the two arms read together."""
    baseline = sorted(frame.loc[~frame["rag"], "short"].unique())
    rag = sorted(frame.loc[frame["rag"], "short"].unique())
    return baseline + rag


@register(
    "p2_generation_fix", "generation_fix",
    "Effect of completion-scoped repetition control on the generated output. "
    "Mistral's retrieval responses stop being cut off and recover the length "
    "the constraint was costing them; nothing else moves.",
    ["data/generation_eval/per_response_truncation_fix.csv"],
    notes="The verification of F-P2-010 on real output rather than on a replay. "
          "Left panel is the claim; right panel is the mechanism.",
)
def _p2_generation_fix(out: Path) -> None:
    """Two panels: share of complete responses, and mean length, before and after."""
    frame = _rerun_frame()
    runs = _rerun_order(frame)
    positions = np.arange(len(runs))
    width = 0.38

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.8, 3.9))

    for ax, column, label, formatter in (
        (left, "complete", "Responses ending in a complete sentence", "pct"),
        (right, "completion_tokens", "Mean response length (tokens)", "num"),
    ):
        before = [frame[(frame["short"] == r) & (frame["arm"] == "before")][column].mean()
                  for r in runs]
        after = [frame[(frame["short"] == r) & (frame["arm"] == "after")][column].mean()
                 for r in runs]
        ax.bar(positions - width / 2, before, width, color=PALETTE["muted"],
               label="Before the fix")
        ax.bar(positions + width / 2, after, width, color=PALETTE["primary"],
               label="After the fix")
        # Mark the two runs the fix was supposed to help.
        for x, run in enumerate(runs):
            if "mistral" in run and "RAG" in run:
                ax.axvspan(x - 0.5, x + 0.5, color=PALETTE["accent"], alpha=0.10,
                           zorder=0)
        ax.set_xticks(positions, runs, fontsize=8, rotation=38, ha="right")
        # The two arms are separate experiments; keep them visually apart.
        ax.axvline(3.5, color=PALETTE["muted"], linewidth=0.8, linestyle=":")
        ax.set_ylabel(label)
        if formatter == "pct":
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        else:
            ax.set_ylim(0, 340)
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    fig.savefig(out)
    plt.close(fig)


@register(
    "p2_length_recovery", "generation_fix",
    "Length of Mistral's retrieval responses before and after the fix. The "
    "spike of answers under 50 tokens, which is what a banned continuation "
    "produced, is gone.",
    ["data/generation_eval/per_response_truncation_fix.csv"],
    notes="Shows the shape of the change, not just the mean. The old mode sits "
          "at the far left because a blocked response stops almost immediately.",
)
def _p2_length_recovery(out: Path) -> None:
    """Overlaid histograms of Mistral RAG response length, before against after."""
    frame = _rerun_frame()
    mistral = frame[(frame["model_id"] == "mistral-7b") & frame["rag"]]
    bins = np.arange(0, 340, 20)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for era, colour, label in (("before", "warn", "Before, sequence-scoped"),
                               ("after", "primary", "After, completion-scoped")):
        values = mistral[mistral["arm"] == era]["completion_tokens"]
        ax.hist(values, bins=bins, alpha=0.62, color=PALETTE[colour],
                label=f"{label} (mean {values.mean():.0f})", edgecolor="white",
                linewidth=0.6)
    ax.axvline(50, color=PALETTE["muted"], linestyle=":", linewidth=1.1)
    ax.text(53, ax.get_ylim()[1] * 0.92, "50 tokens", fontsize=8.5,
            color=PALETTE["muted"])
    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Responses")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


RERUN_GRADING = BASE_DIR / "data" / "grading_20260825"


def _both_passes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Phase 2 and re-run graded responses, both with a boolean rag column."""
    before = _require(GRADING_DIR / "phase2_results_scored.csv")
    after = _require(RERUN_GRADING / "rerun_results_scored.csv")
    for frame in (before, after):
        frame["rag"] = frame["rag_enabled"].astype(bool)
    return before, after


@register(
    "rerun_rubric_comparison", "phase2_rerun_aug",
    "Every rubric measure before and after the three fixes, baseline against "
    "retrieval. The two measures that moved the wrong way in Phase 2 no longer "
    "do, and retrieval-arm quality gains a quarter of a point.",
    ["data/grading/phase2_results_scored.csv",
     "data/grading_20260825/rerun_results_scored.csv"],
    notes="The headline figure for the re-run. Read the arrows, not the bars: "
          "what changed is the direction of the retrieval effect on follow-up "
          "advice and on answering the question.",
)
def _rerun_rubric_comparison(out: Path) -> None:
    """Four small panels, one per measure, each showing both passes and both arms."""
    before, after = _both_passes()
    measures = [
        ("hallucination_flag", "Contains a false claim", True),
        ("quality_score", "Mean rubric quality (0-3)", False),
        ("professional_followup_flag", "Recommends professional follow-up", True),
        ("addresses_question_flag", "Answers the question asked", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.6))
    for ax, (column, label, as_pct) in zip(axes, measures):
        values = [
            [before[~before["rag"]][column].mean(), before[before["rag"]][column].mean()],
            [after[~after["rag"]][column].mean(), after[after["rag"]][column].mean()],
        ]
        positions = np.arange(2)
        width = 0.36
        ax.bar(positions - width / 2, [values[0][0], values[1][0]], width,
               color=PALETTE["muted"], label="Baseline")
        ax.bar(positions + width / 2, [values[0][1], values[1][1]], width,
               color=PALETTE["primary"], label="With retrieval")
        for x, (base, rag) in enumerate(values):
            delta = rag - base
            improves = delta < 0 if column != "quality_score" else delta > 0
            if column in ("professional_followup_flag", "addresses_question_flag"):
                improves = delta > 0
            colour = PALETTE["accent"] if improves else PALETTE["warn"]
            top = max(base, rag)
            text = f"{delta:+.2f}" if column == "quality_score" else f"{delta * 100:+.0f}pp"
            ax.text(x, top * 1.06, text, ha="center", fontsize=9,
                    color=colour, fontweight="bold")
        ax.set_xticks(positions, ["Phase 2", "after\nfixes"], fontsize=9)
        ax.set_title(label, fontsize=9.5)
        if as_pct:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_ylim(0, max(max(v) for v in values) * 1.35)
        else:
            ax.set_ylim(0, 3.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


@register(
    "rerun_error_families", "phase2_rerun_aug",
    "Where the recurring errors came from. Every response that misused a "
    "screening instrument was a baseline response; retrieval eliminated that "
    "family entirely. Fabricated citations and refusals split evenly and are "
    "not a retrieval effect.",
    ["docs/report/numbers/rerun_error_families.csv"],
    notes="The most useful figure in the pass for the report's discussion: it "
          "separates what retrieval fixes from what it does not touch.",
)
def _rerun_error_families(out: Path) -> None:
    """Horizontal paired bars of each error family, split by arm."""
    frame = _require(BASE_DIR / "docs" / "report" / "numbers" /
                     "rerun_error_families.csv").sort_values("baseline_share")
    positions = np.arange(len(frame))
    height = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.barh(positions + height / 2, frame["baseline"], height,
            color=PALETTE["warn"], label="Baseline")
    ax.barh(positions - height / 2, frame["rag"], height,
            color=PALETTE["primary"], label="With retrieval")
    for y, (_, row) in zip(positions, frame.iterrows()):
        if row["rag"] == 0 and row["baseline"] > 0:
            ax.text(row["baseline"] + 0.4, y, "eliminated by retrieval",
                    va="center", fontsize=8.5, color=PALETTE["accent"],
                    fontweight="bold")
    labels = [name.replace(" a ", " a\n") if len(name) > 34 else name
              for name in frame["error_family"]]
    ax.set_yticks(positions, labels, fontsize=8.5)
    ax.set_xlabel("Responses")
    ax.set_xlim(0, frame[["baseline", "rag"]].to_numpy().max() * 1.9)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Schematics. These are drawn, not measured: every number written on them is
# quoted from a table in docs/report/numbers/ and named in the caption.
# --------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, face, edge, fontsize=8.5):
    """Draw one labelled box and return its centre, for arrow anchoring."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3)
    # Edge anchors, so connectors run between boxes instead of through them.
    return {
        "centre": (x + w / 2, y + h / 2),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
    }


def _arrow(ax, start, end, colour, rad=0.0, sides=("right", "left")):
    """
    Draw a connector between two boxes, anchored on the named edges.

    Accepts either the dicts returned by _box or bare coordinate pairs.
    """
    if isinstance(start, dict):
        start = start[sides[0]]
    if isinstance(end, dict):
        end = end[sides[1]]
    # Above the boxes: at a lower zorder the arrowheads are painted over.
    ax.annotate("", xy=end, xytext=start, zorder=4, arrowprops={
        "arrowstyle": "-|>", "color": colour, "linewidth": 1.3,
        "mutation_scale": 13, "connectionstyle": f"arc3,rad={rad}",
        "shrinkA": 5, "shrinkB": 5,
    })


@register(
    "system_architecture", "overview",
    "The pipeline as built. A caregiver turn passes through the Phase 3 router, "
    "then retrieval grounds the answer before generation. Every safety "
    "constraint sits in the system prompt, which retrieval augments rather than "
    "replaces.",
    ["docs/report/numbers/rerun_rubric_paired.csv",
     "docs/report/numbers/p3_experiments.csv",
     "docs/report/numbers/p2_retrieval_configs.csv"],
    notes="Schematic, not measured. The numbers written on it are quoted from "
          "the three tables listed as sources and must be updated if those change.",
)
def _system_architecture(out: Path) -> None:
    """Draw the end-to-end pipeline with the phase each component belongs to."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.grid(False)

    light, primary = PALETTE["light"], PALETTE["primary"]
    accent, warn, muted = PALETTE["accent"], PALETTE["warn"], PALETTE["muted"]

    # Phase bands, drawn first so the boxes sit on top of them.
    for x0, w, label, colour in ((1.05, 2.5, "Phase 3  routing", accent),
                                 (3.75, 2.6, "Phase 2  retrieval", primary),
                                 (6.55, 2.5, "Phase 1  generation", muted)):
        ax.add_patch(plt.Rectangle((x0, 0.55), w, 4.05, facecolor=colour,
                                   alpha=0.07, edgecolor="none", zorder=0))
        ax.text(x0 + w / 2, 4.72, label, ha="center", fontsize=9.5,
                color=colour, fontweight="bold")

    turn = _box(ax, 0.05, 2.45, 0.9, 0.8, "caregiver\nturn", "white", muted)
    rules = _box(ax, 1.15, 3.35, 1.1, 0.65, "rule\npatterns", "white", accent)
    gate = _box(ax, 1.15, 2.45, 1.1, 0.65, "safety\ngate", "white", accent)
    types = _box(ax, 1.15, 1.55, 1.1, 0.65, "type\nclassifier", "white", accent)
    routed = _box(ax, 2.45, 2.45, 1.0, 0.8, "route", light, accent)

    corpus = _box(ax, 3.85, 3.3, 1.15, 0.75, "corpus\n297 passages", "white", primary)
    retrieve = _box(ax, 3.85, 2.1, 1.15, 0.8, "top-5 +\nneighbours", "white", primary)
    prompt = _box(ax, 5.15, 2.1, 1.15, 1.95,
                  "grounded\nsystem prompt\n\nsafety rules\n+ sources\n+ caregiver\ninstructions",
                  light, primary, fontsize=7.5)

    model = _box(ax, 6.65, 2.45, 1.15, 0.9, "LLM\n4-bit", "white", muted)
    control = _box(ax, 6.65, 1.3, 1.15, 0.75,
                   "completion-scoped\nrepetition\ncontrol", "white", muted, fontsize=7)
    answer = _box(ax, 8.05, 2.45, 0.95, 0.9, "response", "white", muted)
    rubric = _box(ax, 8.05, 0.7, 0.95, 1.05,
                  "rubric\n\nquality\nhallucination\noverreach", "white", warn,
                  fontsize=7)

    # A cascade, not a fan: each stage either decides or hands on to the next,
    # so the connectors run down the stack and each stage exits to the route.
    _arrow(ax, turn, rules, accent, rad=0.12)
    _arrow(ax, rules, gate, accent, sides=("bottom", "top"))
    _arrow(ax, gate, types, accent, sides=("bottom", "top"))
    for stage in (rules, gate, types):
        _arrow(ax, stage, routed, accent, rad=-0.08)
    _arrow(ax, routed, retrieve, primary, rad=-0.1)
    _arrow(ax, corpus, retrieve, primary, sides=("bottom", "top"))
    _arrow(ax, retrieve, prompt, primary)
    _arrow(ax, prompt, model, muted)
    _arrow(ax, control, model, muted, sides=("top", "bottom"))
    _arrow(ax, model, answer, muted)
    _arrow(ax, answer, rubric, warn, sides=("bottom", "top"))

    # The constraint that never moves, called out rather than left implicit.
    ax.text(5.72, 1.78, "never diagnoses", ha="center", fontsize=8,
            color=warn, fontweight="bold")

    for x, y, text, colour in (
        (2.55, 1.05, "86.9% routing accuracy\naudited labels, extended set", accent),
        (4.42, 1.35, "74.2% answer\ncoverage@5", primary),
        (7.22, 0.75, "0 truncated\nof 416", muted),
        (8.52, 0.28, "0 diagnostic overreach\nin 1,008 graded", warn),
    ):
        ax.text(x, y, text, ha="center", fontsize=7.5, color=colour, style="italic")

    fig.savefig(out)
    plt.close(fig)


@register(
    "evaluation_provenance", "overview",
    "How every number in the report is produced. Each box is a tracked file and "
    "each arrow a tracked script, so any figure can be traced back to the runs "
    "it came from.",
    ["docs/DATA_GUIDE.md"],
    notes="Schematic. The companion to DATA_GUIDE section 1; use it when "
          "explaining reproducibility rather than results.",
)
def _evaluation_provenance(out: Path) -> None:
    """Draw the chain from benchmark to finding, naming the script at each step."""
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.grid(False)

    primary, accent, warn, muted = (PALETTE["primary"], PALETTE["accent"],
                                    PALETTE["warn"], PALETTE["muted"])
    steps = [
        (0.05, "benchmark\n52 prompts\n+ references", primary),
        (1.72, "generation\nruns\n8 x 52", primary),
        (3.39, "blinded\nbatches\n16 x 26", accent),
        (5.06, "annotations\n416 grades", accent),
        (6.73, "scored and\npaired CSVs", warn),
        (8.40, "findings,\ntables,\nfigures", warn),
    ]
    centres = [_box(ax, x, 1.35, 1.4, 1.05, label, "white", colour, fontsize=7.5)
               for x, label, colour in steps]
    scripts = ["run_phase1.py", "prepare_grading.py", "manual\nannotation",
               "score_rag_rubric.py", "export_report_numbers.py"]
    for i, script in enumerate(scripts):
        _arrow(ax, centres[i], centres[i + 1], muted)
        midpoint = (centres[i]["centre"][0] + centres[i + 1]["centre"][0]) / 2
        ax.text(midpoint, 2.62, script, ha="center", fontsize=6.6,
                color=muted, family="monospace")

    ax.text(5.0, 0.72, "the blinding map is withheld from the annotator and "
            "opened only at the scoring step",
            ha="center", fontsize=8, color=accent, style="italic")
    ax.text(5.0, 0.28, "every box is a tracked file, every arrow a tracked script",
            ha="center", fontsize=8, color=muted, style="italic")
    fig.savefig(out)
    plt.close(fig)


@register(
    "rerun_hallucination_by_category", "phase2_rerun_aug",
    "Hallucination by prompt category after the fixes, baseline against "
    "retrieval. Retrieval clears the instrument and screening categories and "
    "still cannot touch general health knowledge.",
    ["data/grading_20260825/rerun_results_scored.csv"],
    notes="Read against retrieval_by_category: the categories retrieval fails to "
          "help are the categories retrieval itself scores worst on.",
)
def _rerun_hallucination_by_category(out: Path) -> None:
    """Dumbbell plot of per-category hallucination for the re-run."""
    scored = _require(RERUN_GRADING / "rerun_results_scored.csv")
    scored["rag"] = scored["rag_enabled"].astype(bool)
    rows = []
    for category, group in scored.groupby("category"):
        rows.append({
            "category": category.replace("_", " "), "n": len(group),
            "base": group[~group["rag"]]["hallucination_flag"].mean(),
            "rag": group[group["rag"]]["hallucination_flag"].mean(),
        })
    frame = pd.DataFrame(rows).sort_values("base")
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    for y, (_, row) in zip(positions, frame.iterrows()):
        colour = PALETTE["accent"] if row["rag"] <= row["base"] else PALETTE["warn"]
        ax.plot([row["base"], row["rag"]], [y, y], color=colour, lw=2.2, zorder=1,
                solid_capstyle="round")
    ax.scatter(frame["base"], positions, s=78, color=PALETTE["warn"], zorder=3,
               label="Baseline", edgecolors="white", linewidths=0.8)
    ax.scatter(frame["rag"], positions, s=52, color=PALETTE["primary"], zorder=3,
               label="With retrieval", edgecolors="white", linewidths=0.8)
    for y, (_, row) in zip(positions, frame.iterrows()):
        ax.text(1.02, y, f"n={int(row['n'])}", va="center", fontsize=8.5,
                color=PALETTE["muted"], transform=ax.get_yaxis_transform())
    ax.set_yticks(positions, frame["category"], fontsize=9)
    ax.set_xlabel("Responses containing a false claim")
    ax.set_xlim(-0.04, 1.04)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


@register(
    "rerun_quality_distribution", "phase2_rerun_aug",
    "Where the quality gain comes from. Retrieval roughly doubles the share of "
    "responses scoring 3 and halves the share scoring 1, rather than shifting "
    "every response a little.",
    ["data/grading_20260825/rerun_results_scored.csv"],
    notes="The mean alone hides this. Use it when arguing that retrieval changes "
          "the shape of the distribution, not just its centre.",
)
def _rerun_quality_distribution(out: Path) -> None:
    """Stacked bars of the 0-3 quality distribution for each arm."""
    scored = _require(RERUN_GRADING / "rerun_results_scored.csv")
    scored["rag"] = scored["rag_enabled"].astype(bool)
    colours = [PALETTE["warn"], PALETTE["secondary"], PALETTE["light"],
               PALETTE["primary"]]
    labels = ["0 unsafe", "1 inadequate", "2 adequate", "3 strong"]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for row, (arm, group) in enumerate([("Baseline", scored[~scored["rag"]]),
                                        ("With retrieval", scored[scored["rag"]])]):
        left = 0.0
        for score, colour, label in zip(range(4), colours, labels):
            share = float((group["quality_score"] == score).mean())
            if share == 0:
                continue
            ax.barh(row, share, left=left, height=0.55, color=colour,
                    edgecolor="white", linewidth=0.8,
                    label=label if row == 0 else None)
            if share > 0.045:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center",
                        va="center", fontsize=9,
                        color="white" if score in (0, 3) else "black")
            left += share
        ax.text(1.02, row, f"mean {group['quality_score'].mean():.2f}",
                va="center", fontsize=9, color=PALETTE["muted"],
                transform=ax.get_yaxis_transform())
    ax.set_yticks([0, 1], ["Baseline", "With\nretrieval"], fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of responses")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.34),
              ncol=4, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


@register(
    "rerun_hallucination_by_run", "phase2_rerun_aug",
    "Hallucination rate before and after retrieval, by run, on the 25 August "
    "data (n = 52 per bar). Every run falls and every fall is significant on a "
    "paired test, so the effect is not one model or one prompting condition.",
    ["data/grading_20260825/rerun_results_scored.csv",
     "data/grading_20260825/rerun_rubric_paired.csv"],
    notes="The re-run counterpart of p2_hallucination_by_run in folder 03. This "
          "is the one to quote: the July figure is the earlier state.",
)
def _rerun_hallucination_by_run(out: Path) -> None:
    """Grouped bars per run plus pooled, with the paired p value on each pair."""
    scored = _require(RERUN_GRADING / "rerun_results_scored.csv")
    scored["rag"] = scored["rag_enabled"].astype(bool)
    scored["run"] = (scored["model_id"] + "\n"
                     + scored["condition"].str.replace("_", "-", regex=False))
    paired = _require(RERUN_GRADING / "rerun_rubric_paired.csv")
    paired = paired[paired["measure"] == "hallucination_flag"].set_index("run")

    # Build the paired-table key from the columns rather than from the display
    # label: the model name contains a hyphen, so reversing the label loses it.
    pairs = sorted({(m, c) for m, c in zip(scored["model_id"], scored["condition"])})
    labels, keys, base, rag = [], [], [], []
    for model, condition in pairs:
        rows = scored[(scored["model_id"] == model)
                      & (scored["condition"] == condition)]
        labels.append(f"{model}\n{condition.replace('_', '-')}")
        keys.append(f"{model} {condition}")
        base.append(rows[~rows["rag"]]["hallucination_flag"].mean())
        rag.append(rows[rows["rag"]]["hallucination_flag"].mean())
    runs = list(labels)
    labels = labels + ["ALL\npooled"]
    keys.append("ALL")
    base.append(scored[~scored["rag"]]["hallucination_flag"].mean())
    rag.append(scored[scored["rag"]]["hallucination_flag"].mean())

    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    bars_a = ax.bar(positions - width / 2, base, width, color=PALETTE["warn"],
                    label="Baseline, no retrieval")
    bars_b = ax.bar(positions + width / 2, rag, width, color=PALETTE["primary"],
                    label="With retrieval")
    _label_bars(ax, bars_a, base, offset=0.012)
    _label_bars(ax, bars_b, rag, offset=0.012)

    for x, (b, r, key) in enumerate(zip(base, rag, keys)):
        ax.text(x, max(b, r) + 0.072, f"{(r - b) * 100:+.0f}pp", ha="center",
                fontsize=9.5, color=PALETTE["accent"], fontweight="bold")
        if key in paired.index:
            p_value = float(paired.loc[key, "p"])
            ax.text(x, max(b, r) + 0.128, f"p={p_value:.3g}", ha="center",
                    fontsize=7.5, color=PALETTE["muted"])
    ax.axvline(len(runs) - 0.5, color=PALETTE["muted"], lw=0.8, ls=":")
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_ylabel("Responses containing a false claim")
    ax.set_ylim(0, 0.60)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@register(
    "followup_targeting_collapse", "phase2_rerun_aug",
    "Follow-up advice by whether the turn owed a next step. Both arms post the "
    "same flat rate; only the baseline advises the turns that owe one more often "
    "than the turns that do not.",
    ["data/generation_eval/followup_conformance_rerun_20260825.csv",
     "data/generation_eval/followup_discrimination_rerun_20260825.csv"],
    notes="The figure earns its place because the collapse is a gap between two "
          "bars, which a table states but does not show. Error bars are 95% CIs; "
          "the owed bars rest on 10 of 52 prompts and are wide accordingly.",
)
def _followup_targeting_collapse(out: Path) -> None:
    """Plot conformance against the unprompted rate for each arm."""
    summary = _require(GENERATION_EVAL / "followup_conformance_rerun_20260825.csv")
    gap = _require(GENERATION_EVAL / "followup_discrimination_rerun_20260825.csv")
    summary = summary.set_index("arm").loc[["baseline", "rag"]]

    labels = {"baseline": "Baseline", "rag": "Retrieval"}
    positions = np.arange(len(summary))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for offset, (rate_col, low_col, high_col, colour, label) in enumerate([
        ("conformance", "conformance_ci_low", "conformance_ci_high",
         PALETTE["primary"], "Turn owed a next step"),
        ("unprompted_rate", "unprompted_ci_low", "unprompted_ci_high",
         PALETTE["muted"], "Turn owed nothing"),
    ]):
        values = summary[rate_col].to_numpy() * 100
        lower = values - summary[low_col].to_numpy() * 100
        upper = summary[high_col].to_numpy() * 100 - values
        ax.bar(positions + (offset - 0.5) * width, values, width,
               yerr=[lower, upper], capsize=3, color=colour, label=label,
               error_kw={"linewidth": 1, "ecolor": PALETTE["muted"]})

    # The gap is the finding, so it is drawn rather than left to be read off.
    # The label sits above both bars rather than between them: at retrieval the
    # gap is under a point, and a label at the midpoint lands on top of the bar.
    for i, arm in enumerate(summary.index):
        high = summary.loc[arm, "conformance"] * 100
        low = summary.loc[arm, "unprompted_rate"] * 100
        difference = high - low
        ceiling = max(summary.loc[arm, "conformance_ci_high"],
                      summary.loc[arm, "unprompted_ci_high"]) * 100
        if abs(difference) >= 3:
            ax.annotate(
                "", xy=(i, high), xytext=(i, low),
                arrowprops={"arrowstyle": "<->", "color": PALETTE["secondary"],
                            "linewidth": 1.1, "shrinkA": 0, "shrinkB": 0},
            )
        else:
            ax.plot([i - width / 2, i + width / 2], [high, high], linewidth=1.1,
                    color=PALETTE["secondary"])
        ax.text(i, ceiling + 4, f"gap {difference:+.1f}pp", va="bottom",
                ha="center", fontsize=9.5, color=PALETTE["secondary"],
                fontweight="bold")

    flat = summary["flat_rate"].to_numpy() * 100
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{labels[a]}\nflat rate {f:.1f}%"
                        for a, f in zip(summary.index, flat)])
    ax.set_ylabel("Responses giving a next step (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    observed = float(gap["observed"].iloc[0]) * 100
    ci_low = float(gap["ci_low"].iloc[0]) * 100
    ci_high = float(gap["ci_high"].iloc[0]) * 100
    ax.set_title(f"Targeting gap closes by {abs(observed):.1f}pp under retrieval "
                 f"(95% CI [{ci_low:+.1f}, {ci_high:+.1f}])", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def write_manifest(figure_dir: Path = FIGURE_DIR) -> Path:
    """
    Write docs/figures/MANIFEST.md, and a short README inside every folder.

    The manifest is what the report cites: caption, rebuild command, the data the
    numbers came from, and when the figure was last built. The per-folder READMEs
    exist so the directory is navigable without opening the manifest at all.
    """
    from datetime import datetime

    figure_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Report figures",
        "",
        "Generated by `python scripts/make_report_figures.py`. Do not edit by hand.",
        "",
        "Figures live in one folder per stage of the project, numbered in the order",
        "the work happened. Build a whole folder with `--group`, named below.",
        "",
        "| Folder | Group | Holds | Data from |",
        "|---|---|---|---|",
    ]
    for group, (folder, blurb, when) in FOLDERS.items():
        count = sum(1 for f in FIGURES.values() if f.group == group)
        lines.append(f"| `{folder}/` | `{group}` | {blurb} ({count}) | {when} |")
    lines.append("")

    for group, (folder, blurb, when) in FOLDERS.items():
        members = [f for f in FIGURES.values() if f.group == group]
        if not members:
            continue
        lines += [f"## {folder}", "", blurb, "", f"*{when}.*", ""]
        readme = [f"# {folder}", "", blurb, "", f"*{when}.*", "",
                  "Rebuild this folder:", "",
                  f"```bash\npython scripts/make_report_figures.py --group {group}\n```",
                  ""]
        for figure in members:
            path = figure_dir / folder / f"{figure.name}.png"
            if path.exists():
                built = datetime.fromtimestamp(path.stat().st_mtime)
                status = f"built {built:%d %b %Y}"
            else:
                status = "not built yet"
            lines += [
                f"### `{folder}/{figure.name}.png`",
                "",
                f"**Caption.** {figure.caption}",
                "",
                f"- Sources: {', '.join(f'`{s}`' for s in figure.sources)}",
                f"- Rebuild: `python scripts/make_report_figures.py --figure {figure.name}`",
                f"- Status: {status}",
            ]
            if figure.notes:
                lines.append(f"- Note: {figure.notes}")
            lines.append("")
            readme += [f"**`{figure.name}.png`** — {figure.caption}", ""]

        (figure_dir / folder).mkdir(parents=True, exist_ok=True)
        (figure_dir / folder / "README.md").write_text(
            "\n".join(readme) + "\n", encoding="utf-8")

    path = figure_dir / "MANIFEST.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote figure manifest to %s", path)
    return path


def main() -> None:
    """Parse arguments and build the requested figures."""
    parser = argparse.ArgumentParser(description="Build the final report's figures.")
    parser.add_argument("--figure", help="build one figure by name")
    parser.add_argument("--group",
                        help="build one folder's worth of figures; "
                             "see MANIFEST.md for the group names")
    parser.add_argument("--list", action="store_true", help="list registered figures")
    args = parser.parse_args()

    if args.list:
        for name, figure in FIGURES.items():
            logger.info("  %-32s [%s] %s", name, figure.group, figure.caption[:70])
        return

    if args.figure:
        if args.figure not in FIGURES:
            raise ValueError(
                f"Unknown figure '{args.figure}'. Available: {sorted(FIGURES)}"
            )
        selected = [FIGURES[args.figure]]
    elif args.group:
        selected = [f for f in FIGURES.values() if f.group == args.group]
        if not selected:
            raise ValueError(
                f"No figures in group '{args.group}'. Available groups: "
                f"{sorted({f.group for f in FIGURES.values()})}"
            )
    else:
        selected = list(FIGURES.values())

    built, skipped = 0, []
    for figure in selected:
        destination = FIGURE_DIR / folder_for(figure.group)
        destination.mkdir(parents=True, exist_ok=True)
        out = destination / f"{figure.name}.png"
        try:
            figure.build(out)
            built += 1
            logger.info("Built %s/%s", out.parent.name, out.name)
        except FileNotFoundError as exc:
            skipped.append(figure.name)
            logger.warning("Skipped %s: %s", figure.name, exc)

    write_manifest()
    logger.info("Built %d figure(s) into %s", built, FIGURE_DIR)
    if skipped:
        logger.warning("Skipped for missing inputs: %s", ", ".join(skipped))


if __name__ == "__main__":
    main()
