"""BERTScore, ROUGE-L, and hallucination rate computation."""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# bert_score and rouge_score are heavy optional dependencies (they pull in torch
# and transformers). They are imported lazily inside the functions that need
# them so that schema inspection and benchmark validation work in environments
# where the scoring stack is not installed.

# Pinned rather than left to bert_score's default, which is what lang="en"
# resolves to today but is a library decision that can change between versions.
# A BERTScore is only comparable against another computed with the same encoder,
# and the July 2026 runs were scored before anything recorded which one was used.
# roberta-large is that default, so pinning it keeps the old numbers comparable
# while making the choice explicit from here on.
BERTSCORE_MODEL = "roberta-large"


def bertscore_scorer_id() -> str:
    """
    Return the scorer identity to record alongside any BERTScore value.

    Two BERTScore columns are only comparable when this string matches.
    """
    import bert_score

    return f"{BERTSCORE_MODEL}/bert-score-{bert_score.__version__}"


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for each (prediction, reference) pair.

    Uses the pinned BERTSCORE_MODEL rather than resolving from a language code.
    Returns a list of F1 floats, one per pair.
    """
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(predictions, references, model_type=BERTSCORE_MODEL,
                             lang="en", verbose=False)
    return f1.tolist()


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """
    Compute ROUGE-L F-measure for each (prediction, reference) pair.

    Returns a list of fmeasure floats, one per pair.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


def compute_hallucination_rate(scored_csv_path: str) -> dict:
    """
    Compute overall and per (model_id, condition) hallucination rates.

    Expects a CSV with a 'hallucination_flag' column (0 or 1).
    Returns a dict with keys 'overall_rate' and 'breakdown' (list of dicts).
    """
    df = pd.read_csv(scored_csv_path)
    if "hallucination_flag" not in df.columns:
        raise ValueError(
            "Column 'hallucination_flag' not found in the scored CSV. "
            "Manually review responses and add a hallucination_flag column "
            "(0 = no hallucination, 1 = hallucination) before calling this function."
        )

    overall_rate = float(df["hallucination_flag"].mean())

    breakdown = (
        df.groupby(["model_id", "condition"])["hallucination_flag"]
        .mean()
        .reset_index()
        .rename(columns={"hallucination_flag": "hallucination_rate"})
        .to_dict(orient="records")
    )

    return {"overall_rate": overall_rate, "breakdown": breakdown}


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace so comparison ignores cosmetic edits."""
    return " ".join(str(text).lower().split())


def _assert_prompts_match(
    results_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    benchmark_csv_path: str,
) -> None:
    """
    Verify each result row's prompt text matches the benchmark row with the same id.

    Guards against scoring a results file against a benchmark revision that reused
    prompt ids for different questions, which would otherwise pass silently and
    produce meaningless metrics.
    """
    if "prompt" not in results_df.columns or "prompt" not in benchmark_df.columns:
        logger.warning("Prompt text unavailable on one side; skipping prompt-match check.")
        return

    bench_prompts = dict(zip(benchmark_df["prompt_id"], benchmark_df["prompt"]))

    mismatches: list[str] = []
    for _, row in results_df.iterrows():
        pid = row["prompt_id"]
        expected = bench_prompts.get(pid)
        if expected is None:
            continue
        if _normalise(row["prompt"]) != _normalise(expected):
            mismatches.append(pid)

    if mismatches:
        preview = ", ".join(map(str, mismatches[:5]))
        raise ValueError(
            f"Prompt text mismatch on {len(mismatches)} of {len(results_df)} rows "
            f"(e.g. {preview}). The results file and {benchmark_csv_path} use the same "
            "prompt_id values for different questions, so joining them would score each "
            "response against an unrelated reference answer.\n\n"
            "These results were generated against a different benchmark version. Either "
            "point --metrics at the benchmark those results were produced with, or re-run "
            "the benchmark against the current CSV."
        )


def run_auto_metrics(results_csv_path: str, benchmark_csv_path: str) -> pd.DataFrame:
    """
    Join results with reference answers from the benchmark and compute BERTScore + ROUGE-L.

    The benchmark CSV must have a 'reference_answer' column sourced from clinical literature.
    Returns a DataFrame with bertscore_f1 and rouge_l columns added.
    """
    results_df = pd.read_csv(results_csv_path)
    benchmark_df = pd.read_csv(benchmark_csv_path)

    if "reference_answer" not in benchmark_df.columns:
        raise ValueError(
            "The benchmark CSV does not have a 'reference_answer' column. "
            "Each row of data/benchmark/phase1_baseline_benchmark.csv must carry a "
            "'reference_answer' grounded in the cited source ('source_name' / "
            "'source_url' columns). Do not write answers from general knowledge."
        )

    # Carry source provenance through to the metrics output where available.
    # Columns the results file already carries are skipped so the merge does not
    # produce _x / _y suffixed duplicates.
    join_cols = ["prompt_id", "reference_answer"]
    for col in ("source_name", "source_url", "answer_origin"):
        if col in benchmark_df.columns and col not in results_df.columns:
            join_cols.append(col)

    merged = results_df.merge(
        benchmark_df[join_cols],
        on="prompt_id",
        how="left",
    )

    unmatched = merged["reference_answer"].isna().sum()
    if unmatched:
        raise ValueError(
            f"{unmatched} of {len(merged)} result rows have no matching prompt_id in "
            f"{benchmark_csv_path}. The results file and the benchmark are out of sync — "
            "results were most likely generated against a different benchmark version. "
            "Re-run the benchmark against the current CSV before computing metrics."
        )

    # prompt_id alone is not enough. Benchmark revisions reuse the same ids for
    # different questions, so an id-only join can silently score a response
    # against an unrelated reference answer and still produce plausible numbers.
    # Verify the prompt text itself agrees before trusting the join.
    _assert_prompts_match(results_df, benchmark_df, benchmark_csv_path)

    predictions = merged["response"].fillna("").tolist()
    references = merged["reference_answer"].fillna("").tolist()

    logger.info("Computing BERTScore for %d pairs with %s", len(predictions),
                bertscore_scorer_id())
    merged["bertscore_f1"] = compute_bertscore(predictions, references)
    # Recorded per row so a BERTScore column can never be compared against one
    # produced by a different encoder without the mismatch being visible.
    merged["bertscore_scorer"] = bertscore_scorer_id()

    logger.info("Computing ROUGE-L for %d pairs", len(predictions))
    merged["rouge_l"] = compute_rouge_l(predictions, references)

    logger.info(
        "Mean BERTScore F1: %.4f | Mean ROUGE-L: %.4f",
        merged["bertscore_f1"].mean(),
        merged["rouge_l"].mean(),
    )

    return merged
