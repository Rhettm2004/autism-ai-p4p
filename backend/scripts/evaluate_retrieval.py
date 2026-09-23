"""
Measure what retrieval actually puts in front of the model.

Two questions, and the project has only been answering the first one.

  Source recall. Does the top k contain a passage from the source the benchmark
  cites for this prompt? This is what the existing recall curve reports.

  Answer coverage. Does the retrieved text actually contain the answer? Source
  recall counts a hit when any chunk of the right document is retrieved, and a
  document can be forty chunks long. P025 is the case that exposed the gap: the
  right document was retrieved three times over, the chunk stating the FDA
  approval was not, and both models then guessed, in opposite directions
  (F-P2-011).

Answer coverage is measured as the unigram recall of the benchmark's reference
answer against the concatenated retrieved context, over content words. It is a
proxy and it is generous, but it is paired and it moves when the retrieved text
stops containing the answer, which source recall does not.

Every run is recorded under a label so corpus and retriever changes can be
compared on identical prompts.

Usage:

  python scripts/evaluate_retrieval.py --label baseline
  python scripts/evaluate_retrieval.py --label no_boilerplate --compare-to baseline
"""

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.retrieval import Retriever, load_corpus  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = BASE_DIR / "data" / "corpus" / "corpus.csv"
BENCHMARK_PATH = BASE_DIR / "data" / "benchmark" / "phase1_baseline_benchmark.csv"
OUT_DIR = BASE_DIR / "data" / "retrieval_eval"
REGISTRY = OUT_DIR / "experiments.csv"

K_VALUES = (1, 3, 5, 10)
REPORT_K = 5

# Words carrying no topical content, excluded so coverage is not inflated by
# the reference answer and the corpus sharing ordinary English.
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "do", "does", "for", "from", "had", "has", "have", "if",
    "in", "is", "it", "its", "may", "more", "not", "of", "on", "or", "other",
    "should", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "was", "were", "what", "when",
    "which", "who", "will", "with", "would", "you", "your",
}


def content_words(text: str) -> set[str]:
    """Lowercase content words of a passage, for the coverage calculation."""
    words = re.findall(r"[a-z0-9']+", str(text).lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def evaluate(corpus_path: Path, benchmark_path: Path,
             expand_neighbours: bool = False) -> pd.DataFrame:
    """
    Retrieve for every benchmark prompt and score both measures per prompt.

    Returns one row per prompt so results stay paired across configurations.
    """
    passages = load_corpus(str(corpus_path))
    retriever = Retriever().index(passages)
    benchmark = pd.read_csv(benchmark_path)
    if "reference_answer" not in benchmark.columns:
        raise ValueError(
            f"{benchmark_path} has no reference_answer column, so answer coverage "
            f"cannot be computed. Columns present: {list(benchmark.columns)}"
        )

    top_k = max(K_VALUES)
    rows = []
    for _, row in benchmark.iterrows():
        hits = retriever.retrieve(row["prompt"], k=top_k,
                                  expand_neighbours=expand_neighbours)
        names = [p.source_name for p, _ in hits]

        # Source recall, the existing measure: the rank of the first passage
        # from the document the benchmark cites.
        rank = next((i + 1 for i, n in enumerate(names) if n == row["source_name"]), None)

        reference = content_words(row["reference_answer"])
        record = {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "source_name": row["source_name"],
            "rank": rank if rank is not None else math.inf,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "top1_passage": hits[0][0].passage_id if hits else "",
        }
        for k in K_VALUES:
            context = content_words(" ".join(p.text for p, _ in hits[:k]))
            record[f"source_hit@{k}"] = bool(rank is not None and rank <= k)
            record[f"coverage@{k}"] = (
                len(reference & context) / len(reference) if reference else 0.0
            )
        rows.append(record)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> dict:
    """Reduce the per-prompt frame to the headline numbers for the registry."""
    summary = {"n_prompts": len(frame), "mrr": round(frame["reciprocal_rank"].mean(), 4)}
    for k in K_VALUES:
        summary[f"recall@{k}"] = round(frame[f"source_hit@{k}"].mean(), 4)
        summary[f"coverage@{k}"] = round(frame[f"coverage@{k}"].mean(), 4)
    return summary


def paired_delta(current: pd.DataFrame, previous: pd.DataFrame, column: str) -> dict:
    """Compare two runs on the prompts they share, which is what makes it paired."""
    from scipy import stats

    merged = current.merge(previous, on="prompt_id", suffixes=("_now", "_before"))
    now = merged[f"{column}_now"].astype(float)
    before = merged[f"{column}_before"].astype(float)
    difference = now - before
    result = {"n": len(merged), "before": before.mean(), "now": now.mean(),
              "delta": difference.mean(), "improved": int((difference > 0).sum()),
              "worsened": int((difference < 0).sum())}
    if difference.std(ddof=1) > 0:
        _, p_value = stats.ttest_rel(now, before)
        result["p"] = float(p_value)
    else:
        result["p"] = float("nan")
    return result


def main() -> None:
    """Evaluate the current corpus and record the result under a label."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True,
                        help="Name for this configuration, e.g. no_boilerplate")
    parser.add_argument("--compare-to",
                        help="Label of an earlier run to compare against, paired")
    parser.add_argument("--corpus", default=str(CORPUS_PATH),
                        help="Corpus CSV to evaluate (default: the built corpus)")
    parser.add_argument("--expand-neighbours", action="store_true",
                        help="Include each hit's adjacent chunks within the same k "
                             "budget, for questions whose answer sits in the next chunk")
    parser.add_argument("--note", default="", help="What changed in this configuration")
    args = parser.parse_args()

    frame = evaluate(Path(args.corpus), BENCHMARK_PATH, args.expand_neighbours)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_prompt_path = OUT_DIR / f"per_prompt_{args.label}.csv"
    frame.to_csv(per_prompt_path, index=False)

    summary = summarise(frame)
    logger.info("=" * 72)
    logger.info("RETRIEVAL: %s (n = %d prompts)", args.label, summary["n_prompts"])
    logger.info("=" * 72)
    logger.info("  %-6s %14s %16s", "k", "source recall", "answer coverage")
    for k in K_VALUES:
        logger.info("  %-6d %13.1f%% %15.1f%%", k, summary[f"recall@{k}"] * 100,
                    summary[f"coverage@{k}"] * 100)
    logger.info("  MRR %.3f", summary["mrr"])

    if args.compare_to:
        previous_path = OUT_DIR / f"per_prompt_{args.compare_to}.csv"
        if not previous_path.exists():
            available = sorted(p.stem.replace("per_prompt_", "")
                               for p in OUT_DIR.glob("per_prompt_*.csv"))
            raise ValueError(
                f"No run labelled {args.compare_to!r}. Available labels: {available}"
            )
        previous = pd.read_csv(previous_path)
        logger.info("")
        logger.info("PAIRED AGAINST %s", args.compare_to)
        for column in (f"coverage@{REPORT_K}", f"source_hit@{REPORT_K}",
                       "reciprocal_rank"):
            delta = paired_delta(frame, previous, column)
            logger.info("  %-18s %6.3f -> %6.3f  delta %+.3f  better %d worse %d  p=%.4g",
                        column, delta["before"], delta["now"], delta["delta"],
                        delta["improved"], delta["worsened"], delta["p"])

    row = {"label": args.label, "note": args.note,
           "compare_to": args.compare_to or "", **summary}
    registry = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if len(registry) and args.label in set(registry["label"]):
        registry = registry[registry["label"] != args.label]
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True) \
        if len(registry) else pd.DataFrame([row])
    registry.to_csv(REGISTRY, index=False)
    logger.info("")
    logger.info("Wrote %s and updated %s", per_prompt_path, REGISTRY)


if __name__ == "__main__":
    main()
