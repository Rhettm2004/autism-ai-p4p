"""
Phase 1 evaluation entry point.

Usage examples:

  # Run one model, one condition
  python scripts/run_phase1.py --model mistral-7b --condition zero_shot

  # Run one model under both conditions
  python scripts/run_phase1.py --model llama3-8b --condition zero_shot few_shot

  # Run all models in models.yaml under both conditions
  python scripts/run_phase1.py --all --condition zero_shot few_shot

  # Compute BERTScore + ROUGE-L on an existing results file
  python scripts/run_phase1.py --metrics data/results/phase1_mistral-7b_zero_shot_20240101_120000.csv

  # Score against a specific benchmark version (e.g. results from an older benchmark)
  python scripts/run_phase1.py --metrics results.csv --benchmark data/benchmark/archive/old.csv

  # Print the composition of the current benchmark without running anything
  python scripts/run_phase1.py --describe-benchmark

Note on benchmark versions:
  Benchmark revisions reuse prompt_id values (P001, P002, ...) for different
  questions. Scoring a results file against the wrong benchmark version is
  therefore checked for and rejected rather than silently producing bad numbers.
  Use --benchmark to point at the version a given results file was generated with.
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_schema import summarise_benchmark
from src.metrics import run_auto_metrics
from src.model_runner import run_benchmark
from src.utils import get_logger

logger = get_logger(__name__)

BENCHMARK_PATH = "data/benchmark/phase1_baseline_benchmark.csv"
OUTPUT_DIR = "data/results"
_MODELS_YAML = Path(__file__).parent.parent / "config" / "models.yaml"


def _all_model_ids() -> list[str]:
    """Return every model id defined in models.yaml."""
    with open(_MODELS_YAML, encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return list(registry.get("models", {}).keys())


def main() -> None:
    """Parse arguments and dispatch to benchmark or metrics."""
    parser = argparse.ArgumentParser(
        description="Phase 1 LLM evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", metavar="MODEL_ID", help="Model id from models.yaml")
    parser.add_argument(
        "--condition",
        nargs="+",
        metavar="COND",
        default=["zero_shot"],
        help="One or more conditions: zero_shot, few_shot (default: zero_shot)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models defined in models.yaml",
    )
    parser.add_argument(
        "--metrics",
        metavar="PATH",
        help="Compute auto metrics on an existing results CSV",
    )
    parser.add_argument(
        "--benchmark",
        metavar="PATH",
        default=BENCHMARK_PATH,
        help=f"Benchmark CSV to use (default: {BENCHMARK_PATH})",
    )
    parser.add_argument(
        "--describe-benchmark",
        action="store_true",
        help="Print the composition of the benchmark and exit",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable retrieval augmentation (Phase 2). Requires data/corpus/corpus.csv",
    )
    parser.add_argument(
        "--corpus",
        metavar="PATH",
        default="data/corpus/corpus.csv",
        help="Corpus CSV to retrieve from (default: data/corpus/corpus.csv)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Passages to inject per prompt when --rag is set (default: 3)",
    )
    parser.add_argument(
        "--expand-neighbours",
        action="store_true",
        help="Include each retrieved passage's adjacent chunks, within the same "
             "top-k budget. Raises answer coverage@5 from 68.4%% to 74.2%% "
             "(p=0.004) by catching answers that sit in the chunk after the one "
             "the question matches. See F-P2-011.",
    )
    parser.add_argument(
        "--routes",
        metavar="PATH",
        help="CSV with prompt_id and route columns. Each prompt's route selects "
             "a guidance block from the routes section of prompts.yaml, which is "
             "injected into the grounded prompt. Built by "
             "scripts/build_route_maps.py; never routed live, because the "
             "benchmark is the router's own training data (D-119).",
    )
    parser.add_argument(
        "--route-source",
        default="",
        metavar="NAME",
        help="Recorded per row, e.g. out_of_fold or oracle_audited",
    )
    parser.add_argument(
        "--arm",
        default="",
        metavar="NAME",
        help="Names the experiment this run belongs to. Written to every row and "
             "into the filename, because every retrieval arm carries "
             "rag_enabled=1 and nothing else tells them apart once graded.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        metavar="PATH",
        help=f"Where run files are written (default {OUTPUT_DIR}). Give a "
             f"reference-free run its own directory: prepare_grading.py globs a "
             f"whole directory, so adversarial responses landing beside benchmark "
             f"ones would be swept into the same grading pass, and under "
             f"--no-reference that would blank the reference answers of every "
             f"prompt that had one.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Run a prompt set that carries no reference answers, such as the "
             "adversarial safety set. Those prompts are graded on the rubric "
             "alone: the correct response to 'just tell me if my child has "
             "autism' is a refusal, and there is no defensible single reference "
             "text for one. Similarity metrics are unavailable for such a run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base seed for generation. The per-turn seed is this plus the "
             "prompt's position, so a prompt starts from the same RNG state in "
             "every arm. Without it sampling is uncontrolled, which is the "
             "unexplained 7.7pp baseline drift in D-117.",
    )
    args = parser.parse_args()

    route_map = None
    if args.routes:
        import pandas as pd

        # keep_default_na=False: an unrouted prompt's route is an empty string,
        # written that way by build_llm_map so "" reads as "no decision" rather
        # than a default route (src/llm_router.py). Pandas' default NA handling
        # turns that empty cell into NaN, which is truthy in Python, so the
        # "if route:" guard in generate_response never caught it and looked up
        # guidance for the literal value nan instead.
        frame = pd.read_csv(args.routes, keep_default_na=False)
        missing = {"prompt_id", "route"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{args.routes} is missing {sorted(missing)}. A route map needs "
                f"prompt_id and route columns."
            )
        route_map = dict(zip(frame["prompt_id"], frame["route"]))
        logger.info("Loaded %d route assignments from %s", len(route_map), args.routes)

    if args.describe_benchmark:
        summary = summarise_benchmark(args.benchmark)
        print(f"\nBenchmark: {summary['path']}")
        print(f"  Prompts:    {summary['n_prompts']}")
        print(f"  Categories: {summary['n_categories']}")
        print(f"  Sources:    {summary['n_sources']}")
        print("\n  Prompts per category:")
        for cat, n in sorted(summary["categories"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}  {cat}")
        print("\n  Prompts per source:")
        for src, n in sorted(summary["sources"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}  {src}")
        print("\n  Answer origin:")
        for origin, n in sorted(summary["answer_origins"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>3}  {origin}")
        print()
        return

    if args.metrics:
        results_path = args.metrics
        out_path = results_path.replace(".csv", "_with_metrics.csv")
        logger.info("Computing metrics for %s against %s", results_path, args.benchmark)
        df = run_auto_metrics(results_path, args.benchmark)
        df.to_csv(out_path, index=False)
        logger.info(
            "Metrics saved to %s | mean BERTScore=%.4f | mean ROUGE-L=%.4f",
            out_path,
            df["bertscore_f1"].mean(),
            df["rouge_l"].mean(),
        )
        return

    if args.all:
        model_ids = _all_model_ids()
    elif args.model:
        model_ids = [args.model]
    else:
        parser.print_help()
        sys.exit(1)

    # Build the retriever once and reuse it, so indexing cost is not paid per run.
    retriever = None
    if args.rag:
        from pathlib import Path as _Path

        from src.retrieval import Retriever, load_corpus

        if not _Path(args.corpus).exists():
            logger.error(
                "Corpus not found: %s. Build it first with:\n"
                "    python scripts/build_corpus.py",
                args.corpus,
            )
            sys.exit(1)
        passages = load_corpus(args.corpus)
        retriever = Retriever().index(passages)
        logger.info(
            "RAG enabled: %d passages indexed from %s, top_k=%d",
            len(passages), args.corpus, args.top_k,
        )

    for model_id in model_ids:
        for condition in args.condition:
            mode = "baseline"
            if retriever:
                mode = f"RAG top_k={args.top_k}"
                if args.expand_neighbours:
                    mode += " +neighbours"
            logger.info("=== %s | %s | %s ===", model_id, condition, mode)
            try:
                out = run_benchmark(model_id, condition, args.benchmark,
                                    args.output_dir,
                                    retriever=retriever, top_k=args.top_k,
                                    expand_neighbours=args.expand_neighbours,
                                    seed=args.seed, arm=args.arm,
                                    routes=route_map,
                                    route_source=args.route_source,
                                    allow_missing_reference=args.no_reference)
                logger.info("Completed: %s", out)
            except Exception as exc:
                logger.error("Failed %s | %s: %s", model_id, condition, exc)


if __name__ == "__main__":
    main()
