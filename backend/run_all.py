"""
One command for the whole pipeline.

There is a distinction this script is built around, and getting it wrong would
quietly corrupt the experiment record.

**Re-derivations** are pure functions of committed data. Running them twice
produces the same files, so they are safe to run whenever and are what
--analysis-only does: the conformance tables, the arm comparison, every report
number and every figure. If one of these changes without an input changing,
something is wrong and the tests should have caught it.

**Experiments** append a labelled row to a registry — `evaluate_router.py` to
`data/router_eval/experiments.csv`, `evaluate_retrieval.py` to its own. Their
whole value is that the registry is a history of what was tried and what it
scored. Re-running them from a convenience wrapper would append a duplicate row
every time somebody rebuilt a figure, and the history would stop meaning
anything. **They are deliberately not run here.** Run them by hand, with a new
--label, when you have actually changed something:

    python scripts/evaluate_router.py --config cascade --label my_change \\
        --compare-to baseline
    python scripts/evaluate_retrieval.py --label my_change

Blind grading is likewise not automated and never will be. `prepare_grading.py`
writes the batches and refuses to overwrite a blinding map; a person grades them.

Usage:

  python run_all.py                       re-derive tables and figures, no GPU
  python run_all.py --full --rag          add the GPU generation runs
  python run_all.py --full --rag --arms   add the routed and union arms
  python run_all.py --full --llm-router  add the LLM router, variant B
  python run_all.py --verbose             stream each stage's output
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_logger  # noqa: E402

logger = get_logger("run_all")

REPO = Path(__file__).parent
BENCHMARK = REPO / "data" / "benchmark" / "phase1_baseline_benchmark.csv"
ADVERSARIAL = REPO / "data" / "benchmark" / "router_adversarial_v1.csv"
CORPUS = REPO / "data" / "corpus" / "corpus.csv"
SCORED = REPO / "data" / "grading_20260825" / "rerun_results_scored.csv"
# The v3 oracle map, archived. The August 2026 grading pass covers the
# 52-prompt v3 benchmark, and scoring it against the current oracle.csv would
# attach v4 routes to v3 questions of the same id (D-124).
ORACLE_MAP_V3 = REPO / "data" / "route_maps" / "archive" / "oracle_v3_52prompt.csv"
PREDICTED_MAP = REPO / "data" / "route_maps" / "predicted.csv"
UNION_MAP = REPO / "data" / "route_maps" / "union.csv"
LLM_MAP = REPO / "data" / "route_maps" / "llm_predicted.csv"
ADVERSARIAL_MAP = REPO / "data" / "route_maps" / "adversarial_predicted.csv"

# The two run directories the published era comparison comes from.
RUNS_JULY = REPO / "results_thursday"
RUNS_AUGUST = REPO / "results_20260825"

CONDITIONS = ["zero_shot", "few_shot"]


@dataclass
class Stage:
    """One step of the pipeline, with what it needs before it can run."""

    name: str
    command: list[str]
    requires: list[Path] = field(default_factory=list)
    needs_gpu: bool = False
    optional: bool = False
    note: str = ""


def missing_inputs(stage: Stage) -> list[Path]:
    """Return the required paths that do not exist."""
    return [path for path in stage.requires if not path.exists()]


def run_stage(stage: Stage, verbose: bool) -> tuple[str, float]:
    """
    Run one stage and return its outcome and duration.

    An optional stage whose inputs are absent is skipped with the reason named,
    because a compendium that silently omits a step is worse than one that says
    which step it could not take.
    """
    absent = missing_inputs(stage)
    if absent:
        names = ", ".join(p.name for p in absent)
        if stage.optional:
            logger.info("SKIP  %-52s (needs %s)", stage.name, names)
            return "skipped", 0.0
        raise FileNotFoundError(
            f"{stage.name} needs {names}, which does not exist. "
            f"{stage.note or 'Generate it before running this stage.'}"
        )

    started = time.time()
    result = subprocess.run(
        stage.command, cwd=REPO,
        capture_output=not verbose, text=True,
    )
    elapsed = time.time() - started

    if result.returncode != 0:
        if not verbose and result.stderr:
            for line in result.stderr.strip().splitlines()[-12:]:
                logger.error("      %s", line)
        logger.error("FAIL  %-52s %6.1fs", stage.name, elapsed)
        return "failed", elapsed

    logger.info("OK    %-52s %6.1fs", stage.name, elapsed)
    return "ok", elapsed


def generation_stages(args) -> list[Stage]:
    """
    The GPU stages, in the order a booking should run them.

    Arms are not all run at the same breadth, and that is deliberate rather than
    an economy. The baseline and retrieval arms run both models and both
    conditions, because Phase 2's claim is about retrieval in general and should
    not rest on one configuration. The routed arms and their control run one
    model and one condition — the strongest available — because their claim is
    that routing changes the answers at all, and proving that four times over at
    weaker configurations costs a booking without strengthening it.

    The consequence is the comparison happens at one configuration, where five
    arms answer the same prompts and can be paired per prompt. Say so in the
    report: the ladder is measured at llama3-8b few_shot, and the breadth arms
    sit alongside it rather than inside it.
    """
    python = sys.executable
    stages: list[Stage] = []

    retrieval = ["--rag", "--top-k", str(args.top_k)]
    if args.expand_neighbours:
        retrieval.append("--expand-neighbours")
    seed = ["--seed", str(args.seed)] if args.seed is not None else []

    def command(models: list[str] | None, conditions: list[str], arm: str,
                route_map: Path | None = None, route_source: str = "",
                rag: bool = False) -> list[str]:
        """Build one run_phase1 invocation for one arm."""
        out = [python, "scripts/run_phase1.py"]
        # --all is every model in models.yaml; a named model narrows the arm.
        out += ["--all"] if models is None else [
            arg for model in models for arg in ("--model", model)]
        out += ["--condition", *conditions]
        if rag:
            out += retrieval
        out += seed
        if args.arms or arm != "baseline":
            out += ["--arm", arm]
        if route_map is not None:
            out += ["--routes", route_map.relative_to(REPO).as_posix(),
                    "--route-source", route_source]
        return out

    # ---- breadth arms: every model, every condition ----
    stages.append(Stage(
        "Baseline generation, all models and conditions (GPU)",
        command(None, args.condition, "baseline"),
        requires=[BENCHMARK], needs_gpu=True,
    ))
    if args.rag:
        stages.append(Stage(
            "Retrieval generation, all models and conditions (GPU)",
            command(None, args.condition, "rag", rag=True),
            requires=[BENCHMARK, CORPUS], needs_gpu=True,
        ))

    if not args.arms:
        return stages

    # ---- ladder arms: one model, one condition, the strongest available ----
    model = [args.router_arm_model]
    condition = [args.router_arm_condition]
    where = f"{args.router_arm_model} {args.router_arm_condition}"

    # Router A replays frozen out-of-fold decisions rather than routing live,
    # because the benchmark is this router's training data.
    stages.append(Stage(
        f"Router A generation, {where} (GPU)",
        command(model, condition, "routed_a", PREDICTED_MAP, "out_of_fold",
                rag=True),
        requires=[BENCHMARK, CORPUS, PREDICTED_MAP], needs_gpu=True,
        note="Build the route maps first: python scripts/build_route_maps.py",
    ))

    if args.llm_router:
        # Router B's decisions reaching generation. This is the arm that answers
        # whether a better classifier produces better answers, which is a
        # different question from whether it classifies better and the only one
        # evaluate_llm_router.py cannot settle.
        stages.append(Stage(
            f"Router B generation, {where} (GPU)",
            command(model, condition, "routed_b", LLM_MAP, "llm_router",
                    rag=True),
            requires=[BENCHMARK, CORPUS, LLM_MAP], needs_gpu=True,
            note="Built by the LLM routing stage earlier in this run.",
        ))

    # The control that answers "why not put all seven blocks in the system
    # prompt and skip the router". Every prompt gets every route's obligations,
    # so a routed gain over this one cannot be extra prompt text.
    stages.append(Stage(
        f"Union-control generation, {where} (GPU)",
        command(model, condition, "union", UNION_MAP, "union_control", rag=True),
        requires=[BENCHMARK, CORPUS, UNION_MAP], needs_gpu=True,
    ))
    return stages


def llm_router_stages(args) -> list[Stage]:
    """
    Variant B of the router: the LLM classifier, scored against variant A.

    Runs in the same booking as generation because it needs the same
    GPU-resident models, and because the point of building it now was to avoid a
    second booking to answer whether the fitted router earns its place.

    --against pairs each run against variant A's out-of-fold predictions on the
    same prompts. Without it the two variants are two numbers rather than a
    comparison.
    """
    python = sys.executable
    stages = []
    for model_id in args.llm_router_models:
        for variant in args.llm_router_variants:
            label = f"llm_{model_id}_{variant}"
            command = [python, "scripts/evaluate_llm_router.py",
                       "--model", model_id, "--variant", variant,
                       "--label", label, "--track", args.track,
                       "--repeats", str(args.llm_router_repeats),
                       "--base-seed", str(args.seed if args.seed is not None
                                          else 20260827)]
            if args.against:
                command += ["--against", args.against]
            stages.append(Stage(
                f"LLM router: {model_id} {variant} (GPU)",
                command, requires=[BENCHMARK], needs_gpu=True,
                note="Variant B. Not fitted, so one pass is already leak-free.",
            ))
    return stages


def adversarial_stages(args) -> list[Stage]:
    """The held-out safety run: three arms over twenty prompts, zero-shot only."""
    python = sys.executable
    common = [python, "scripts/run_phase1.py", "--all", "--condition", "zero_shot",
              "--benchmark", ADVERSARIAL.relative_to(REPO).as_posix(),
              "--output-dir", "results_adversarial", "--no-reference"]
    retrieval = ["--rag", "--top-k", str(args.top_k)]
    if args.expand_neighbours:
        retrieval.append("--expand-neighbours")
    seed = ["--seed", str(args.seed)] if args.seed is not None else []

    return [
        Stage("Adversarial baseline (GPU)",
              common + seed + ["--arm", "baseline"],
              requires=[ADVERSARIAL], needs_gpu=True),
        Stage("Adversarial retrieval (GPU)",
              common + retrieval + seed + ["--arm", "rag"],
              requires=[ADVERSARIAL, CORPUS], needs_gpu=True),
        Stage("Adversarial routed (GPU)",
              common + retrieval + seed + [
                  "--routes", ADVERSARIAL_MAP.relative_to(REPO).as_posix(),
                  "--route-source", "live_embeddings_word_extended",
                  "--arm", "routed"],
              requires=[ADVERSARIAL, CORPUS, ADVERSARIAL_MAP], needs_gpu=True,
              note="Build it first: "
                   "python scripts/build_route_maps.py --adversarial"),
    ]


def analysis_stages(args) -> list[Stage]:
    """
    The re-derivations. Safe to run any time, on any machine, with no GPU.

    Ordered so that anything reading a file comes after whatever writes it:
    the evaluation scripts write into data/generation_eval/, which
    export_report_numbers.py reads, which make_report_figures.py reads.
    """
    python = sys.executable
    stages = [
        Stage(
            "Route-appropriate follow-up conformance",
            [python, "scripts/evaluate_followup.py", "--scored", str(SCORED),
             "--route-map", ORACLE_MAP_V3.relative_to(REPO).as_posix(),
             "--label", "rerun_20260825"],
            requires=[SCORED, ORACLE_MAP_V3], optional=True,
            note="Needs the August grading pass and the v3 route map it "
                 "belongs to.",
        ),
        Stage(
            "Generation comparison, July against August",
            [python, "scripts/evaluate_generation.py", "--arms",
             f"before={RUNS_JULY.name}", f"after={RUNS_AUGUST.name}",
             "--label", "truncation_fix"],
            requires=[RUNS_JULY, RUNS_AUGUST], optional=True,
            note="Needs both run directories.",
        ),
        Stage(
            "Export every report number",
            [python, "scripts/export_report_numbers.py"],
        ),
        Stage(
            "Rebuild every figure",
            [python, "scripts/make_report_figures.py"],
        ),
    ]
    if args.sync:
        stages.append(Stage(
            "Sync notes to the report folder",
            [python, "scripts/sync_report_notes.py"],
            optional=True,
            note="Writes outside the repository, so it is opt-in.",
        ))
    return stages


def main() -> None:
    """Assemble the requested stages, run them in order, and report the tally."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="Add the GPU generation runs. Without it only the "
                             "re-derivations run, which need no GPU.")
    parser.add_argument("--rag", action="store_true",
                        help="Add the retrieval arm to the generation runs")
    parser.add_argument("--arms", action="store_true",
                        help="Add the route-conditioned and union-control arms, "
                             "and name every arm explicitly. Required for the "
                             "end-to-end comparison, since all retrieval arms "
                             "carry rag_enabled=1 and nothing else separates them.")
    parser.add_argument("--router-arm-model", default="llama3-8b", metavar="ID",
                        help="Model for the routed arms and their control "
                             "(default llama3-8b). The baseline and retrieval "
                             "arms always run every model; the ladder runs one, "
                             "so the comparison sits at a single configuration.")
    parser.add_argument("--router-arm-condition", default="few_shot",
                        metavar="COND",
                        help="Condition for the routed arms (default few_shot)")
    parser.add_argument("--llm-router", action="store_true",
                        help="Also evaluate the LLM router (variant B) against "
                             "the fitted one, in the same booking")
    parser.add_argument("--llm-router-models", nargs="+", default=["mistral-7b"],
                        metavar="ID", help="Models to run the LLM router with "
                             "(default mistral-7b)")
    parser.add_argument("--llm-router-variants", nargs="+",
                        default=["zero_shot", "few_shot"], metavar="V",
                        help="Classification prompts to try")
    parser.add_argument("--llm-router-repeats", type=int, default=3,
                        metavar="N", help="Passes per model and variant. "
                             "Measures sampling variance, not fold variance: "
                             "this router is not fitted and has no folds.")
    parser.add_argument("--llm-route-from", metavar="LABEL",
                        help="Which LLM routing run feeds the llm_routed "
                             "generation arm. Defaults to the first model and "
                             "variant given.")
    parser.add_argument("--against", metavar="LABEL",
                        default="router_v4_topic_ext",
                        help="Variant A out-of-fold predictions to pair the LLM "
                             "router against (default router_v4_topic_ext)")
    parser.add_argument("--track", default="topic", metavar="NAME",
                        help="Ground-truth track for the router runs. Default "
                             "topic; audited does not load against benchmark v4 "
                             "(D-127).")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run the held-out safety set instead of the "
                             "benchmark: 20 prompts, three arms, zero-shot only")
    parser.add_argument("--condition", nargs="+", default=CONDITIONS,
                        metavar="COND", help=f"Conditions to run "
                        f"(default {' '.join(CONDITIONS)})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Passages retrieved per prompt (default 5)")
    parser.add_argument("--expand-neighbours", action="store_true",
                        help="Expand each hit to its neighbouring chunks")
    parser.add_argument("--seed", type=int,
                        help="Base generation seed, shared across arms so a "
                             "prompt starts from the same RNG state in each")
    parser.add_argument("--fetch-corpus", action="store_true",
                        help="Rebuild the RAG corpus from sources.yaml first")
    parser.add_argument("--sync", action="store_true",
                        help="Copy notes and figures to the report folder at the end")
    parser.add_argument("--verbose", action="store_true",
                        help="Stream each stage's output instead of capturing it")
    args = parser.parse_args()

    if (args.rag or args.arms or args.adversarial or args.llm_router)             and not args.full:
        parser.error("--rag, --arms, --adversarial and --llm-router only apply "
                     "with --full")

    stages: list[Stage] = []
    if args.full:
        stages.append(Stage(
            "Pre-flight checks",
            [sys.executable, "scripts/preflight.py"],
            requires=[BENCHMARK, CORPUS],
            note="Every check must pass before any GPU time is spent.",
        ))
    if args.fetch_corpus:
        stages.append(Stage("Rebuild the RAG corpus",
                            [sys.executable, "scripts/build_corpus.py"]))
    if args.full:
        if args.adversarial:
            stages += adversarial_stages(args)
        else:
            # Order matters and is not cosmetic. The LLM-routed arm generates
            # from a map that does not exist until variant B has classified, so
            # classification and map-building come before generation. Preflight
            # still runs first, ahead of everything.
            if args.llm_router:
                stages += llm_router_stages(args)
                if args.arms:
                    stages.append(Stage(
                        "Build the LLM router's route map",
                        [sys.executable, "scripts/build_route_maps.py",
                         "--llm", args.llm_route_from or
                         f"llm_{args.llm_router_models[0]}_"
                         f"{args.llm_router_variants[0]}"],
                        requires=[BENCHMARK],
                    ))
            stages += generation_stages(args)
    stages += analysis_stages(args)

    logger.info("=" * 78)
    logger.info("PIPELINE: %d stage(s), %d needing a GPU",
                len(stages), sum(s.needs_gpu for s in stages))
    logger.info("=" * 78)

    outcomes: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}
    total = 0.0
    for stage in stages:
        outcome, elapsed = run_stage(stage, args.verbose)
        outcomes[outcome] += 1
        total += elapsed
        # Pre-flight exists to stop a booking being wasted on a broken path.
        # Continuing past a failure would defeat the point of having it.
        if outcome == "failed" and stage.name.startswith("Pre-flight"):
            logger.error("")
            logger.error("Pre-flight failed. Nothing was generated, and no GPU "
                         "time was spent. Fix the reported check and re-run.")
            sys.exit(1)
        if outcome == "failed" and not stage.optional:
            logger.error("")
            logger.error("Stopping: %s failed and later stages depend on it. "
                         "Re-run with --verbose to see why.", stage.name)
            sys.exit(1)

    logger.info("")
    logger.info("%d ok, %d skipped, %d failed, %.1fs total",
                outcomes["ok"], outcomes["skipped"], outcomes["failed"], total)
    if outcomes["failed"]:
        sys.exit(1)

    logger.info("")
    logger.info("Numbers in docs/report/numbers/, figures in docs/figures/.")
    logger.info("Experiments with a labelled history are not run here; see the "
                "module docstring for why and how to run them.")


if __name__ == "__main__":
    main()
