"""
Exercise the whole generation path on CPU before booking a GPU.

A DeepNet booking is hours. Every code path this pipeline uses can be run first
on a tiny model on a laptop, and the failures worth catching are not subtle ones
about model quality: they are a keyword argument the installed transformers
version does not accept, a retriever option that never reaches run_benchmark, a
crash on the fallback prompt format. All of those surface in ten seconds here.

What this checks:

  1. generate() accepts the completion-scoped logits processors and produces
     text, on both repetition scopes.
  2. The n-gram constraint actually reaches the sampler, verified by making the
     prompt and the forced continuation collide.
  3. The corpus on this machine is loadable and its passage ids parse, which
     neighbour expansion depends on.
  4. Every option is threaded from the command line through to the function
     that uses it, and every name the results dict reads is bound where it is
     read. Both fail silently, and only after the model has loaded.
  5. Every enabled model carries repetition_scope, without which the run
     reproduces the truncation it is meant to fix.

It does not check output quality. The model is gpt2 and its answers are
nonsense; the point is that the plumbing holds.

Usage:

  python scripts/preflight.py
  python scripts/preflight.py --model sshleifer/tiny-gpt2
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.generation_control import build_logits_processors  # noqa: E402
from src.model_runner import generate_response  # noqa: E402
from src.retrieval import Retriever, load_corpus  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = BASE_DIR / "data" / "corpus" / "corpus.csv"

CHECKS: list = []
FAILURES: list[str] = []


def check(func):
    """Register a pre-flight check."""
    CHECKS.append(func)
    return func


def _load(model_name: str):
    """Load a small model on CPU, the same way but without quantisation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@check
def check_generation_runs_under_both_scopes(model, tokenizer):
    """generate() must accept the processors and return text under either scope."""
    for scope in ("completion", "sequence"):
        config = {"max_new_tokens": 24, "temperature": 0.1, "do_sample": True,
                  "repetition_penalty": 1.15, "no_repeat_ngram_size": 6,
                  "repetition_scope": scope}
        text, latency, prompt_tokens, completion_tokens = generate_response(
            "What is the M-CHAT used for?", "You are a screening assistant.",
            tokenizer, model, config,
        )
        assert isinstance(text, str), f"scope={scope} returned {type(text)}"
        assert prompt_tokens > 0, f"scope={scope} recorded {prompt_tokens} prompt tokens"
        assert completion_tokens > 0, f"scope={scope} generated nothing"
        assert latency > 0, f"scope={scope} recorded no latency"
        logger.info("    scope=%-10s %d prompt tokens, %d generated",
                    scope, prompt_tokens, completion_tokens)


@check
def check_the_constraint_reaches_the_sampler(model, tokenizer):
    """
    The processors must actually be applied, not silently dropped.

    Built directly and called on a sequence whose completion repeats an n-gram,
    because a processor that generate() ignores would still pass the check
    above.
    """
    import torch

    prompt_length = 4
    sequence = [1, 2, 3, 4] + [10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14]
    processors = build_logits_processors(
        {"no_repeat_ngram_size": 6, "repetition_penalty": 1.15}, prompt_length
    )
    assert len(processors) == 2, f"expected 2 processors, got {len(processors)}"
    scores = torch.zeros((1, 100))
    for processor in processors:
        scores = processor(torch.tensor([sequence]), scores)
    assert torch.isinf(scores[0, 15]), \
        "the n-gram constraint did not ban the repeated continuation"


@check
def check_corpus_is_usable(model, tokenizer):
    """The corpus must exist here and carry ids neighbour expansion can parse."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No corpus at {CORPUS_PATH}. It is gitignored, so build it on this "
            f"machine first: python scripts/build_corpus.py"
        )
    passages = load_corpus(str(CORPUS_PATH))
    parseable = sum(1 for p in passages if "_c" in p.passage_id)
    assert parseable == len(passages), (
        f"{len(passages) - parseable} passage ids do not match "
        f"{{source}}_c{{index}}, so neighbour expansion cannot find neighbours"
    )
    retriever = Retriever().index(passages)
    plain = retriever.retrieve("What is the M-CHAT used for?", k=5)
    expanded = retriever.retrieve("What is the M-CHAT used for?", k=5,
                                  expand_neighbours=True)
    assert len(plain) == len(expanded) == 5, \
        f"expected 5 passages either way, got {len(plain)} and {len(expanded)}"
    assert [p.passage_id for p, _ in plain] != [p.passage_id for p, _ in expanded], \
        "neighbour expansion returned the same passages, so the flag does nothing"
    logger.info("    %d passages, expansion changes the retrieved set", len(passages))


@check
def check_the_run_options_are_threaded_through(model, tokenizer):
    """
    Every new option must reach the function that uses it.

    run_benchmark loads its own model from models.yaml, so it cannot be
    exercised here without adding a toy model to that file, and anything added
    there would be picked up by --all and run on the GPU. The failure mode worth
    catching is instead structural: an argument added to one function and not
    passed by its caller, which is silent and would waste the whole booking.
    """
    import inspect

    from src.model_runner import run_benchmark

    parameters = inspect.signature(run_benchmark).parameters
    assert "expand_neighbours" in parameters,         "run_benchmark does not accept expand_neighbours"

    # The results dict is built inside run_benchmark, so a name that is not
    # bound there fails only at run time, after the model is loaded.
    import ast

    source = (BASE_DIR / "src" / "model_runner.py").read_text(encoding="utf-8")
    function = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "run_benchmark")
    bound = {arg.arg for arg in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            bound |= {inner.id for inner in ast.walk(node)
                      if isinstance(inner, ast.Name)}
    for name in ("expand_neighbours", "model_config", "route", "route_guidance"):
        assert name in bound, f"{name} is used in run_benchmark but never bound there"

    runner = (BASE_DIR / "scripts" / "run_phase1.py").read_text(encoding="utf-8")
    assert "expand_neighbours=args.expand_neighbours" in runner,         "run_phase1.py never forwards --expand-neighbours to run_benchmark"
    assert "--expand-neighbours" in runner,         "run_phase1.py does not expose --expand-neighbours"
    for flag, forwarded in (("--routes", "routes=route_map"),
                            ("--arm", "arm=args.arm"),
                            ("--seed", "seed=args.seed"),
                            ("--route-source", "route_source=args.route_source")):
        assert flag in runner, f"run_phase1.py does not expose {flag}"
        assert forwarded in runner,             f"run_phase1.py accepts {flag} but never forwards it to run_benchmark"


@check
def check_every_enabled_model_has_the_repetition_fix(model, tokenizer):
    """
    The truncation fix is a config value, so a model missing it runs the old way.

    Silent and expensive: the run would complete and the responses would be cut
    off exactly as before.
    """
    import yaml

    from src.generation_control import VALID_SCOPES

    models = yaml.safe_load(
        (BASE_DIR / "config" / "models.yaml").read_text(encoding="utf-8")
    )["models"]
    for model_id, config in models.items():
        scope = config.get("repetition_scope")
        assert scope in VALID_SCOPES, (
            f"{model_id} has repetition_scope={scope!r}, which is not one of "
            f"{VALID_SCOPES}. Without 'completion' this run reproduces the "
            f"truncation of F-P2-010."
        )
        logger.info("    %-12s repetition_scope=%s", model_id, scope)


@check
def check_the_grounded_prompt_carries_real_sources(model, tokenizer):
    """The prompt the GPU run builds must contain both the sources and the rules."""
    import yaml

    from src.model_runner import build_grounded_system_prompt

    conditions = yaml.safe_load(
        (BASE_DIR / "config" / "prompts.yaml").read_text(encoding="utf-8")
    )["conditions"]
    retriever = Retriever().index(load_corpus(str(CORPUS_PATH)))
    hits = retriever.retrieve("What is the M-CHAT used for?", k=5,
                              expand_neighbours=True)
    context = "\n\n".join(f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits)

    for name, condition in conditions.items():
        prompt = build_grounded_system_prompt(condition["system_prompt"], context)
        assert "--- SOURCES ---" in prompt, f"{name}: sources block missing"
        assert "when to speak to a professional" in prompt,             f"{name}: the restated caregiver instructions are missing (D-114)"
        assert len(prompt.split()) > 200, f"{name}: prompt suspiciously short"
    logger.info("    grounded prompt is %d words with %d passages",
                len(prompt.split()), len(hits))


@check
def check_route_maps_cover_the_benchmark(model, tokenizer):
    """
    Every prompt must have a route, and every route must have guidance.

    A prompt missing from the map halts the run mid-booking; a route missing
    guidance would be worse, because run_benchmark raises on it only when that
    prompt comes up, which may be forty responses in.
    """
    import pandas as pd

    from src.model_runner import load_route_guidance
    from src.router import ROUTES

    benchmark = pd.read_csv(BASE_DIR / "data" / "benchmark"
                            / "phase1_baseline_benchmark.csv")
    # The benchmark maps only. adversarial_predicted.csv covers a different
    # prompt set by design, and globbing it in here would report the held-out
    # safety set as a coverage failure of the benchmark.
    maps = [p for p in sorted((BASE_DIR / "data" / "route_maps").glob("*.csv"))
            if not p.name.startswith("adversarial")]
    # llm_predicted.csv is absent until router B has classified, which happens
    # inside the same booking. Its absence before the run is expected, not a
    # fault; when it is present it is checked like any other map.
    if not maps:
        raise FileNotFoundError(
            "No route maps in data/route_maps/. Build them first: "
            "python scripts/build_route_maps.py"
        )
    for path in maps:
        frame = pd.read_csv(path)
        assigned = set(frame["prompt_id"])
        missing = set(benchmark["prompt_id"]) - assigned
        assert not missing, f"{path.name} has no route for {sorted(missing)[:5]}"
        for route in frame["route"].dropna().unique():
            # An empty route means the router reached no decision and the turn
            # generates unrouted. That is a real outcome for the LLM router, not
            # a malformed map, so it needs no guidance and must not be looked up.
            if not str(route).strip():
                continue
            # Control blocks are underscore-prefixed and are deliberately not
            # routes: they are selected by an arm, never by the router. They
            # still need guidance that loads.
            if not route.startswith("_"):
                assert route in ROUTES, f"{path.name} uses unknown route {route!r}"
            load_route_guidance(route)
        logger.info("    %-18s %d prompts, %d distinct routes",
                    path.name, len(frame), frame["route"].nunique())


@check
def check_every_arm_writes_a_distinct_filename(model, tokenizer):
    """
    Arms must not collide on disk.

    prepare_grading.py's only human-readable provenance is the source filename,
    and all three retrieval arms carry rag_enabled=1.
    """
    def tag(retriever, top_k, arm):
        return "_".join(p for p in (f"rag{top_k}" if retriever else "", arm) if p)

    stems = {tag(None, 5, "baseline"), tag(True, 5, "rag"),
             tag(True, 5, "routed_a"), tag(True, 5, "routed_b"),
             tag(True, 5, "union")}
    assert len(stems) == 5, f"arms collide on disk: {sorted(stems)}"
    logger.info("    distinct stems: %s", ", ".join(sorted(stems)))


@check
def check_a_routed_prompt_keeps_its_safety_rules(model, tokenizer):
    """
    Route guidance must specialise the rules, never displace them.

    Checked for every route because a single route losing the safety block would
    be invisible in the aggregate and catastrophic in the one place it mattered.
    """
    import yaml

    from src.model_runner import build_grounded_system_prompt, load_route_guidance
    from src.router import ROUTES

    conditions = yaml.safe_load(
        (BASE_DIR / "config" / "prompts.yaml").read_text(encoding="utf-8")
    )["conditions"]
    retriever = Retriever().index(load_corpus(str(CORPUS_PATH)))
    hits = retriever.retrieve("What is the M-CHAT used for?", k=5,
                              expand_neighbours=True)
    context = "\n\n".join(f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits)

    for condition in conditions.values():
        for route in list(ROUTES) + ["_union", "_placebo"]:
            prompt = build_grounded_system_prompt(
                condition["system_prompt"], context, load_route_guidance(route))
            assert "--- SOURCES ---" in prompt, f"{route}: sources block missing"
            assert "when to speak to a professional" in prompt, \
                f"{route}: displaced the caregiver restatement"
            assert "never" in prompt.lower() or "not" in prompt.lower(), \
                f"{route}: safety rules missing"
    logger.info("    all %d routes plus 2 control blocks keep rules and sources",
                len(ROUTES))


@check
def check_the_longest_routed_prompt_still_fits(model, tokenizer):
    """
    The worst-case prompt must fit the context window.

    Few-shot plus five expanded passages plus the union block is the longest
    string this pipeline can build. Silent left-truncation would drop the system
    prompt, which is where every safety rule lives.
    """
    import yaml

    from src.model_runner import build_grounded_system_prompt, load_route_guidance

    conditions = yaml.safe_load(
        (BASE_DIR / "config" / "prompts.yaml").read_text(encoding="utf-8")
    )["conditions"]
    retriever = Retriever().index(load_corpus(str(CORPUS_PATH)))
    longest = 0
    for prompt_text in ("What is the M-CHAT used for?",
                        "What are the treatments for autism?"):
        hits = retriever.retrieve(prompt_text, k=5, expand_neighbours=True)
        context = "\n\n".join(f"[Source: {p.source_name}]\n{p.text}"
                              for p, _ in hits)
        built = build_grounded_system_prompt(
            conditions["few_shot"]["system_prompt"], context,
            load_route_guidance("_union"))
        longest = max(longest, len(tokenizer(built)["input_ids"]))
    # Mistral-7B-Instruct-v0.1 has the smallest window of the two models at 8192.
    assert longest < 7000, (
        f"worst-case routed prompt is {longest} tokens on this tokenizer, which "
        f"leaves too little room for the answer within an 8192 window"
    )
    logger.info("    worst-case routed prompt: %d tokens", longest)


@check
def check_the_adversarial_run_can_start(model, tokenizer):
    """
    The reference-free path must load, and only under its own flag.

    The adversarial set has no reference_answer, source_name, source_url or
    answer_origin. Without the exemption validate_benchmark_csv raises before
    the model loads, which is a wasted booking; with the exemption applied by
    default a real benchmark that lost its reference column would run for two
    hours and produce responses nothing can be scored against.
    """
    from src.benchmark_schema import BenchmarkValidationError, validate_benchmark_csv

    adversarial = BASE_DIR / "data" / "benchmark" / "router_adversarial_v1.csv"
    try:
        validate_benchmark_csv(str(adversarial))
    except BenchmarkValidationError:
        pass
    else:
        raise AssertionError("the reference-free set validated without --no-reference")

    frame = validate_benchmark_csv(str(adversarial), allow_missing_reference=True)
    assert len(frame) == 20, f"expected 20 adversarial prompts, found {len(frame)}"
    logger.info("    %d prompts load under --no-reference, %d attack families",
                len(frame), frame["attack_family"].nunique())


@check
def check_the_adversarial_set_is_still_held_out(model, tokenizer):
    """
    None of the twenty may have entered the router's training data.

    D-023 holds this set out permanently. If a prompt leaked into the extended
    set, the routed arm would be routing prompts the router had been fitted on,
    and the 13-of-20 figure the experiment rests on would be meaningless. The
    extended set has grown since that decision, so this is checked rather than
    assumed.
    """
    import pandas as pd

    from src.router_eval import load_labelled_prompts

    adversarial = pd.read_csv(BASE_DIR / "data" / "benchmark"
                              / "router_adversarial_v1.csv", dtype=str)
    # Topic rather than audited: the audited track applies v3's per-prompt
    # corrections and refuses to load against benchmark v4 (F-V4-007). Which
    # track is used makes no difference to this check — it compares prompt text,
    # and the labels play no part.
    training = load_labelled_prompts(taxonomy="topic", dataset="extended")
    overlap = set(adversarial["prompt"]) & set(training["prompt"])
    assert not overlap, f"held-out prompts found in training: {sorted(overlap)[:3]}"
    logger.info("    %d adversarial prompts, none of the %d training prompts",
                len(adversarial), len(training))


@check
def check_a_misrouted_safety_turn_still_refuses(model, tokenizer):
    """
    The seven the router misses must still meet the refusal rules.

    This is the whole safety claim: routing is one layer, and the system prompt
    beneath it forbids diagnosing regardless of which route was chosen. If a
    misrouted turn built a prompt without rule 1, the experiment would be
    measuring a hole this check could have found for free. Verified on the
    actual misrouted prompts rather than a hypothetical one.
    """
    import pandas as pd
    import yaml

    from src.model_runner import build_grounded_system_prompt, load_route_guidance

    route_map = BASE_DIR / "data" / "route_maps" / "adversarial_predicted.csv"
    if not route_map.exists():
        raise FileNotFoundError(
            f"{route_map} not built. Run: "
            f"python scripts/build_route_maps.py --adversarial"
        )
    routes = pd.read_csv(route_map)
    missed = routes[routes["route"] != "safety_deflect"]

    conditions = yaml.safe_load(
        (BASE_DIR / "config" / "prompts.yaml").read_text(encoding="utf-8")
    )["conditions"]
    retriever = Retriever().index(load_corpus(str(CORPUS_PATH)))
    hits = retriever.retrieve("Can you tell me if my child has autism?", k=5,
                              expand_neighbours=True)
    context = "\n\n".join(f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits)

    for route in sorted(set(missed["route"])):
        for condition in conditions.values():
            prompt = build_grounded_system_prompt(
                condition["system_prompt"], context, load_route_guidance(route))
            assert "NEVER provide, imply, or suggest a clinical" in prompt, \
                f"{route}: the diagnosis prohibition is missing"
            assert "only a qualified clinician can diagnose" in prompt, \
                f"{route}: the decline-and-explain rule is missing"
    logger.info("    %d misrouted turns land on %s; every one keeps the "
                "refusal rules", len(missed), ", ".join(sorted(set(missed["route"]))))


@check
def check_the_route_maps_match_the_current_benchmark(model, tokenizer):
    """
    The maps must have been built from the benchmark that is about to run.

    Benchmark v4 reuses all 52 of v3's prompt ids and 49 of them now ask a
    different question, so a stale route map joins cleanly on prompt_id and
    attaches the wrong route to almost every prompt. Nothing downstream raises,
    the run completes, and the routed arm is silently meaningless. Checked on
    prompt count and id set, which is enough to catch a map built from a
    different revision.
    """
    import pandas as pd

    benchmark = pd.read_csv(BASE_DIR / "data" / "benchmark"
                            / "phase1_baseline_benchmark.csv")
    ids = set(benchmark["prompt_id"])
    optional = {"llm_predicted.csv"}
    for name in ("predicted.csv", "oracle.csv", "union.csv", "llm_predicted.csv"):
        path = BASE_DIR / "data" / "route_maps" / name
        if name in optional and not path.exists():
            logger.info("    %s not built yet; router B classifies during the "
                        "run", name)
            continue
        assert path.exists(), (
            f"{name} is missing. Rebuild the maps for this benchmark: "
            f"python scripts/build_route_maps.py"
        )
        frame = pd.read_csv(path)
        extra = set(frame["prompt_id"]) - ids
        assert not extra and len(frame) == len(benchmark), (
            f"{name} has {len(frame)} rows for a {len(benchmark)}-prompt "
            f"benchmark. It was built from a different revision; rebuild it: "
            f"python scripts/build_route_maps.py"
        )
    logger.info("    all 3 maps match the %d-prompt benchmark", len(benchmark))


@check
def check_the_llm_router_can_classify(model, tokenizer):
    """
    The LLM routing path must run end to end before a booking is spent on it.

    Exercises the real path — build the classification prompt, generate under
    the label budget, parse the output — on the small CPU model. What is checked
    is that the machinery works and that a nonsense answer is handled, not that
    the answer is right: gpt2 knows nothing about routing and its output should
    come back UNPARSEABLE. A crash here, or a parser that raised on prose, would
    otherwise surface two hundred prompts into a GPU run.
    """
    import pandas as pd

    from src.llm_router import UNPARSEABLE, VARIANTS, LLMRouter, parse_route
    from src.router import ROUTES

    benchmark = pd.read_csv(BASE_DIR / "data" / "benchmark"
                            / "phase1_baseline_benchmark.csv")
    longest = max(benchmark["prompt"], key=len)

    for variant in VARIANTS:
        router = LLMRouter(model, tokenizer, {"max_new_tokens": 8,
                                              "temperature": 0.1, "top_p": 0.9},
                           variant=variant)
        route, decided_by = router.route(longest)
        assert route in ROUTES or route == UNPARSEABLE, \
            f"{variant} produced {route!r}, which is neither a route nor UNPARSEABLE"
        assert decided_by == "llm", f"{variant} attributed the decision to {decided_by}"

        # classify() is the path the evaluation actually takes, because it is
        # the one that returns the raw completion. The raw text is the only
        # evidence for why an unparseable output was unparseable, and losing it
        # would not crash anything: the run would finish, the CSV would look
        # fine, and the question would need another booking to answer. Checked
        # here for the same reason as everything else in this function.
        classified, _by, raw = router.classify(longest)
        assert isinstance(raw, str), f"{variant} returned {type(raw)} as raw output"
        assert classified == parse_route(raw), \
            f"{variant} recorded raw text that does not parse to its own route"

    # The parser must never raise, whatever the model emits.
    for junk in ("", "   ", "\n\n", "I'm sorry, I can't help with that.",
                 "referral general_knowledge", "Label:", "```json\n{}\n```"):
        assert parse_route(junk) in ROUTES + [UNPARSEABLE]
    logger.info("    both variants classify and parse; longest prompt is %d chars",
                len(longest))


@check
def check_the_classification_prompt_fits_the_window(model, tokenizer):
    """
    The few-shot classification prompt plus the longest question must fit.

    Smaller than the generation prompts by a wide margin, but checked rather
    than assumed: the route definitions are long, few-shot adds nine examples,
    and silent left-truncation would drop the definitions and leave the model
    guessing from examples alone.
    """
    import pandas as pd

    from src.llm_router import load_router_prompt

    benchmark = pd.read_csv(BASE_DIR / "data" / "benchmark"
                            / "phase1_baseline_benchmark.csv")
    longest = max(benchmark["prompt"], key=len)

    worst = 0
    for variant in ("zero_shot", "few_shot"):
        system, template = load_router_prompt(variant)
        combined = system + "\n" + template.format(prompt=longest)
        worst = max(worst, len(tokenizer(combined)["input_ids"]))
    assert worst < 3000, (
        f"the worst-case classification prompt is {worst} tokens, which is far "
        f"larger than expected for a labelling task and should be investigated"
    )
    logger.info("    worst-case classification prompt: %d tokens", worst)


@check
def check_the_demo_chat_builds_and_streams_a_turn(model, tokenizer):
    """
    scripts/chat.py must build a real turn and stream an answer before a meeting.

    Two halves, because gpt2's 1024-token window cannot take a real grounded
    prompt. The turn is built at full size from the real corpus and router-free
    settings, which checks the prompt path; the streaming path is then exercised
    through the demo's own stream_answer on a short prompt. A streamer that never
    receives its stop signal hangs the demo rather than failing it, which is the
    fault worth catching here and not in front of an audience.
    """
    import importlib.util
    from types import SimpleNamespace

    from src.chat import (DemoSettings, load_cite_instruction, load_concise_instruction,
                          load_condition_prompt, prepare_turn)

    retriever = Retriever().index(load_corpus(str(CORPUS_PATH)))
    turn = prepare_turn("Who is allowed to administer the M-CHAT?",
                        DemoSettings(use_router=False, cite=True),
                        load_condition_prompt("few_shot"), retriever=retriever)
    assert "--- SOURCES ---" in turn.system_prompt, "demo turn has no sources block"
    assert load_cite_instruction() in turn.system_prompt, \
        "cite mode did not append its instruction"
    assert turn.system_prompt.endswith(load_concise_instruction()), \
        "the demo's default concise style was not appended last"
    assert all(source["url"] for source in turn.sources), \
        "a retrieved source has no url, so the demo would print no link for it"

    spec = importlib.util.spec_from_file_location(
        "demo_chat", BASE_DIR / "scripts" / "chat.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    short = SimpleNamespace(question="What is the M-CHAT?",
                            system_prompt="Answer in one sentence.")
    response, stats = demo.stream_answer(
        short, model, tokenizer,
        {"max_new_tokens": 12, "temperature": 0.1, "do_sample": True},
        1, demo.Style(False))
    assert isinstance(response, str) and stats["completion_tokens"] > 0, \
        "the demo streamed nothing"
    logger.info("    demo turn is %d words with %d sources; streaming returned %d "
                "tokens", len(turn.system_prompt.split()), len(turn.sources),
                stats["completion_tokens"])


@check
def check_no_script_references_an_undefined_name(model, tokenizer):
    """
    Static scan for undefined names across src/, scripts/ and run_all.py.

    This exact fault has now cost real time twice. A module-level constant used
    inside a function but never defined imports cleanly, passes every test that
    does not call that function, and raises the moment a run reaches it: once as
    `config.get` where the variable was `model_config`, and once as `EVAL_DIR`
    in build_route_maps, which would have failed while building the LLM route
    map partway through a booking.

    Only undefined names are treated as failures. Unused imports and shadowed
    locals are style, and this file re-imports pandas inside checks on purpose.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", "src", "scripts", "run_all.py", "tests"],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    if "No module named pyflakes" in result.stderr:
        logger.warning("    pyflakes is not installed; skipping the static scan")
        return
    undefined = [line for line in result.stdout.splitlines()
                 if "undefined name" in line.lower()]
    assert not undefined, "undefined names found: " + " | ".join(undefined)
    logger.info("    no undefined names in src/, scripts/, tests/ or run_all.py")


def main() -> None:
    """Run every pre-flight check and report whether the GPU run is safe to start."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2",
                        help="Small CPU model to exercise the path with")
    args = parser.parse_args()

    logger.info("Loading %s on CPU", args.model)
    model, tokenizer = _load(args.model)

    for func in CHECKS:
        name = func.__name__.replace("check_", "").replace("_", " ")
        try:
            func(model, tokenizer)
            logger.info("PASS  %s", name)
        except Exception as exc:
            FAILURES.append(name)
            logger.error("FAIL  %s: %s", name, exc)

    logger.info("")
    if FAILURES:
        logger.error("%d check(s) failed. Do not start the GPU run.", len(FAILURES))
        sys.exit(1)
    logger.info("All checks passed. The generation path is safe to run on GPU.")


if __name__ == "__main__":
    main()
