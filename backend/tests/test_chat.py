"""
Tests for the interactive demo layer.

The load-bearing test is the byte-identity one. The demo is only worth showing
if what the audience sees is the system the report measured, and the cheapest
way for that to stop being true is for the chat to build its prompt slightly
differently from run_benchmark. Asserted against the real builder rather than a
copy of its output.

Router fitting loads the MiniLM encoder, so the first run downloads it; no GPU
and no generation model is needed.

Run standalone, no pytest:

  python tests/test_chat.py
"""

import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import (  # noqa: E402
    ADOPTED_CONFIG, COMMANDS, ROUTER_TRACK, CitationStream, DemoSettings,
    TranscriptWriter, build_system_prompt, format_sources, load_cite_instruction,
    load_concise_instruction, load_condition_prompt, load_demo_router, load_examples,
    order_sources_by_citation, parse_command, prepare_turn, prompt_sections,
    renumber_citations,
    route_turn, tier_label,
)
from src.model_runner import (  # noqa: E402
    build_grounded_system_prompt, generate_response, load_route_guidance,
)
from src.retrieval import Passage, Retriever  # noqa: E402
from src.router import ROUTE_SAFETY  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.chat")

BASE_DIR = Path(__file__).parent.parent
PASSED = []
FAILED = []

_ROUTER = None


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _router():
    """Fit the demo router once for the whole run."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = load_demo_router()
    return _ROUTER


def _corpus() -> list[Passage]:
    """A small corpus with a repeated source and one tier-5 source."""
    return [
        Passage("mchat_c0", "The M-CHAT-R is a screening questionnaire for toddlers "
                "aged 16 to 30 months.", "M-CHAT", "https://mchatscreen.com/",
                {"source_id": "mchat", "authority": "3"}),
        Passage("mchat_c1", "M-CHAT-R scoring bands: 0-2 low, 3-7 medium, 8-20 high.",
                "M-CHAT", "https://mchatscreen.com/",
                {"source_id": "mchat", "authority": "3"}),
        Passage("nice_c0", "NICE recommends referral to an autism team when concerns "
                "about development are raised.", "NICE CG128",
                "https://www.nice.org.uk/guidance/cg128",
                {"source_id": "nice", "authority": "1"}),
        Passage("clinic_c0", "Our clinic offers autism screening for toddlers.",
                "Example Clinic", "https://clinic.example/",
                {"source_id": "clinic", "authority": "5"}),
    ]


def _hits(query: str = "M-CHAT screening toddlers", k: int = 4) -> list:
    """Retrieve from the small corpus, uncapped so the repeated source recurs."""
    return Retriever().index(_corpus()).retrieve(query, k=k, max_per_source=0)


# --------------------------------------------------------------------------
# The measurement-integrity tests
# --------------------------------------------------------------------------

@test
def test_default_prompt_is_byte_identical_to_the_evaluated_arm():
    """With cite off, a routed turn must send exactly what run_benchmark sends."""
    condition = load_condition_prompt("few_shot")
    hits = _hits()
    context = "\n\n".join(f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits)
    for route in ("general_knowledge", ROUTE_SAFETY, ""):
        expected = build_grounded_system_prompt(
            condition, context, load_route_guidance(route) if route else None)
        assert build_system_prompt(condition, hits, route) == expected, route


@test
def test_no_retrieval_sends_the_condition_prompt_unchanged():
    """run_benchmark injects guidance only into a grounded prompt; so must the demo."""
    condition = load_condition_prompt("few_shot")
    assert build_system_prompt(condition, [], "referral") == condition


@test
def test_cite_mode_differs_only_by_numbering_and_the_instruction():
    """Cite mode must keep every rule and source, adding labels and one block."""
    condition = load_condition_prompt("few_shot")
    hits = _hits()
    cited = build_system_prompt(condition, hits, "referral", cite=True)
    measured = build_system_prompt(condition, hits, "referral")
    instruction = load_cite_instruction()
    assert cited.endswith(instruction)
    stripped = re.sub(r"\[\d+\] Source: ([^\n]*)", r"[Source: \1]",
                      cited[:-len(instruction)].rstrip())
    assert stripped == measured, "cite mode changed more than the labels"
    assert "NEVER provide, imply, or suggest a clinical" in cited


@test
def test_concise_style_is_appended_after_everything_measured():
    """Concise mode must leave the measured prompt intact and add one block at the end."""
    condition = load_condition_prompt("few_shot")
    hits = _hits()
    instruction = load_concise_instruction()
    for cite in (False, True):
        base = build_system_prompt(condition, hits, "referral", cite=cite)
        concise = build_system_prompt(condition, hits, "referral", cite=cite, concise=True)
        assert concise == f"{base}\n\n{instruction}", f"cite={cite}"
    assert build_system_prompt(condition, [], "referral", concise=True) \
        == f"{condition}\n\n{instruction}"


@test
def test_concise_style_keeps_the_safety_obligations():
    """Brevity must not read as permission to drop a refusal or the next step."""
    text = " ".join(load_concise_instruction().lower().split())
    assert "decline" in text and "next step" in text


@test
def test_demo_defaults_to_concise_and_measured_is_one_switch_away():
    """The chat app style is the default; turning it off gives the evaluated prompt."""
    assert DemoSettings().concise is True
    condition = load_condition_prompt("few_shot")
    retriever = Retriever().index(_corpus())
    question = "What are the M-CHAT-R scoring bands?"
    measured = prepare_turn(question, DemoSettings(use_router=False, concise=False, top_k=3),
                            condition, retriever=retriever)
    styled = prepare_turn(question, DemoSettings(use_router=False, top_k=3),
                          condition, retriever=retriever)
    assert measured.system_prompt == build_system_prompt(condition, measured.hits)
    assert styled.system_prompt == f"{measured.system_prompt}\n\n{load_concise_instruction()}"
    assert styled.settings["concise"] is True and measured.settings["concise"] is False


@test
def test_prompt_sections_rebuild_the_prompt_and_label_the_route_guidance():
    """The labelled parts must join to exactly what was sent, with guidance labelled."""
    condition = load_condition_prompt("few_shot")
    hits = _hits()
    for route in ("", "referral"):
        for cite in (False, True):
            for concise in (False, True):
                prompt = build_system_prompt(condition, hits, route, cite=cite,
                                             concise=concise)
                sections = prompt_sections(prompt, condition, route, cite=cite,
                                           concise=concise)
                assert "".join(x["text"] for x in sections) == prompt
                expected = (["condition", "grounding"] + (["route"] if route else [])
                            + ["sources"] + (["cite"] if cite else [])
                            + (["concise"] if concise else []))
                assert [x["key"] for x in sections] == expected
                if route:
                    guidance = next(x["text"] for x in sections if x["key"] == "route")
                    assert guidance.strip() == load_route_guidance(route)
    unrouted = build_system_prompt(condition, [], "referral", concise=True)
    keys = [x["key"] for x in prompt_sections(unrouted, condition, "", concise=True)]
    assert keys == ["condition", "concise"]


@test
def test_router_constants_match_the_route_map_builder():
    """The demo must route with the configuration the routed arm was built with."""
    script = (BASE_DIR / "scripts" / "build_route_maps.py").read_text(encoding="utf-8")
    assert f'ADOPTED_CONFIG = "{ADOPTED_CONFIG}"' in script
    assert f'ROUTER_TRACK = "{ROUTER_TRACK}"' in script


# --------------------------------------------------------------------------
# Additive changes to existing modules
# --------------------------------------------------------------------------

@test
def test_generate_response_streamer_defaults_to_none():
    """Batch callers never pass a streamer, so its default must leave them unchanged."""
    parameter = inspect.signature(generate_response).parameters["streamer"]
    assert parameter.default is None


@test
def test_route_scores_are_a_distribution_whose_argmax_is_the_route():
    """Without a rule firing, the classifier's top score is the route it returns."""
    router, _ = _router()
    text = "What does the research say about how common this is in girls?"
    scores = router.route_scores(text)
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    route, decided_by = router.route(text)
    assert decided_by == "classifier"
    assert route == max(scores, key=scores.get)


# --------------------------------------------------------------------------
# Routing panel
# --------------------------------------------------------------------------

@test
def test_rule_layer_decides_an_explicit_diagnosis_request():
    """The panel must attribute a rule decision to the rule and name the pattern."""
    router, training = _router()
    info = route_turn(router, training, "Can you tell me if my child has autism?")
    assert info["route"] == ROUTE_SAFETY
    assert info["decided_by"] == "rule"
    assert "can you" in info["rule"]
    assert len(info["top_scores"]) == 3


@test
def test_training_data_is_flagged_and_new_text_is_not():
    """A verbatim training prompt must be flagged as seen by the router."""
    router, training = _router()
    seen = route_turn(router, training, "What is autism spectrum disorder?")
    fresh = route_turn(router, training, "Is a tablet at dinner bad for speech?")
    assert seen["seen_in_training"] is True
    assert fresh["seen_in_training"] is False


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

@test
def test_sources_are_grouped_numbered_linked_and_tier_flagged():
    """One entry per document, numbered by rank, with URL and a tier-5 flag."""
    hits = _hits()
    sources = format_sources(hits)
    names = [s["name"] for s in sources]
    assert len(names) == len(set(names)), "a source appeared twice"
    assert [s["number"] for s in sources] == list(range(1, len(sources) + 1))
    assert sources[0]["name"] == hits[0][0].source_name
    assert all(s["url"].startswith("https://") for s in sources)
    mchat = next(s for s in sources if s["name"] == "M-CHAT")
    assert len(mchat["passage_ids"]) == 2
    clinic = next(s for s in sources if s["name"] == "Example Clinic")
    assert clinic["low_authority"] and not mchat["low_authority"]
    assert "commercial" in tier_label("5") and "guideline" in tier_label("1")
    assert tier_label("") == "tier unrecorded"


@test
def test_cited_numbers_match_the_printed_source_list():
    """[n] in the context must be the number printed beside that source."""
    hits = _hits()
    cited = build_system_prompt(load_condition_prompt("zero_shot"), hits, cite=True)
    for source in format_sources(hits):
        assert f"[{source['number']}] Source: {source['name']}" in cited


@test
def test_citations_are_renumbered_in_order_of_first_use():
    """An answer citing [2] then [3] must read [1] then [2]."""
    text, mapping, invalid = renumber_citations(
        "Screening [2] differs [3]. Again [2].", {1, 2, 3, 4})
    assert text == "Screening [1] differs [2]. Again [1]."
    assert mapping == {2: 1, 3: 2} and invalid == []
    grouped, mapping, _ = renumber_citations("Both [3, 2].", {1, 2, 3})
    assert grouped == "Both [1, 2]." and mapping == {3: 1, 2: 2}


@test
def test_citation_renumbering_leaves_code_lists_and_unknown_numbers_alone():
    """Code, ordinary lists and invented source numbers must pass through."""
    text, mapping, _ = renumber_citations(
        "```python\nx = [1, 2]\n```\nSee [2].", {1, 2})
    assert text == "```python\nx = [1, 2]\n```\nSee [1]."
    listed, mapping, invalid = renumber_citations("numbers = [1, 2, 3, 4, 5]", {1, 2, 3})
    assert listed == "numbers = [1, 2, 3, 4, 5]" and mapping == {} and invalid == []
    wrong, _, invalid = renumber_citations("As shown [7].", {1, 2})
    assert wrong == "As shown [7]." and invalid == [7]


@test
def test_streamed_renumbering_matches_renumbering_the_whole_answer():
    """Whatever the chunk size, the streamed text must equal the final render."""
    answer = "Start [3] then [1, 3] and `code` then ```\n[9]\n``` end [2]. [12"
    expected, _, _ = renumber_citations(answer, {1, 2, 3})
    for size in (1, 2, 3, 5, 100):
        stream = CitationStream({1, 2, 3})
        shown = "".join(stream.feed(answer[i:i + size])
                        for i in range(0, len(answer), size)) + stream.flush()
        assert shown == expected, f"chunk size {size}: {shown!r}"


@test
def test_sources_are_reordered_to_match_renumbered_citations():
    """Cited sources first in citation order, then the rest, counting 1, 2, 3."""
    sources = format_sources(_hits())
    assert len(sources) >= 3
    second, third = sources[1]["number"], sources[2]["number"]
    ordered = order_sources_by_citation(sources, {third: 1, second: 2})
    assert [s["number"] for s in ordered] == list(range(1, len(sources) + 1))
    assert ordered[0]["name"] == sources[2]["name"]
    assert ordered[1]["name"] == sources[1]["name"]
    assert [s["cited"] for s in ordered] == [True, True] + [False] * (len(sources) - 2)
    assert sources[0]["number"] == 1, "the original sources list was modified"


# --------------------------------------------------------------------------
# Commands, examples, transcripts, the no-LLM path
# --------------------------------------------------------------------------

@test
def test_parse_command_accepts_every_command_and_rejects_the_rest():
    """Questions pass through; typos are answered with the valid options."""
    assert parse_command("What is autism?") is None
    assert parse_command("/help").name == "help"
    assert parse_command("/cite ON").arg == "on"
    assert parse_command("/ex 3").arg == 3
    for name, kind in COMMANDS.items():
        line = f"/{name}" + {"on_off": " off", "number": " 1", None: ""}[kind]
        assert parse_command(line).name == name
    for bad in ("/cit on", "/cite maybe", "/ex zero", "/ex 0", "/help me", "/router"):
        try:
            parse_command(bad)
        except ValueError as exc:
            if bad == "/cit on":
                assert "/cite" in str(exc), "unknown command did not list options"
        else:
            raise AssertionError(f"{bad!r} was accepted")


@test
def test_demo_examples_are_verbatim_from_their_named_files():
    """Every scripted question must exist word for word in the file it cites."""
    import pandas as pd

    examples = load_examples()
    assert examples, "config/demo.yaml has no examples"
    for example in examples:
        file, prompt_id = example["from"].rsplit(" ", 1)
        frame = pd.read_csv(BASE_DIR / file, dtype=str, keep_default_na=False)
        row = frame[frame["prompt_id"] == prompt_id]
        assert len(row) == 1, f"{example['from']} not found"
        assert row["prompt"].iloc[0] == example["question"], example["from"]


@test
def test_no_llm_turn_runs_end_to_end_and_is_transcribed():
    """The laptop backup path: route, retrieve, build, record, without a model."""
    router, training = _router()
    retriever = Retriever().index(_corpus())
    settings = DemoSettings(top_k=3)
    turn = prepare_turn("My child scored high on the M-CHAT, what now?", settings,
                        load_condition_prompt("few_shot"), router, training, retriever)
    assert turn.route_info and turn.hits and turn.sources
    assert "--- SOURCES ---" in turn.system_prompt
    assert turn.guidance_injected
    assert "".join(x["text"] for x in turn.sections) == turn.system_prompt
    assert "route" in [x["key"] for x in turn.sections]

    off = prepare_turn(turn.question, DemoSettings(use_router=False, use_rag=False),
                       load_condition_prompt("few_shot"))
    assert off.route_info is None and not off.guidance_injected

    with tempfile.TemporaryDirectory() as directory:
        writer = TranscriptWriter(Path(directory))
        writer.write(turn, "response text", {"latency_ms": 1.0})
        writer.write(off)
        lines = writer.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["system_prompt"] == turn.system_prompt
        assert record["route"]["route"] == turn.route_info["route"]


def main() -> None:
    """Run every registered test and report the tally."""
    for func in list(PASSED):
        try:
            func()
            logger.info("PASS  %s", func.__name__)
        except Exception as exc:
            FAILED.append(func.__name__)
            logger.error("FAIL  %s: %s: %s", func.__name__, type(exc).__name__, exc)
    logger.info("%d passed, %d failed", len(PASSED) - len(FAILED), len(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
