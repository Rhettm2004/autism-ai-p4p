"""
Live terminal demo: ask the screening assistant a question and watch it decide.

Each turn shows three panels: what router A decided and why, which passages
retrieval pulled, and the model's answer streaming in, followed by the sources
as numbered links. The model is sent the grounded prompt the routed arm was
measured with (llama3-8b, few_shot, top_k 5), plus, by default, an unevaluated
concise-style instruction suited to a chat app. --measured or /concise off
removes it, and the prompt is then exactly the evaluated one. See docs/DEMO.md
for the meeting runbook.

Usage:

  # The full demo, on a CUDA machine (the uni GPU node)
  python scripts/chat.py

  # Backup on a laptop with no GPU: router, retrieval, sources and the prompt,
  # without loading the model
  python scripts/chat.py --no-llm

  # Start with the experimental inline citations switched on
  python scripts/chat.py --cite

  # Start with the exact evaluated prompt, without the concise-style instruction
  python scripts/chat.py --measured

  # Another model or condition from config/models.yaml and config/prompts.yaml
  python scripts/chat.py --model mistral-7b --condition zero_shot

In the chat, type a question or a command:

  /examples        list the scripted demo questions      /ex 3   ask one
  /prompt          show the full system prompt of the last turn
  /cite on|off     inline [n] citations (experimental, not evaluated)
  /concise on|off  short chat-app replies (on by default, not evaluated)
  /router on|off   route-conditioned guidance on or off
  /rag on|off      retrieval on or off (off = few-shot baseline)
  /help            /quit

This is an interactive front end, so it writes to stdout directly; the logger
is kept for diagnostics and is quiet unless --verbose is given.
"""

import argparse
import logging
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import (  # noqa: E402
    ADOPTED_CONFIG, ROUTER_TRACK, CitationStream, DemoSettings, TranscriptWriter,
    load_condition_prompt, load_demo_router, load_examples,
    order_sources_by_citation, parse_command, prepare_turn, renumber_citations,
    tier_label,
)
from src.model_runner import load_route_guidance  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CORPUS_PATH = "data/corpus/corpus.csv"
WIDTH = 88


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

class Style:
    """ANSI styling that switches itself off for NO_COLOR and redirected output."""

    CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
             "blue": "34", "magenta": "35", "cyan": "36"}

    def __init__(self, enabled: bool):
        """Enable colour only when asked and the terminal can show it."""
        self.enabled = enabled

    def __call__(self, text: str, *names: str) -> str:
        """Wrap text in the named styles."""
        if not self.enabled or not names:
            return text
        codes = ";".join(self.CODES[n] for n in names)
        return f"\033[{codes}m{text}\033[0m"


def out(text: str = "") -> None:
    """Write a line to the terminal immediately."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def header(title: str, s: Style, colour: str = "cyan") -> None:
    """Print a panel header."""
    out()
    out(s(f"── {title} " + "─" * max(4, WIDTH - len(title) - 4), "bold", colour))


def field_line(label: str, value: str, s: Style) -> None:
    """Print an aligned label and value, wrapping long values."""
    lines = textwrap.wrap(value, WIDTH - 16) or [""]
    out(f"  {s(label.ljust(12), 'dim')}  {lines[0]}")
    for line in lines[1:]:
        out(" " * 16 + line)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def render_banner(args, s: Style) -> None:
    """Print what is running and what the audience should know about it."""
    out()
    out(s("Autism screening assistant: live demo", "bold"))
    model = "not loaded (--no-llm)" if args.no_llm else args.model
    field_line("model", model, s)
    field_line("condition", args.condition, s)
    field_line("retrieval", f"TF-IDF over {CORPUS_PATH}, top_k {args.top_k}"
               + (", neighbour expansion" if args.expand_neighbours else ""), s)
    field_line("router", f"router A, {ADOPTED_CONFIG} fitted on the extended set, "
               f"{ROUTER_TRACK} track", s)
    out()
    if args.measured:
        out(s("  Sending the exact grounded prompt the routed arm was measured with.",
              "dim"))
    else:
        out(s("  Concise chat-app style is on: an unevaluated instruction appended to "
              "the measured prompt.", "dim"))
        out(s("  /concise off sends exactly the prompt the routed arm was measured "
              "with.", "dim"))
    out(s("  Each question is answered on its own, with no conversation memory, as "
          "in the evaluation.", "dim"))
    out(s("  Routing is live here; the evaluated arms replayed out-of-fold decisions "
          "(D-119).", "dim"))
    out()
    out("Type a question, " + s("/examples", "bold") + " for the demo script, or "
        + s("/help", "bold") + ".")


def render_help(s: Style) -> None:
    """Print the command list."""
    header("COMMANDS", s)
    for command, meaning in (
        ("/examples", "list the scripted demo questions"),
        ("/ex N", "ask demo question N"),
        ("/prompt", "show the full system prompt and question of the last turn"),
        ("/cite on|off", "inline [n] citations: experimental, not evaluated"),
        ("/concise on|off", "short chat-app replies; off sends the measured prompt "
                            "exactly"),
        ("/router on|off", "route guidance on or off, to compare the same question"),
        ("/rag on|off", "retrieval on or off; off is the few-shot baseline"),
        ("/quit", "leave"),
    ):
        field_line(command, meaning, s)


def render_examples(examples: list[dict], s: Style) -> None:
    """Print the numbered demo script."""
    header("DEMO QUESTIONS", s)
    for number, example in enumerate(examples, start=1):
        out(f"  {s(str(number).rjust(2), 'bold')}  {example['question']}")
        for line in textwrap.wrap(" ".join(example["shows"].split()), WIDTH - 6):
            out(" " * 6 + s(line, "dim"))
        out(" " * 6 + s(f"from {example['from']}", "dim"))


def render_router(turn, s: Style) -> None:
    """Print what the router decided, which stage decided it, and what it injects."""
    if turn.route_info is None:
        header("ROUTER (off)", s, "magenta")
        out(s("  No route guidance is injected; the grounded prompt is the plain "
              "retrieval arm.", "dim"))
        return
    info = turn.route_info
    header("ROUTER", s, "magenta")
    stage = {"rule": "rule layer (no model consulted)",
             "classifier": "classifier"}.get(info["decided_by"], info["decided_by"])
    colour = "red" if info["route"] == "safety_deflect" else "green"
    out(f"  {s('route'.ljust(12), 'dim')}  {s(info['route'], 'bold', colour)}"
        f"   decided by {stage}")
    if info["rule"]:
        field_line("rule", info["rule"], s)
    scores = "   ".join(f"{name} {p:.2f}" for name, p in info["top_scores"])
    field_line("classifier", scores, s)
    if turn.guidance_injected:
        field_line("guidance", "this text was added to the system prompt, between the "
                   "grounding rules and the sources (/prompt shows it in place):", s)
        guidance = " ".join(load_route_guidance(info["route"]).split())
        for line in textwrap.wrap(guidance, WIDTH - 20):
            out(" " * 16 + s("│ " + line, "magenta"))
    else:
        field_line("guidance", "not injected: guidance is only added to a grounded "
                   "prompt, and retrieval is off", s)
    if info["seen_in_training"]:
        out(s("  ! This exact text is router training data, so this decision is not "
              "a held-out one.", "yellow"))


SECTION_TITLES = {
    "condition": ("CONDITION PROMPT  safety rules and few-shot examples", "cyan"),
    "grounding": ("GROUNDING RULES  added because retrieval is on", "blue"),
    "route": ("ROUTE GUIDANCE  {route}, added by the router", "magenta"),
    "sources": ("SOURCES  the retrieved passages", "blue"),
    "cite": ("CITATION INSTRUCTION  demo only, /cite", "yellow"),
    "concise": ("CONCISE STYLE  demo only, /concise", "yellow"),
}


def render_prompt(turn, s: Style) -> None:
    """Print the last turn's system prompt as labelled parts, in reading order."""
    route = turn.route_info["route"] if turn.route_info else ""
    order = " → ".join(section["key"] for section in turn.sections)
    out()
    out(s(f"System prompt, in the order the model reads it: {order} → question", "bold"))
    if not turn.guidance_injected:
        reason = ("the router is off" if turn.route_info is None else
                  "retrieval is off, and guidance is only added to a grounded prompt")
        out(s(f"No route guidance in this prompt: {reason}.", "dim"))
    for section in turn.sections:
        title, colour = SECTION_TITLES[section["key"]]
        header(title.format(route=route), s, colour)
        out(section["text"].strip("\n"))
    header("QUESTION  sent as the user message", s)
    out(turn.question)


def render_retrieved(turn, s: Style) -> None:
    """Print the retrieved passages in rank order."""
    if not turn.settings["use_rag"]:
        header("RETRIEVAL (off)", s, "blue")
        out(s("  No sources: the model answers from the few-shot prompt alone.", "dim"))
        return
    header(f"RETRIEVED  top {len(turn.hits)}", s, "blue")
    if not turn.hits:
        out(s("  Nothing retrieved; the model gets the condition prompt alone.", "yellow"))
        return
    # Rank, not source number: in cite mode the sources are renumbered after the
    # answer, in the order it cites them, so a number printed here would not
    # match the list below.
    for rank, (passage, score) in enumerate(turn.hits, start=1):
        passage_id = passage.passage_id
        if len(passage_id) > 40:
            passage_id = passage_id[:37] + "..."
        out(f"  {rank}.  {score:.3f}  "
            f"{s(passage_id.ljust(40), 'dim')}  {passage.source_name}")


def render_sources(turn, s: Style, sources: list[dict] | None = None,
                   invalid: list[int] | None = None) -> None:
    """
    Print one numbered link per source document.

    sources, when given, is the list reordered to match a cited answer: cited
    sources first, then those the model was given but did not cite.
    """
    sources = turn.sources if sources is None else sources
    if not sources:
        return
    header("SOURCES", s, "blue")
    cite_view = any("cited" in source for source in sources)
    if cite_view and not any(source["cited"] for source in sources):
        out(s("  The answer cites none of these sources.", "yellow"))
    for number in invalid or []:
        out(s(f"  ! The answer cites [{number}], which is not one of the sources.",
              "yellow"))
    shown_uncited_heading = False
    for source in sources:
        if (cite_view and not source["cited"] and any(x["cited"] for x in sources)
                and not shown_uncited_heading):
            out(s("  Given to the model but not cited:", "dim"))
            shown_uncited_heading = True
        tier = tier_label(source["authority"])
        flag = s("  ! commercial or blog content (D-125)", "yellow") \
            if source["low_authority"] else ""
        out(f"  {s('[' + str(source['number']) + ']', 'bold')} {source['name']}"
            f"  {s(tier, 'dim')}{flag}")
        url = source["url"]
        if url.startswith(("http://", "https://")):
            out(f"      {s(url, 'cyan')}")
        elif url:
            # Sources that block fetching are compiled by hand under
            # data/corpus/manual/, with their citations inside the file.
            out(f"      {s('compiled reference, citations inside: data/corpus/' + url, 'dim')}")
        else:
            out(f"      {s('(no url recorded)', 'dim')}")
    if not turn.settings["cite"]:
        out(s("  These are the extracts the model was given; the answer is not "
              "annotated sentence by sentence.", "dim"))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_llm(args, s: Style):
    """Load the generation model, with plain-language errors for the usual failures."""
    import torch

    from src.model_runner import load_model_and_tokenizer, load_model_config

    if not torch.cuda.is_available() and not args.allow_cpu:
        out(s("No CUDA GPU is visible, so the 8B model cannot load in 4-bit here.",
              "red"))
        out("Run this on the GPU node, or use the backup:  python scripts/chat.py "
            "--no-llm")
        out("(--allow-cpu forces a full-precision CPU load: tens of GB of RAM and "
            "minutes per answer.)")
        sys.exit(1)

    config = load_model_config(args.model)
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as exc:
        text = str(exc).lower()
        if any(marker in text for marker in ("gated", "401", "403", "access to model",
                                             "restricted")):
            out(s(f"Hugging Face refused access to {config['hf_model_id']}.", "red"))
            out("Accept Meta's licence on the model page, then run:  "
                "huggingface-cli login")
            sys.exit(1)
        raise
    return model, tokenizer, config


def stream_answer(turn, model, tokenizer, config: dict, seed: int | None,
                  s: Style, citations: CitationStream | None = None) -> tuple[str, dict]:
    """
    Generate the answer on a worker thread and print tokens as they arrive.

    citations, in cite mode, renumbers [n] markers in order of first use as the
    text streams. The returned response is the raw model output either way.
    """
    from transformers import TextIteratorStreamer

    from src.model_runner import generate_response

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    result: dict = {}

    def work() -> None:
        """Run generation; always release the streamer, even on failure."""
        try:
            result["value"] = generate_response(
                turn.question, turn.system_prompt, tokenizer, model, config,
                seed=seed, streamer=streamer)
        except Exception as exc:
            result["error"] = exc
            streamer.end()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    out()
    try:
        for chunk in streamer:
            sys.stdout.write(citations.feed(chunk) if citations else chunk)
            sys.stdout.flush()
        if citations is not None:
            sys.stdout.write(citations.flush())
            sys.stdout.flush()
    except KeyboardInterrupt:
        out(s("\n  (display stopped; waiting for the model to finish the turn)",
              "yellow"))
    thread.join()
    out()
    if "error" in result:
        raise result["error"]

    response, latency_ms, prompt_tokens, completion_tokens = result["value"]
    stats = {
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # Shown so the concise instruction's effect on length is visible per turn.
        "words": len(response.split()),
    }
    rate = completion_tokens / (latency_ms / 1000) if latency_ms else 0.0
    out(s(f"  {stats['words']} words, {latency_ms / 1000:.1f}s, {prompt_tokens} "
          f"prompt tokens, {completion_tokens} generated, {rate:.0f} tok/s", "dim"))
    return response, stats


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def startup_step(label: str, func, s: Style):
    """Run one startup step and print how long it took."""
    sys.stdout.write(f"  {label} ... ")
    sys.stdout.flush()
    started = time.perf_counter()
    value = func()
    out(s(f"done ({time.perf_counter() - started:.1f}s)", "green"))
    return value


def quieten_logs(verbose: bool) -> None:
    """Keep library and pipeline logging off the demo screen unless asked for."""
    if verbose:
        return
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(("src", "transformers", "sentence_transformers", "httpx",
                            "huggingface_hub", "accelerate", "bitsandbytes")):
            logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    """Start the demo and answer questions until /quit."""
    parser = argparse.ArgumentParser(
        description="Live terminal demo of the screening assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="llama3-8b", metavar="MODEL_ID",
                        help="Model id from config/models.yaml (default llama3-8b)")
    parser.add_argument("--condition", default="few_shot",
                        help="Condition from config/prompts.yaml (default few_shot)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Passages retrieved per question (default 5, as the "
                             "ladder)")
    parser.add_argument("--expand-neighbours", action="store_true",
                        help="Include each hit's adjacent chunks within top_k")
    parser.add_argument("--corpus", default=CORPUS_PATH, metavar="PATH",
                        help=f"Corpus CSV (default {CORPUS_PATH})")
    parser.add_argument("--seed", type=int,
                        help="Fix sampling: turn N uses seed + N")
    parser.add_argument("--cite", action="store_true",
                        help="Start with experimental inline citations on")
    parser.add_argument("--measured", action="store_true",
                        help="Start with the concise-style instruction off, so the "
                             "prompt is exactly the evaluated routed arm's")
    parser.add_argument("--no-llm", action="store_true",
                        help="Do not load a model: router, retrieval, sources and "
                             "prompt only")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Load the model on CPU anyway (very slow)")
    parser.add_argument("--no-color", action="store_true", help="Plain output")
    parser.add_argument("--verbose", action="store_true",
                        help="Show pipeline and library logging")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s = Style(enabled=not args.no_color and "NO_COLOR" not in os.environ
              and sys.stdout.isatty())
    quieten_logs(args.verbose)

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    condition_prompt = load_condition_prompt(args.condition)
    if not Path(args.corpus).exists():
        out(s(f"Corpus not found: {args.corpus}", "red"))
        out("It is gitignored, so build it on this machine first:  "
            "python scripts/build_corpus.py")
        sys.exit(1)

    out(s("Starting up", "bold"))
    from src.retrieval import Retriever, load_corpus

    retriever = startup_step(
        f"indexing corpus {args.corpus}",
        lambda: Retriever().index(load_corpus(args.corpus)), s)
    router, training_texts = startup_step(
        f"fitting router A ({ADOPTED_CONFIG}, {ROUTER_TRACK} track)",
        load_demo_router, s)
    examples = load_examples()

    model = tokenizer = model_config = None
    if not args.no_llm:
        model, tokenizer, model_config = startup_step(
            f"loading {args.model}", lambda: load_llm(args, s), s)
        quieten_logs(args.verbose)
        from src.model_runner import generate_response

        startup_step("warm-up generation", lambda: generate_response(
            "Hello", condition_prompt, tokenizer, model,
            {**model_config, "max_new_tokens": 4}), s)

    settings = DemoSettings(cite=args.cite, concise=not args.measured, top_k=args.top_k,
                            expand_neighbours=args.expand_neighbours)
    writer = TranscriptWriter()
    render_banner(args, s)

    last_turn = None
    turn_index = 0
    while True:
        try:
            line = input("\n" + s("caregiver > ", "bold", "green"))
        except (EOFError, KeyboardInterrupt):
            out()
            break
        if not line.strip():
            continue

        try:
            command = parse_command(line)
        except ValueError as exc:
            out(s(f"  {exc}", "yellow"))
            continue

        question = line.strip()
        if command is not None:
            if command.name in ("quit", "exit"):
                break
            if command.name == "help":
                render_help(s)
                continue
            if command.name == "examples":
                render_examples(examples, s)
                continue
            if command.name == "prompt":
                if last_turn is None:
                    out(s("  No turn yet.", "yellow"))
                    continue
                render_prompt(last_turn, s)
                continue
            if command.name in ("cite", "concise", "router", "rag"):
                attribute = {"cite": "cite", "concise": "concise",
                             "router": "use_router", "rag": "use_rag"}[command.name]
                setattr(settings, attribute, command.arg == "on")
                out(s(f"  /{command.name} {command.arg}", "green"))
                if settings.cite and not settings.use_rag:
                    out(s("  Citations need retrieval; /cite has no effect while "
                          "/rag is off.", "yellow"))
                continue
            if command.name == "ex":
                if command.arg > len(examples):
                    out(s(f"  There are {len(examples)} demo questions; see "
                          f"/examples.", "yellow"))
                    continue
                question = examples[command.arg - 1]["question"]
                out(s(f"  {question}", "bold"))

        turn_index += 1
        turn = prepare_turn(question, settings, condition_prompt, router,
                            training_texts, retriever)
        last_turn = turn
        render_router(turn, s)
        render_retrieved(turn, s)

        response, stats = "", {}
        if model is None:
            header("ANSWER (model not loaded)", s, "yellow")
            out(s("  --no-llm: /prompt shows exactly what the model would be sent.",
                  "dim"))
        else:
            notes = []
            if settings.concise:
                notes.append("concise style")
            if settings.cite and settings.use_rag:
                notes.append("inline citations")
            title = "ANSWER"
            if notes:
                title += f"  {' + '.join(notes)}: not the evaluated prompt"
            header(title, s, "green")
            seed = None if args.seed is None else args.seed + turn_index
            source_numbers = {source["number"] for source in turn.sources}
            citations = (CitationStream(source_numbers)
                         if settings.cite and turn.sources else None)
            try:
                response, stats = stream_answer(turn, model, tokenizer, model_config,
                                                seed, s, citations)
            except Exception as exc:
                out(s(f"  Generation failed: {exc}", "red"))
                stats = {"error": str(exc)}

        shown, invalid = None, None
        if model is not None and settings.cite and turn.sources and response:
            displayed, mapping, invalid = renumber_citations(response, source_numbers)
            shown = order_sources_by_citation(turn.sources, mapping)
            # The raw output is kept as response; what the audience read, and how
            # its numbers map back to the prompt's, is kept beside it.
            stats.update({"response_displayed": displayed,
                          "citation_map": {str(k): v for k, v in mapping.items()}})
        render_sources(turn, s, shown, invalid)
        writer.write(turn, response, stats)

    if writer.path.exists():
        out(s(f"Transcript saved to {writer.path}", "dim"))


if __name__ == "__main__":
    main()
