"""
Interactive demo layer: one caregiver turn through router, retrieval and prompt.

scripts/chat.py is the terminal front end. Everything that decides what the
model is sent lives here instead, so it can be tested on a CPU without torch and
so the `--no-llm` backup mode runs exactly the same path as the full demo.

**The demo is built on the measured system, not a lookalike.** A turn is built
with the same functions `run_benchmark` uses: the same condition prompt, the
same `[Source: name]` context format, the same route guidance, and the router
configuration and training track the route maps were built with. Two
presentation instructions can be appended after that prompt, both demo-only and
unevaluated: `concise`, on by default because answers are read in a chat
window, and `cite`, which numbers the sources and asks for inline citations.
The front end labels either when it is on. With both off, the test suite
asserts the system prompt is byte-identical to the one the batch pipeline would
build, so `--measured` shows exactly the evaluated arm.

**What differs from the evaluation, and must be said when this is shown.**
Routing here is live. The evaluated arms replayed out-of-fold decisions (D-119),
because the benchmark is the router's training data. A question typed verbatim
from that training data is therefore routed by a router that has seen it, and
`route_turn` flags that rather than letting it pass as a clean decision.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from src.model_runner import build_grounded_system_prompt, load_route_guidance
from src.router import DIAGNOSIS_REQUEST_PATTERNS

BASE_DIR = Path(__file__).parent.parent
PROMPTS_YAML = BASE_DIR / "config" / "prompts.yaml"
DEMO_YAML = BASE_DIR / "config" / "demo.yaml"
TRANSCRIPT_DIR = BASE_DIR / "logs" / "demo"

# Must match ADOPTED_CONFIG and ROUTER_TRACK in scripts/build_route_maps.py, which
# built the maps the routed arm was measured with. Duplicated rather than imported
# because scripts/ is not a package; tests/test_chat.py asserts the two agree.
ADOPTED_CONFIG = "embeddings_word"
ROUTER_TRACK = "topic"

# Tier 5 is the commercial and blog content admitted in D-125. Flagged beside the
# link so nobody in the room reads an ABA provider's page as clinical guidance.
LOW_AUTHORITY_TIER = "5"
# Tiers 1-3 are defined in the header of data/corpus/sources.yaml. Tier 4 holds
# autism organisations and charities (the National Autistic Society, Altogether
# Autism), and tier 5 the commercial and blog pages of D-125.
TIER_LABELS = {
    "1": "tier 1 clinical guideline",
    "2": "tier 2 public health body",
    "3": "tier 3 instrument documentation",
    "4": "tier 4 autism organisation",
    "5": "tier 5 commercial or blog",
}

ON_OFF = ("on", "off")
# name -> the argument it takes: None for no argument, "on_off", or "number".
COMMANDS: dict[str, str | None] = {
    "help": None,
    "examples": None,
    "ex": "number",
    "prompt": None,
    "cite": "on_off",
    "concise": "on_off",
    "router": "on_off",
    "rag": "on_off",
    "quit": None,
    "exit": None,
}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def load_condition_prompt(condition: str) -> str:
    """Return the system prompt for a condition from config/prompts.yaml."""
    config = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    conditions = config.get("conditions", {})
    if condition not in conditions:
        raise ValueError(
            f"Condition '{condition}' not found in prompts.yaml. "
            f"Available conditions: {', '.join(conditions)}"
        )
    return conditions[condition]["system_prompt"]


def _load_chat_instruction(key: str, command: str) -> str:
    """Return one instruction from the chat: block of config/prompts.yaml."""
    config = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    try:
        return config["chat"][key].strip()
    except (KeyError, TypeError, AttributeError):
        raise ValueError(
            f"{PROMPTS_YAML} has no chat.{key}, which {command} needs."
        ) from None


def load_cite_instruction() -> str:
    """Return the demo-only inline citation instruction from config/prompts.yaml."""
    return _load_chat_instruction("cite_instruction", "/cite on")


def load_concise_instruction() -> str:
    """Return the demo-only concise chat style instruction from config/prompts.yaml."""
    return _load_chat_instruction("concise_instruction", "/concise on")


def load_examples(path: Path = DEMO_YAML) -> list[dict]:
    """Return the scripted demo questions, each with question, from and shows."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    examples = config.get("examples") or []
    for number, example in enumerate(examples, start=1):
        missing = {"question", "from", "shows"} - set(example)
        if missing:
            raise ValueError(
                f"Example {number} in {path} is missing {sorted(missing)}. Every "
                f"example needs question, from and shows."
            )
    return examples


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def load_demo_router():
    """
    Fit the adopted router on the track the route maps were built with.

    Returns (router, training_texts). The training texts are kept so a turn that
    is verbatim training data can be flagged as routed by a router that saw it.
    """
    from src.router_eval import get_config, load_labelled_prompts

    training = load_labelled_prompts(taxonomy=ROUTER_TRACK, dataset="extended")
    router = get_config(ADOPTED_CONFIG).build().fit(
        training["prompt"].tolist(), training["route"].tolist())
    return router, {str(p).strip() for p in training["prompt"]}


def matched_rule(text: str) -> str:
    """Return the first diagnosis-request pattern the text matches, or ""."""
    lowered = str(text).lower()
    for pattern in DIAGNOSIS_REQUEST_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return ""


def route_turn(router, training_texts: set[str], text: str) -> dict:
    """
    Route one turn and return everything the router panel shows.

    top_scores are the classifier's own probabilities, reported even when the
    rule layer decided, so the audience can see what the classifier alone would
    have done.
    """
    route, decided_by = router.route(text)
    scores = router.route_scores(text)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    return {
        "route": route,
        "decided_by": decided_by,
        "rule": matched_rule(text) if decided_by == "rule" else "",
        "top_scores": [(name, round(p, 4)) for name, p in top],
        "seen_in_training": str(text).strip() in training_texts,
    }


# --------------------------------------------------------------------------
# Sources and prompt construction
# --------------------------------------------------------------------------

def tier_label(authority: str) -> str:
    """Describe a source's authority tier for the sources panel."""
    authority = str(authority).strip()
    if not authority:
        return "tier unrecorded"
    return TIER_LABELS.get(authority, f"tier {authority}")


def _source_key(passage) -> str:
    """Identify a passage's source document, falling back to its name."""
    return passage.meta.get("source_id") or passage.source_name


def format_sources(hits: list) -> list[dict]:
    """
    Collapse retrieved passages into one numbered entry per source document.

    Numbered in order of first appearance, which is retrieval rank, so [1] is the
    source of the best-matching passage. The same numbering is used for the
    context blocks in cite mode, so a citation in the answer points at the entry
    printed under it.
    """
    sources: list[dict] = []
    position: dict[str, int] = {}
    for passage, score in hits:
        key = _source_key(passage)
        if key not in position:
            position[key] = len(sources)
            authority = str(passage.meta.get("authority", ""))
            sources.append({
                "number": len(sources) + 1,
                "name": passage.source_name,
                "url": passage.source_url,
                "authority": authority,
                "low_authority": authority == LOW_AUTHORITY_TIER,
                "best_score": float(score),
                "passage_ids": [],
            })
        entry = sources[position[key]]
        entry["passage_ids"].append(passage.passage_id)
        entry["best_score"] = max(entry["best_score"], float(score))
    return sources


def measured_context(hits: list) -> str:
    """Format passages exactly as run_benchmark does for the evaluated arms."""
    return "\n\n".join(f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits)


def cited_context(hits: list) -> str:
    """Format passages with the source numbers used by format_sources."""
    numbers = {}
    for passage, _ in hits:
        numbers.setdefault(_source_key(passage), len(numbers) + 1)
    return "\n\n".join(
        f"[{numbers[_source_key(p)]}] Source: {p.source_name}\n{p.text}"
        for p, _ in hits
    )


def build_system_prompt(condition_prompt: str, hits: list, route: str = "",
                        cite: bool = False, concise: bool = False) -> str:
    """
    Build the system prompt for one turn.

    Mirrors run_benchmark. With no hits (retrieval off, or nothing retrieved) the
    condition prompt is used unchanged and route guidance is not injected,
    because the batch pipeline only injects guidance into a grounded prompt. With
    hits, and cite and concise both off, the result is byte-identical to the
    evaluated arm.

    The demo-only instructions are appended after everything the arm measured,
    never inserted into it, so switching one off recovers the measured prompt
    exactly. Concise goes last, as the final thing read before the question,
    because it governs how the whole reply is written.
    """
    if not hits:
        prompt = condition_prompt
    else:
        guidance = load_route_guidance(route) if route else None
        context = cited_context(hits) if cite else measured_context(hits)
        prompt = build_grounded_system_prompt(condition_prompt, context, guidance)
        if cite:
            prompt = f"{prompt}\n\n{load_cite_instruction()}"
    if concise:
        prompt = f"{prompt}\n\n{load_concise_instruction()}"
    return prompt


# --------------------------------------------------------------------------
# Inline citations (cite mode only)
# --------------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A citation or code fence that has started but not finished. Held back while
# streaming, because whether it is a citation is not known until it closes.
_UNFINISHED = re.compile(r"(\[[\d,\s]*|`{1,2})$")


def renumber_citations(text: str, valid: set[int]) -> tuple[str, dict[int, int], list[int]]:
    """
    Renumber inline citations in order of first use.

    The prompt numbers sources by retrieval rank, so an answer that leans on the
    second and third extracts cites [2] and [3] and never [1]. This maps the
    first source cited to [1], the next new one to [2], and so on, so the numbers
    a caregiver reads always start at 1 and count up.

    Text inside ``` code fences is left alone, and a bracketed group is only
    renumbered when every number in it is a real source, so a list such as
    [1, 2, 3, 4, 5] is not mistaken for citations. A single bracketed number
    that names no source is left as written and returned, so the panel can say
    the answer cited something that does not exist.

    Returns (renumbered text, {prompt number: displayed number}, invalid numbers).
    """
    mapping: dict[int, int] = {}
    invalid: list[int] = []

    def replace(match: re.Match) -> str:
        """Renumber one bracketed group, or leave it untouched."""
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        if not all(n in valid for n in numbers):
            if len(numbers) == 1 and numbers[0] not in invalid:
                invalid.append(numbers[0])
            return match.group(0)
        for n in numbers:
            mapping.setdefault(n, len(mapping) + 1)
        return "[" + ", ".join(str(mapping[n]) for n in numbers) + "]"

    parts = text.split("```")
    for index in range(0, len(parts), 2):
        parts[index] = _CITATION.sub(replace, parts[index])
    return "```".join(parts), mapping, invalid


class CitationStream:
    """
    Apply renumber_citations to an answer that arrives in chunks.

    Each chunk re-renders everything received so far, less any citation or code
    fence still open at the end, and emits only the new part. Because the
    renumbering depends only on the text before a citation, what has already
    been printed never needs to change.
    """

    def __init__(self, valid: set[int]):
        """Start an empty stream over the given source numbers."""
        self.valid = set(valid)
        self.mapping: dict[int, int] = {}
        self.invalid: list[int] = []
        self._raw = ""
        self._emitted = ""

    def feed(self, chunk: str) -> str:
        """Add a chunk and return the text that is now safe to print."""
        self._raw += chunk
        unfinished = _UNFINISHED.search(self._raw)
        safe = self._raw[:unfinished.start()] if unfinished else self._raw
        return self._emit(safe)

    def flush(self) -> str:
        """Return whatever was held back once the answer is complete."""
        return self._emit(self._raw)

    def _emit(self, text: str) -> str:
        """Render text and return the part not yet printed."""
        rendered, self.mapping, self.invalid = renumber_citations(text, self.valid)
        new = rendered[len(self._emitted):]
        self._emitted = rendered
        return new


def order_sources_by_citation(sources: list[dict], mapping: dict[int, int]) -> list[dict]:
    """
    Renumber the sources list to match the renumbered citations.

    Cited sources come first, numbered as the answer cites them. Sources the
    model was given but did not cite follow, numbered on from there and marked
    cited False, so the list still counts 1, 2, 3 with no gap. The input list is
    not modified.
    """
    by_number = {source["number"]: source for source in sources}
    cited = [dict(by_number[prompt_number], number=shown, cited=True)
             for prompt_number, shown in sorted(mapping.items(), key=lambda kv: kv[1])]
    rest = [source for source in sources if source["number"] not in mapping]
    uncited = [dict(source, number=len(cited) + offset, cited=False)
               for offset, source in enumerate(rest, start=1)]
    return cited + uncited


SOURCES_MARKER = "--- SOURCES ---"
# The parts of a system prompt, in the order build_system_prompt assembles them.
SECTION_KEYS = ("condition", "grounding", "route", "sources", "cite", "concise")


def _find_last(text: str, part: str, name: str) -> int:
    """Return where part last starts in text, or raise naming what is missing."""
    at = text.rfind(part)
    if at == -1:
        raise ValueError(f"The {name} is not in this system prompt")
    return at


def prompt_sections(system_prompt: str, condition_prompt: str, route: str = "",
                    cite: bool = False, concise: bool = False) -> list[dict]:
    """
    Split a built system prompt into labelled parts, for display.

    Exists so a presenter can point at exactly what the router added. Each part
    is found by locating its known text inside the prompt that was actually
    sent, never by rebuilding it, so what is labelled is what the model read,
    and the parts concatenate back to the prompt byte for byte.

    Parts, in order: condition (safety rules and examples), grounding (the
    retrieval instructions), route (the guidance block, only when a route was
    injected), sources, then the demo-only cite and concise instructions.
    Separating blank lines stay attached to the part before them.
    """
    if not system_prompt.startswith(condition_prompt):
        raise ValueError("The system prompt does not start with the condition prompt")
    starts = [(0, "condition")]
    sources_at = system_prompt.find(SOURCES_MARKER)
    if sources_at != -1:
        starts.append((len(condition_prompt), "grounding"))
        if route:
            at = system_prompt.find(load_route_guidance(route), len(condition_prompt),
                                    sources_at)
            if at == -1:
                raise ValueError(f"The guidance for route '{route}' is not in this "
                                 f"system prompt")
            starts.append((at, "route"))
        starts.append((sources_at, "sources"))
        if cite:
            starts.append((_find_last(system_prompt, load_cite_instruction(),
                                      "citation instruction"), "cite"))
    if concise:
        starts.append((_find_last(system_prompt, load_concise_instruction(),
                                  "concise instruction"), "concise"))
    starts.sort()
    ends = [start for start, _ in starts[1:]] + [len(system_prompt)]
    return [{"key": key, "text": system_prompt[start:end]}
            for (start, key), end in zip(starts, ends)]


@dataclass
class DemoSettings:
    """The switches a presenter can flip mid-demo."""

    use_router: bool = True
    use_rag: bool = True
    cite: bool = False
    # On by default because the demo stands in for a chat app. Off reproduces the
    # evaluated prompt exactly.
    concise: bool = True
    top_k: int = 5
    expand_neighbours: bool = False


@dataclass
class PreparedTurn:
    """Everything decided about a turn before the model is asked to answer it."""

    question: str
    route_info: dict | None
    hits: list
    sources: list[dict]
    system_prompt: str
    settings: dict = field(default_factory=dict)
    # The system prompt split into labelled parts; see prompt_sections.
    sections: list[dict] = field(default_factory=list)

    @property
    def guidance_injected(self) -> bool:
        """True when a route's guidance block is actually in the system prompt."""
        return bool(self.route_info and self.route_info["route"] and self.hits)


def prepare_turn(question: str, settings: DemoSettings, condition_prompt: str,
                 router=None, training_texts: set[str] | None = None,
                 retriever=None) -> PreparedTurn:
    """Route, retrieve and build the system prompt for one caregiver turn."""
    route_info = None
    if settings.use_router:
        if router is None:
            raise ValueError("use_router is on but no router was supplied")
        route_info = route_turn(router, training_texts or set(), question)

    hits = []
    if settings.use_rag:
        if retriever is None:
            raise ValueError("use_rag is on but no retriever was supplied")
        hits = retriever.retrieve(question, k=settings.top_k,
                                  expand_neighbours=settings.expand_neighbours)

    route = route_info["route"] if route_info else ""
    system_prompt = build_system_prompt(condition_prompt, hits, route,
                                        cite=settings.cite, concise=settings.concise)
    sections = prompt_sections(system_prompt, condition_prompt, route if hits else "",
                               cite=settings.cite and bool(hits),
                               concise=settings.concise)
    return PreparedTurn(
        question=question,
        route_info=route_info,
        hits=hits,
        sources=format_sources(hits),
        system_prompt=system_prompt,
        sections=sections,
        settings={"use_router": settings.use_router, "use_rag": settings.use_rag,
                  "cite": settings.cite, "concise": settings.concise,
                  "top_k": settings.top_k,
                  "expand_neighbours": settings.expand_neighbours},
    )


# --------------------------------------------------------------------------
# Commands and transcripts
# --------------------------------------------------------------------------

@dataclass
class Command:
    """A parsed slash command."""

    name: str
    arg: str | int | None = None


def parse_command(line: str) -> Command | None:
    """
    Parse a slash command, or return None when the line is a question.

    Raises ValueError naming the valid options for an unknown command or a bad
    argument, so a typo during a demo is answered rather than sent to the model.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split()
    name = parts[0].lower() if parts else ""
    args = parts[1:]
    if name not in COMMANDS:
        raise ValueError(
            f"Unknown command '/{name}'. Available: "
            + ", ".join(f"/{c}" for c in COMMANDS)
        )
    kind = COMMANDS[name]
    if kind is None:
        if args:
            raise ValueError(f"/{name} takes no argument")
        return Command(name)
    if len(args) != 1:
        expected = "on or off" if kind == "on_off" else "a number"
        raise ValueError(f"/{name} needs one argument: {expected}")
    if kind == "on_off":
        if args[0].lower() not in ON_OFF:
            raise ValueError(f"/{name} takes on or off, got '{args[0]}'")
        return Command(name, args[0].lower())
    if not args[0].isdigit() or int(args[0]) < 1:
        raise ValueError(f"/{name} takes a positive number, got '{args[0]}'")
    return Command(name, int(args[0]))


class TranscriptWriter:
    """
    Append one JSON record per demo turn to logs/demo/.

    Kept so anything said in a meeting can be looked up afterwards with the exact
    prompt, sources and route behind it. logs/ is gitignored: a transcript is a
    record of a conversation, not an experimental artefact.
    """

    def __init__(self, directory: Path = TRANSCRIPT_DIR):
        """Choose a timestamped file; it is created on the first write."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(directory) / f"chat_{stamp}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"Transcript already exists: {self.path}")

    def write(self, turn: PreparedTurn, response: str = "",
              stats: dict | None = None) -> None:
        """Append a turn and its response."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "question": turn.question,
            "settings": turn.settings,
            "route": turn.route_info,
            "guidance_injected": turn.guidance_injected,
            "retrieved_passage_ids": [p.passage_id for p, _ in turn.hits],
            "sources": turn.sources,
            "system_prompt": turn.system_prompt,
            "response": response,
            **(stats or {}),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
