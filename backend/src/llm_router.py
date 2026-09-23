"""
Variant B of the Phase 3 router: an LLM classifier.

Variant A is `src/router.py` — hand-written rule patterns in front of a fitted
scikit-learn classifier. It is untouched by this file. The two exist side by
side so they can be run against each other on identical prompts, which is the
only way to answer whether the fitted router earns its place or whether a
general-purpose model does the job better with no training at all.

**What makes this comparable, and what does not.**

The fitted router is scored by repeated K-fold: each prompt is predicted only
while it is held out. This router is never fitted, so every prompt is held out
by construction and a single pass is already leak-free. The fair head-to-head is
therefore this router's one pass against variant A's out-of-fold predictions
over the same prompts, paired per prompt. `scripts/evaluate_llm_router.py` does
exactly that and refuses to compare against anything else.

What is *not* comparable is cost. Variant A routes a turn in microseconds on a
CPU; this one needs a GPU-resident 7B model and roughly a second per turn. If
the two score the same, variant A wins on every practical ground, and the report
should say so rather than treating a tie as a reason to adopt the larger thing.

**The unparseable outcome is real and is measured.** A classifier that emits
free text can emit something that is not a label. That is not an implementation
detail to be swallowed by a default: a router that fails to decide is a router
that would send a caregiver's turn nowhere. Unparseable outputs are recorded as
`UNPARSEABLE` and reported as their own rate. They are scored as incorrect,
which is the least generous reading and the honest one.

**But an unparseable rate measures the parser as much as the model, so the raw
completion is kept.** The first run of this router reported 39.7% unparseable
and stored nothing but the parsed label, which made the number uninterpretable:
a model emitting prose and a model emitting `general knowledge` that the parser
discarded are the same row in that file. `classify()` therefore returns the raw
decoded text and the evaluation script persists it. Every later question about
this router - which failure mode dominates, whether a wider token budget helps,
whether a different parser would have scored it differently - is then answerable
from the artefact on a CPU instead of from another GPU booking.

**No safety thumb on the scale.** The classification prompt does not tell the
model to prefer the safety route when uncertain. Variant A has no such
instruction either, and adding one here would lift safety recall for a reason
that has nothing to do with being an LLM. A safety-biased variant is a separate
named configuration, not a tweak to this one.
"""

import re
from pathlib import Path

import yaml

from src.router import ROUTE_SAFETY, ROUTES, is_diagnosis_request
from src.utils import get_logger

logger = get_logger(__name__)

PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"

# Recorded when the model's output does not name exactly one route. Never mapped
# to a real route: a router that could not decide must not be scored as though
# it decided, and the rate is a headline result of this approach.
UNPARSEABLE = "UNPARSEABLE"

# The model is asked for a bare label, so a handful of tokens is plenty. Kept
# small deliberately: a long budget invites the model to explain itself, which
# is the main source of unparseable output.
#
# Twelve is not tight on the labels themselves. Under the Mistral tokenizer the
# longest is five tokens (`misinformation_correction`, `result_explanation`,
# `caregiver_support`) and the shortest is two (`referral`), so truncation needs
# a preamble of at least seven tokens before it can cut a label in half. That
# rules truncation out as the main cause of the observed unparseable rate,
# though not as a contributor.
MAX_LABEL_TOKENS = 12

VARIANTS = ("zero_shot", "few_shot")


def load_router_prompt(variant: str = "zero_shot") -> tuple[str, str]:
    """
    Return (system_prompt, user_template) for a classification variant.

    The few-shot system prompt is stored as the zero-shot body plus examples,
    substituted here rather than duplicated in the file. Duplicating forty lines
    of route definitions would let the two variants drift apart, and a
    difference in the definitions would be indistinguishable from a difference
    caused by the examples.
    """
    config = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    block = config.get("llm_router")
    if not block:
        raise ValueError(
            f"{PROMPTS_PATH} has no llm_router: block, so there is no frozen "
            f"classification prompt to route with."
        )
    if variant not in block:
        available = [k for k in block if k != "_meta"]
        raise ValueError(
            f"Unknown LLM router variant {variant!r}. Available: {available}"
        )

    system = block[variant]["system_prompt"]
    if "__ZERO_SHOT_BODY__" in system:
        system = system.replace("__ZERO_SHOT_BODY__",
                                block["zero_shot"]["system_prompt"].strip())
    return system, block[variant]["user_template"]


# One pattern per route, matching the label however the model punctuated it.
# The parts of a label may be joined by an underscore, a space or a hyphen, in
# any case: "general_knowledge", "general knowledge" and "General-Knowledge" all
# name the same route and differ only in typography. Matching only the
# underscore form scored the latter two as UNPARSEABLE, which records a model
# that did decide as a model that did not.
_ROUTE_PATTERNS = {
    route: re.compile(
        r"\b" + r"[\s_\-]+".join(re.escape(part) for part in route.split("_")) + r"\b")
    for route in ROUTES
}


def parse_route(text: str) -> str:
    """
    Extract exactly one route from the model's output, or UNPARSEABLE.

    Deliberately strict in one direction and forgiving in another. Forgiving
    about wrapping: leading whitespace, a trailing full stop, surrounding quotes,
    a "Label:" echo, and the separator between a label's words are all ignored,
    because none of them change which route was named. Strict about ambiguity:
    if the output names two different routes, it is unparseable rather than
    resolved by taking the first, since a model that listed two did not choose
    one.

    Tolerating the separator is a repair to the instrument, not a loosening of
    the scoring. It cannot turn a wrong label into a right one and it cannot
    invent a decision where none was named; it can only stop a decision the
    model did make from being recorded as a failure to decide. It is not purely
    monotone, though: an output naming one route in underscores and a second in
    prose used to parse as the first and is now correctly ambiguous, so the
    unparseable rate is not guaranteed to fall on every prompt.
    """
    named = routes_named(text)
    if len(named) == 1:
        return next(iter(named))
    return UNPARSEABLE


def routes_named(text: str) -> set[str]:
    """
    Return every route the output names, which may be none or several.

    Split out from parse_route so the *reason* an output was unparseable stays
    recoverable. Zero routes named and three routes named are the same
    UNPARSEABLE to a scorer and completely different failures to fix: one is a
    model that answered the message instead of labelling it, the other is a
    model that would not commit. Diagnosis reads this; scoring reads
    parse_route, and the two cannot drift because parse_route is defined in
    terms of this.
    """
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"^(label|answer|category|route)\s*[:\-]\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"\'`').rstrip(".").strip()
    return {route for route, pattern in _ROUTE_PATTERNS.items()
            if pattern.search(cleaned)}


def truncated_label(text: str) -> str | None:
    """
    Return the route an output was cut off partway through naming, if exactly one.

    A label lost to the token budget leaves a proper prefix of itself at the end
    of the output: "safety_defl", "misinformation_correc". If that prefix fits
    exactly one route it identifies which; if it fits several it identifies
    nothing and returns None, because "re" begins both `referral` and
    `result_explanation`.

    Diagnosis only, and deliberately not wired into parse_route. Recovering a
    label the model never finished emitting would score a decision it did not
    make, which is the one thing the unparseable outcome exists to prevent.
    """
    tail = re.split(r"[^\w\-]+", str(text).strip().lower())[-1]
    tail = tail.replace("-", "_")
    if not tail:
        return None
    candidates = {route for route in ROUTES
                  if route.startswith(tail) and route != tail}
    return next(iter(candidates)) if len(candidates) == 1 else None


class LLMRouter:
    """
    Routes a caregiver turn by asking a local instruction-tuned model.

    Exposes fit() and route() so it satisfies the same informal interface as
    IntentRouter and CascadeRouter. fit() is a no-op that records nothing: this
    router does not learn from the training prompts, and pretending otherwise by
    quietly using them for few-shot selection would destroy the leak-free
    property that makes a single pass a fair comparison.
    """

    def __init__(self, model, tokenizer, config: dict,
                 variant: str = "zero_shot", use_rules: bool = False,
                 max_label_tokens: int = MAX_LABEL_TOKENS):
        """
        Build a router around an already-loaded model and tokenizer.

        The model is passed in rather than loaded here so one GPU-resident model
        can serve both the routing pass and the generation runs in a single
        booking, which is the difference between a fifteen-minute experiment and
        an hour of reloading weights.

        use_rules prefixes the same rule layer variant A uses. Off by default,
        so the headline number measures the LLM alone; turned on it isolates
        what the LLM adds on top of the rules, which is the comparison that
        matters if the rules are being kept either way.

        max_label_tokens defaults to the frozen budget so the adopted
        configuration is unchanged by this parameter existing. Raising it is a
        deliberate, recorded choice rather than a tweak, because the budget is
        part of what the variant is. The one thing it does not do is force a
        second booking to compare budgets: generation is a left-to-right
        extension under a fixed seed, so the first MAX_LABEL_TOKENS tokens of a
        longer completion are byte-for-byte the ones a MAX_LABEL_TOKENS run
        would have produced. Record the raw completion at a generous budget and
        the tight-budget result is recoverable offline, on a CPU, with no GPU at
        all.
        """
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Available: {list(VARIANTS)}")
        if max_label_tokens < 1:
            raise ValueError(
                f"max_label_tokens must be at least 1, got {max_label_tokens}. "
                f"The frozen budget is {MAX_LABEL_TOKENS}."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.variant = variant
        self.use_rules = use_rules
        self.max_label_tokens = max_label_tokens
        self.system_prompt, self.user_template = load_router_prompt(variant)

    def fit(self, prompts: list[str], routes: list[str]) -> "LLMRouter":
        """
        Accept training data and ignore it, for interface compatibility.

        Logged rather than silent, so nobody reading an evaluation log concludes
        this router was trained on anything.
        """
        logger.info("LLMRouter.fit ignoring %d training prompts; this router is "
                    "not fitted, which is why one pass over the data is already "
                    "leak-free", len(prompts))
        return self

    def _classify(self, text: str, seed: int | None = None) -> str:
        """Ask the model for one label and return the raw decoded output."""
        # Imported here rather than at module scope so the prompt-building and
        # parsing logic stays importable, and testable, without torch present.
        from src.model_runner import generate_response

        response, _latency, _prompt_tokens, _completion_tokens = generate_response(
            self.user_template.format(prompt=text),
            self.system_prompt,
            self.tokenizer,
            self.model,
            {**self.config, "max_new_tokens": self.max_label_tokens},
            seed=seed,
        )
        return response

    def classify(self, text: str, seed: int | None = None) -> tuple[str, str, str]:
        """
        Route a turn and return the raw completion alongside the decision.

        Returns (route, decided_by, raw), where raw is the model's decoded
        output before parsing, or "" for a turn the rule layer answered without
        asking the model.

        This exists because the first router B run could not be diagnosed. It
        recorded a 39.7% unparseable rate and nothing else, so there was no way
        to tell a model that emitted prose from a model that emitted a correct
        label the parser then threw away, and answering that needed another
        booking rather than another query. The raw text is the whole evidence
        for every claim about *why* this router fails, and it costs a CSV column.

        route() is left at its two-tuple contract deliberately: it is the shared
        informal interface with IntentRouter and CascadeRouter, and widening it
        here would make this router the odd one out at every call site.
        """
        if self.use_rules and is_diagnosis_request(text):
            return ROUTE_SAFETY, "rule", ""
        raw = self._classify(text, seed=seed)
        return parse_route(raw), "llm", raw

    def route(self, text: str, seed: int | None = None) -> tuple[str, str]:
        """
        Route a single turn.

        Returns (route, decided_by), where decided_by is "rule" or "llm", so a
        turn caught by the rule layer stays distinguishable from one the model
        decided. Matches the contract of CascadeRouter.route.
        """
        route, decided_by, _raw = self.classify(text, seed=seed)
        return route, decided_by

    def route_batch(self, texts: list[str], seed: int | None = None) -> list[str]:
        """Route many turns, returning just the route labels."""
        return [self.route(text, seed=seed)[0] for text in texts]
