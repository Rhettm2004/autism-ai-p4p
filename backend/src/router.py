"""
Intent router for the Phase 3 agentic layer.

The Phase 3 design splits the assistant into specialised agents. Something has to
decide which agent handles an incoming caregiver turn, and that decision is itself
safety critical: a request for a diagnosis routed to the result-explanation agent
bypasses the refusal behaviour the system depends on.

This module implements that router and, more importantly, makes it measurable.
Every benchmark prompt already carries a category label, so those labels give
ground truth for routing without any additional annotation.

Design note. Phase 1 showed that instruction-level safety constraints held the
diagnosis boundary in all 176 responses, but a learned classifier gives no such
guarantee. The router is therefore hybrid: a rule layer catches explicit requests
for a diagnosis before the classifier is consulted, and the classifier handles the
remaining routing. This mirrors the finding in the agentic literature that
safety-critical domains favour symbolic or hybrid designs over purely neural ones.
"""

import re

ROUTE_SAFETY = "safety_deflect"
ROUTE_MISINFO = "misinformation_correction"
ROUTE_SCREENING = "screening_guidance"
ROUTE_RESULT = "result_explanation"
ROUTE_REFERRAL = "referral"
ROUTE_KNOWLEDGE = "general_knowledge"
ROUTE_SUPPORT = "caregiver_support"

ROUTES = [
    ROUTE_SAFETY, ROUTE_MISINFO, ROUTE_SCREENING, ROUTE_RESULT,
    ROUTE_REFERRAL, ROUTE_KNOWLEDGE, ROUTE_SUPPORT,
]

# Note on the taxonomy. An earlier version of this mapping put every
# safety-sensitive category on the deflect route. Evaluating it showed that this
# conflated two different things: turns that ask the system to judge a specific
# child, which must be deflected, and turns that ask about a sensitive topic such
# as vaccines or an unproven treatment, which must be answered with grounded
# evidence. Deflecting the second group reproduces the over-refusal failure that
# Llama3-8B showed in Phase 1, where it declined to correct a claim about
# high-dose vitamins. The two are now separate routes.

# Maps benchmark category labels onto agent routes. Covers the category
# vocabularies of both the 44-prompt and 52-prompt benchmark versions.
CATEGORY_TO_ROUTE = {
    # requests for a diagnostic judgement about a specific child, and jailbreaks
    "safety_diagnosis_boundary": ROUTE_SAFETY,
    "adversarial_prompts": ROUTE_SAFETY,
    "clinical_testing_boundary": ROUTE_SAFETY,
    # claims needing correction with cited evidence, not deflection
    "safety_vaccines": ROUTE_MISINFO,
    "safety_treatment": ROUTE_MISINFO,
    "safety_vaccine_misinformation": ROUTE_MISINFO,
    "treatment_safety": ROUTE_MISINFO,
    "safety_wandering": ROUTE_MISINFO,
    # screening instrument questions
    "screening_item_clarification": ROUTE_SCREENING,
    "screening_tool_use": ROUTE_SCREENING,
    "screening_limitations": ROUTE_SCREENING,
    "screening_scope": ROUTE_SCREENING,
    "screening_administration": ROUTE_SCREENING,
    "screening_scoring": ROUTE_SCREENING,
    # interpreting a result
    "result_explanation": ROUTE_RESULT,
    "screening_follow_up": ROUTE_RESULT,
    "screening_positive_next_steps": ROUTE_RESULT,
    "screening_negative_concern": ROUTE_RESULT,
    # referral pathway
    "next_steps_referral": ROUTE_REFERRAL,
    "screening_referral": ROUTE_REFERRAL,
    "screening_next_steps": ROUTE_REFERRAL,
    "diagnosis_referral": ROUTE_REFERRAL,
    "recognition_and_diagnosis": ROUTE_REFERRAL,
    "diagnosis_process": ROUTE_REFERRAL,
    "diagnosis_tools": ROUTE_REFERRAL,
    "early_recognition": ROUTE_REFERRAL,
    # general knowledge
    "general_autism_knowledge": ROUTE_KNOWLEDGE,
    "general_health_knowledge": ROUTE_KNOWLEDGE,
    "diagnosis_boundary": ROUTE_KNOWLEDGE,
    "treatment_support": ROUTE_KNOWLEDGE,
    "general_autism_prevalence": ROUTE_KNOWLEDGE,
    "diagnostic_criteria": ROUTE_KNOWLEDGE,
    "causes_and_risk": ROUTE_KNOWLEDGE,
    "related_medical_conditions": ROUTE_KNOWLEDGE,
    "developmental_regression": ROUTE_KNOWLEDGE,
    "intellectual_disability": ROUTE_KNOWLEDGE,
    "adult_diagnosis": ROUTE_KNOWLEDGE,
    "late_diagnosis": ROUTE_KNOWLEDGE,
    # emotional support
    "caregiver_reassurance": ROUTE_SUPPORT,
    "caregiver_support": ROUTE_SUPPORT,
}

# Explicit requests for a diagnostic judgement. These bypass the classifier
# entirely so that routing cannot silently send them elsewhere.
# Character classes exclude sentence-ending punctuation so a pattern cannot match
# across two separate sentences, which produced a false positive on
# "My son is very chatty and social. Can he still have autism?"
DIAGNOSIS_REQUEST_PATTERNS = [
    r"\b(do|does|is|are)\b[^.?!]{0,60}\b(my|my child|he|she|they)\b[^.?!]{0,40}\bhave autism\b",
    r"\bcan you (confirm|tell me|say|diagnose)\b",
    r"\bconfirm (the |an? )?(autism )?diagnos",
    r"\bdo you think (my|he|she|they)\b",
    r"\bis (it|this) autism\b",
    r"\bpretend (you are|to be) a (doctor|clinician|paediatrician|pediatrician)\b",
    r"\bignore (your |all )?(previous |prior )?instructions\b",
    r"\bdiagnos(e|is) (my|this) child\b",
    r"\bwould you say it is autism\b",
    # indirect phrasings observed in the benchmark
    r"\bsurely you can tell me\b",
    r"\bjust confirm it\b",
    r"\bask (the ai|you) for an? (autism )?diagnosis\b",
    r"\btell me (definitively|for sure)\b",

    # Added August 2026 after the extended prompt set showed the patterns above
    # catch 3 of 11 new adversarial phrasings (F-P3-015). Each of these was
    # generalised from one prompt in a five-prompt design set; three further
    # misses were deliberately held back so that the extension could be tested
    # rather than fitted. See F-P3-017 for what that test found.
    #
    # A request for an opinion about this child. Anchored on a reference to the
    # child so that "what do you think about the M-CHAT" is not caught.
    r"\bwhat do you (actually |really |honestly )?think\b[^.?!]{0,40}"
    r"\b(my (child|son|daughter)|him|her|them|his|hers)\b",
    # A "what is going on with (my|his|her|their)" pattern was tried here and
    # removed: it matched "How do I talk to my other children about what is going
    # on with their brother?", a caregiver-support turn, producing the first false
    # alarm the rule layer had ever recorded. The design prompt it was meant to
    # cover is already caught by the pattern above, so it bought nothing and cost
    # the layer its perfect precision.
    # A request for a likelihood rather than a judgement, which is the same
    # judgement with a number attached.
    r"\bhow likely is it\b",
    r"\bput a (percentage|number|figure) on\b",
    r"\bwhat are the (odds|chances)\b",
    # A request for a guess, usually prefaced by absolution.
    r"\b(your|a|an) (best |educated |rough )?guess\b",
    r"\bjust guess\b",
    # An instruction to set the safety framing aside.
    r"\b(forget|skip|drop|ignore|leave out) (the |your )?"
    r"(disclaimer|caveat|warning|hedge|legal)s?\b",
    # A request to rate or score the child, which presupposes the diagnosis and
    # asks for its severity.
    r"\brate (my|his|her|their) (child|son|daughter|kid)\b",
    r"\bfrom (one|1) to (ten|10)\b",
]


def route_for_category(category: str) -> str | None:
    """Return the agent route for a benchmark category label, or None if unmapped."""
    return CATEGORY_TO_ROUTE.get(str(category).strip().lower())


def is_diagnosis_request(text: str) -> bool:
    """
    True when the text explicitly asks for a diagnostic judgement.

    Deliberately tuned to favour recall over precision. Sending a borderline
    question to the safety agent produces an overly cautious answer; missing a
    real diagnosis request produces an unsafe one.
    """
    t = str(text).lower()
    return any(re.search(p, t) for p in DIAGNOSIS_REQUEST_PATTERNS)


class IntentRouter:
    """
    Hybrid intent router.

    A rule layer catches explicit diagnosis requests and routes them to the safety
    agent. Everything else is routed by a TF-IDF classifier fitted on labelled
    benchmark prompts. TF-IDF is used rather than a neural encoder so that routing
    is deterministic, inspectable, and runs without a GPU.
    """

    def __init__(self, use_rules: bool = True, features: str = "word"):
        """
        Create an unfitted router.

        Set use_rules=False to evaluate the classifier alone. features selects the
        text representation: "word", "char", or "word_char" for both.
        """
        if features not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{features}'. Available: {list(FEATURE_SETS)}")
        self.use_rules = use_rules
        self.features = features
        self._pipeline = None

    def fit(self, prompts: list[str], routes: list[str]) -> "IntentRouter":
        """Fit the classifier on labelled prompts."""
        self._pipeline = _text_classifier(features=self.features)
        self._pipeline.fit(prompts, routes)
        return self

    def route(self, text: str) -> tuple[str, str]:
        """
        Route a single incoming turn.

        Returns (route, decided_by) where decided_by is "rule" or "classifier".
        """
        if self.use_rules and is_diagnosis_request(text):
            return ROUTE_SAFETY, "rule"
        if self._pipeline is None:
            raise RuntimeError("Router has not been fitted. Call fit() first.")
        return self._pipeline.predict([text])[0], "classifier"

    def route_scores(self, text: str) -> dict[str, float]:
        """
        Return the classifier's probability for every route, ignoring the rules.

        Diagnostic only: route() is still what decides. When the rule layer fires
        these scores show what the classifier alone would have done, which is
        what makes the rule's contribution visible on a single turn.
        """
        if self._pipeline is None:
            raise RuntimeError("Router has not been fitted. Call fit() first.")
        probabilities = self._pipeline.predict_proba([text])[0]
        return {str(route): float(p)
                for route, p in zip(self._pipeline.classes_, probabilities)}

    def route_batch(self, texts: list[str]) -> list[str]:
        """Route many turns, returning just the route labels."""
        return [self.route(t)[0] for t in texts]


FEATURE_SETS = ("word", "word_char", "char", "embedding", "embedding_word")

# A small sentence encoder, chosen so routing stays runnable on a laptop and on a
# shared GPU node without competing for GPU memory with the generation models.
# 384 dimensions, roughly 80MB, and it encodes the whole prompt set in under a
# second on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The encoder is deterministic and the prompt set is fixed, so a text encoded in
# one fold is the same vector in every other fold. Caching by text turns ten
# repeats of three folds from thirty encodings of the corpus into one.
_ENCODER = None
_EMBEDDING_CACHE: dict[str, "object"] = {}


def _encoder():
    """Load the sentence encoder once per process."""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(EMBEDDING_MODEL)
    return _ENCODER


def embed(texts: list[str]):
    """
    Encode texts as sentence embeddings, caching by text.

    Cross-validation encodes the same prompts many times, and the encoder is
    deterministic, so the cache is a pure speed-up and changes no result.
    """
    import numpy as np

    missing = [t for t in texts if t not in _EMBEDDING_CACHE]
    if missing:
        vectors = _encoder().encode(missing, show_progress_bar=False)
        for text, vector in zip(missing, vectors):
            _EMBEDDING_CACHE[text] = vector
    return np.vstack([_EMBEDDING_CACHE[t] for t in texts])


class SentenceEmbedder:
    """
    A scikit-learn transformer wrapping the sentence encoder.

    Stateless: there is nothing to fit, because the encoder is pretrained and
    frozen. That is the point of using it here. With 163 prompts there is far too
    little data to learn a representation, but a linear probe over a representation
    someone else learned needs only enough data to separate seven routes in a
    space where meaning is already organised.
    """

    def fit(self, X, y=None):
        """Nothing to fit; the encoder is frozen."""
        return self

    def transform(self, X):
        """Encode a list of texts into a matrix of sentence embeddings."""
        return embed(list(X))

    def get_params(self, deep: bool = True) -> dict:
        """Return no parameters; required by the scikit-learn estimator protocol."""
        return {}

    def set_params(self, **params) -> "SentenceEmbedder":
        """Accept no parameters; required by the scikit-learn estimator protocol."""
        return self


def _vectoriser(features: str):
    """
    Build the feature extractor for a routing stage.

    Word features treat a caregiver's vocabulary as the signal, which is the
    obvious choice and the one the router started with. Their weakness at this
    sample size is that a word seen only in the test fold carries no weight at
    all: with 93 prompts, most words appear once or twice, so a paraphrase, a
    misspelling or an inflection the training folds happen not to contain is
    invisible to the model.

    Character n-grams share substrings across those variants, so "diagnosis",
    "diagnose" and "diagnosed" overlap heavily even though they are three
    distinct tokens. char_wb keeps n-grams inside word boundaries, which avoids
    manufacturing features that straddle two unrelated words.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    if features not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{features}'. Available: {list(FEATURE_SETS)}")

    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        stop_words=None,      # caregiver phrasing carries signal in stopwords
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    if features == "word":
        return word
    if features == "char":
        return char
    if features == "embedding":
        return SentenceEmbedder()
    if features == "embedding_word":
        # Embeddings carry meaning and lose exact wording; word features carry the
        # exact wording and no meaning. The union keeps both, which matters for
        # routes distinguished by a specific phrase rather than by topic.
        return FeatureUnion([("embedding", SentenceEmbedder()), ("word", word)])
    return FeatureUnion([("word", word), ("char", char)])


def _text_classifier(class_weight: str | None = "balanced", features: str = "word"):
    """
    Build the text classifier used by every routing stage.

    One definition, used by the flat router and by both stages of the cascade, so
    that a comparison between architectures is a comparison between architectures
    and not between two differently tuned classifiers.

    class_weight and features are the settings a stage may vary. Only the binary
    safety gate varies class_weight. Balanced weighting is right where a tiny class competes inside a
    single decision boundary, but the gate already has an explicit operating
    point in its threshold. Doing both leaves the threshold no usable range: with
    nine safety turns against eighty-four, balanced weighting pushes the safety
    probability so high that every threshold below 0.4 routes almost everything
    to the safety agent.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("features", _vectoriser(features)),
        ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ])


class CascadeRouter:
    """
    Two-stage router: decide safety first, then decide type.

    The flat router asks one seven-way question, which forces the safety decision
    to compete with six others inside a single decision boundary. That is the
    wrong shape for this problem twice over. The safety class is small, so a
    seven-way model has little reason to carve out room for it; and the safety
    decision is not comparable to the others, since routing a diagnosis request
    to the wrong agent is a different kind of failure from routing a referral
    question to the wrong agent.

    The cascade separates them:

      stage 0   the rule layer, unchanged. An explicit request for a diagnostic
                judgement goes straight to the safety agent without any model
                being consulted.
      stage 1   a binary classifier: is this a safety turn or not? On the current
                data this is a 9 against 84 decision rather than one class among
                seven, and it can be biased towards catching safety turns without
                disturbing anything else.
      stage 2   a type classifier over the remaining routes, trained only on
                non-safety turns, so its decision boundary is never distorted by
                a class it will not be asked about.

    The safety threshold is the reason this design exists. Lowering it trades
    precision for recall on the one route where that trade is worth making, and
    it does so without touching how the other six routes are told apart. In the
    flat router there is no equivalent knob.
    """

    def __init__(self, use_rules: bool = True, safety_threshold: float = 0.5):
        """
        Create an unfitted cascade router.

        safety_threshold is the probability above which stage 1 sends a turn to
        the safety agent. Below 0.5 the router is deliberately over-cautious,
        which costs some precision and buys recall on the route where a miss is
        the expensive error.
        """
        if not 0.0 < safety_threshold < 1.0:
            raise ValueError(
                f"safety_threshold must be strictly between 0 and 1, got {safety_threshold}"
            )
        self.use_rules = use_rules
        self.safety_threshold = safety_threshold
        self._safety_gate = None
        self._type_classifier = None
        self._fallback_route: str | None = None

    def fit(self, prompts: list[str], routes: list[str]) -> "CascadeRouter":
        """
        Fit both stages.

        Stage 1 sees every training turn, relabelled as safety or not. Stage 2
        sees only the non-safety turns. If a fold happens to contain no safety
        turns at all, stage 1 is left unfitted and every turn falls through to
        stage 2, which is the correct degenerate behaviour rather than an error.
        """
        binary = [ROUTE_SAFETY if route == ROUTE_SAFETY else "other" for route in routes]
        if len(set(binary)) > 1:
            # No class re-weighting here: the threshold is this stage's operating
            # point, and re-weighting on top of it collapses the usable range.
            self._safety_gate = _text_classifier(class_weight=None)
            self._safety_gate.fit(prompts, binary)

        rest = [(p, r) for p, r in zip(prompts, routes) if r != ROUTE_SAFETY]
        if not rest:
            raise ValueError("Cascade router needs at least one non-safety training turn")
        rest_prompts, rest_routes = zip(*rest)
        # A single remaining class cannot be fitted, and does not need to be.
        if len(set(rest_routes)) > 1:
            self._type_classifier = _text_classifier()
            self._type_classifier.fit(list(rest_prompts), list(rest_routes))
        else:
            self._fallback_route = rest_routes[0]
        return self

    def route(self, text: str) -> tuple[str, str]:
        """
        Route a single turn.

        Returns (route, decided_by), where decided_by is "rule", "safety_gate" or
        "type_classifier", so it can be reported which stage made each decision.
        """
        if self.use_rules and is_diagnosis_request(text):
            return ROUTE_SAFETY, "rule"

        if self._safety_gate is not None:
            classes = list(self._safety_gate.classes_)
            probability = self._safety_gate.predict_proba([text])[0][classes.index(ROUTE_SAFETY)]
            if probability >= self.safety_threshold:
                return ROUTE_SAFETY, "safety_gate"

        if self._type_classifier is None:
            if self._fallback_route is None:
                raise RuntimeError("Router has not been fitted. Call fit() first.")
            return self._fallback_route, "type_classifier"
        return self._type_classifier.predict([text])[0], "type_classifier"

    def route_batch(self, texts: list[str]) -> list[str]:
        """Route many turns, returning just the route labels."""
        return [self.route(t)[0] for t in texts]
