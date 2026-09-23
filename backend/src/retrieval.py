"""
Retrieval layer for Phase 2.

The Phase 1 results located the system's factual failures precisely: hallucination
sits at 43.8% on screening instrument questions and 0% on the safety categories.
Retrieval is the intended fix, so the retriever needs to be evaluated on its own
before any generation runs. If it cannot surface the right passage, grounding
cannot repair those errors, and no amount of prompt engineering downstream will
help.

The backend is pluggable. TF-IDF is the default because it is deterministic,
inspectable and needs no GPU, which makes it a fair baseline to measure a dense
embedding retriever against rather than an arbitrary starting point.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Passage:
    """A single indexed chunk of source text."""

    passage_id: str
    text: str
    source_name: str
    source_url: str = ""
    meta: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    """
    Split text into overlapping word-count chunks.

    Overlap prevents a fact that straddles a chunk boundary from being lost, which
    matters here because the facts that matter most, such as an age range or a
    scoring cutoff, are short and easily split.
    """
    words = str(text).split()
    if len(words) <= chunk_size:
        return [" ".join(words)] if words else []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


class TfidfBackend:
    """Sparse lexical retrieval over the indexed passages."""

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        """Create an unfitted TF-IDF backend."""
        self.ngram_range = ngram_range
        self._vec = None
        self._matrix = None

    def fit(self, texts: list[str]) -> "TfidfBackend":
        """Build the term-document matrix."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(ngram_range=self.ngram_range, sublinear_tf=True)
        self._matrix = self._vec.fit_transform(texts)
        return self

    def scores(self, query: str):
        """Return a similarity score against every indexed passage."""
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vec.transform([query])
        return cosine_similarity(q, self._matrix)[0]


class EmbeddingBackend:
    """
    Dense retrieval using a sentence-transformer encoder.

    Kept behind the same interface as TfidfBackend so the two can be compared
    without changing any evaluation code.
    """

    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Create an unfitted embedding backend."""
        self.model_name = model_name
        self._model = None
        self._embeddings = None

    def fit(self, texts: list[str]) -> "EmbeddingBackend":
        """Encode and store the passage embeddings."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        return self

    def scores(self, query: str):
        """Return cosine similarity against every indexed passage."""
        q = self._model.encode([query], normalize_embeddings=True)
        return (self._embeddings @ q.T).ravel()


class Retriever:
    """
    Passage retriever over a corpus of source-grounded chunks.

    Exposes the interface the Phase 2 benchmark runner expects, so it can be
    passed straight into run_benchmark once generation is wired up.
    """

    def __init__(self, backend=None):
        """Create a retriever with the given backend, defaulting to TF-IDF."""
        self.backend = backend or TfidfBackend()
        self.passages: list[Passage] = []

    def index(self, passages: list[Passage]) -> "Retriever":
        """Index a corpus of passages."""
        if not passages:
            raise ValueError("Cannot index an empty passage list")
        self.passages = passages
        self.backend.fit([p.text for p in passages])
        return self

    def _neighbours_of(self, passage: Passage) -> list[Passage]:
        """
        Return the passages immediately before and after one in its document.

        Passage ids are built as {source_id}_c{index} by scripts/build_corpus.py,
        so document order is recoverable from the id alone.
        """
        if not hasattr(self, "_by_id"):
            self._by_id = {p.passage_id: p for p in self.passages}
        match = re.match(r"^(.*)_c(\d+)$", passage.passage_id)
        if not match:
            return []
        stem, index = match.group(1), int(match.group(2))
        found = []
        for offset in (1, -1):
            neighbour = self._by_id.get(f"{stem}_c{index + offset}")
            if neighbour is not None:
                found.append(neighbour)
        return found

    def expand_with_neighbours(self, hits: list[tuple[Passage, float]],
                               k: int) -> list[tuple[Passage, float]]:
        """
        Interleave each hit with its neighbouring chunks, up to a budget of k.

        A question often matches the chunk that names a topic while the answer
        sits in the chunk after it. P025 asks about medication treatment,
        retrieves the chunk carrying that page title, and misses the next chunk,
        which is the one stating which drugs are FDA approved (F-P2-011).

        The budget is the same k as plain retrieval, so this trades lower-ranked
        matches for the immediate context of higher-ranked ones rather than
        making the prompt longer.
        """
        chosen: list[tuple[Passage, float]] = []
        seen: set[str] = set()
        for passage, score in hits:
            for candidate in [passage] + self._neighbours_of(passage):
                if candidate.passage_id in seen:
                    continue
                seen.add(candidate.passage_id)
                # Neighbours inherit a fraction of the score purely so the
                # ordering stays stable; nothing downstream reads the value.
                chosen.append((candidate, score if candidate is passage else score * 0.5))
                if len(chosen) == k:
                    return chosen
        return chosen

    def retrieve(self, query: str, k: int = 3, exclude_ids: set[str] | None = None,
                 max_per_source: int = 2, expand_neighbours: bool = False) -> list[tuple[Passage, float]]:
        """
        Return the top k passages for a query, highest score first.

        exclude_ids supports held-out evaluation: when a query was derived from a
        specific passage, that passage is excluded so a retrieval hit reflects
        finding the right source rather than recovering the answer key.

        max_per_source caps how many passages any single source may occupy in the
        result. Without it, a long document crowds out short authoritative ones
        purely because it contributes more passages. On the 425-passage corpus,
        the M-CHAT FAQ (16 passages) took all six top slots for a scoring
        question while the authoritative M-CHAT scoring page (4 passages, and the
        only source stating the 0-2 / 3-7 / 8-20 bands) ranked 20th. Set to 0 to
        disable.
        """
        scores = self.backend.scores(query)
        ranked = sorted(zip(self.passages, scores), key=lambda t: -t[1])
        if exclude_ids:
            ranked = [(p, s) for p, s in ranked if p.passage_id not in exclude_ids]

        if not max_per_source:
            hits = ranked[:k]
            return self.expand_with_neighbours(hits, k) if expand_neighbours else hits

        picked: list[tuple[Passage, float]] = []
        per_source: dict[str, int] = {}
        for p, s in ranked:
            src = p.meta.get("source_id") or p.source_name
            if per_source.get(src, 0) >= max_per_source:
                continue
            picked.append((p, s))
            per_source[src] = per_source.get(src, 0) + 1
            if len(picked) == k:
                break
        # If the cap left us short, top up from the remaining ranked passages.
        if len(picked) < k:
            chosen = {p.passage_id for p, _ in picked}
            picked += [(p, s) for p, s in ranked
                       if p.passage_id not in chosen][:k - len(picked)]
        if expand_neighbours:
            return self.expand_with_neighbours(picked, k)
        return picked

    def as_context(self, query: str, k: int = 3) -> str:
        """
        Return retrieved passages formatted for injection into a system prompt.

        Each passage carries its source name so the generated answer can be
        attributed, which is required for the grounding claim to be checkable.
        """
        hits = self.retrieve(query, k=k)
        blocks = [f"[Source: {p.source_name}]\n{p.text}" for p, _ in hits]
        return "\n\n".join(blocks)


def load_corpus(csv_path: str) -> list[Passage]:
    """
    Load the corpus built by scripts/build_corpus.py.

    This is the corpus used for real grounding: full source pages fetched from the
    urls in data/corpus/sources.yaml, rather than the benchmark's extracted
    answers.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"passage_id", "text", "source_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s): {', '.join(sorted(missing))}. "
            "Rebuild it with scripts/build_corpus.py."
        )
    return [
        Passage(
            passage_id=r["passage_id"],
            text=r["text"],
            source_name=r["source_name"],
            source_url=r.get("source_url", ""),
            meta={"source_id": r.get("source_id", ""),
                  "authority": r.get("authority", "3")},
        )
        for _, r in df.iterrows()
    ]


def build_corpus_from_benchmark(df, chunk_size: int = 60, overlap: int = 15) -> list[Passage]:
    """
    Build an indexable corpus from a benchmark CSV's reference answers.

    Note on scope. This corpus is the extracted reference answers, not the full
    source documents. It therefore supports a discrimination test, meaning can the
    retriever pick the correct passage out of a pool of topically similar ones,
    but not a full grounding test. The full test requires the source pages and is
    scheduled with the Phase 2 corpus build.
    """
    passages: list[Passage] = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["reference_answer"], chunk_size, overlap)
        for j, ch in enumerate(chunks):
            passages.append(Passage(
                passage_id=f"{row['prompt_id']}_c{j}",
                text=ch,
                source_name=row.get("source_name", ""),
                source_url=row.get("source_url", ""),
                meta={"prompt_id": row["prompt_id"], "category": row.get("category", "")},
            ))
    return passages
