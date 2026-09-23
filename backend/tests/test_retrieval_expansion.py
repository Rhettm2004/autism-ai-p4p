"""
Tests for neighbour expansion in the retriever.

The failure this exists for: a question matches the chunk that names a topic
while the answer sits in the chunk after it. Source-level recall calls that a
hit, because the right document was retrieved, and the model still has no answer
in front of it.

Run standalone, no pytest:

  python tests/test_retrieval_expansion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import Passage, Retriever  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.retrieval_expansion")

PASSED = []
FAILED = []


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _corpus() -> list[Passage]:
    """
    A miniature corpus with the shape that caused the failure.

    doc_c0 carries the title words a title-like query matches. doc_c1 carries
    the answer and shares no vocabulary with the query.
    """
    return [
        Passage(passage_id="doc_c0", text="Medication Treatment for Autism overview",
                source_name="Doc", meta={"source_id": "doc"}),
        Passage(passage_id="doc_c1", text="Risperidone is approved for irritability",
                source_name="Doc", meta={"source_id": "doc"}),
        Passage(passage_id="doc_c2", text="Providers usually prescribe on a trial basis",
                source_name="Doc", meta={"source_id": "doc"}),
        Passage(passage_id="other_c0", text="Screening is not diagnosis",
                source_name="Other", meta={"source_id": "other"}),
        Passage(passage_id="other_c1", text="Refer to a specialist for assessment",
                source_name="Other", meta={"source_id": "other"}),
    ]


def _ids(hits) -> list[str]:
    """Passage ids of a retrieval result, in rank order."""
    return [p.passage_id for p, _ in hits]


@test
def test_the_answer_chunk_is_missed_without_expansion():
    """Establish the failure first, so the fix is measured against something."""
    retriever = Retriever().index(_corpus())
    # At k=1 the query matches only the chunk carrying the title words. The
    # answer chunk shares no vocabulary with the query, so it cannot rank.
    hits = retriever.retrieve("Medication Treatment for Autism", k=1)
    assert _ids(hits) == ["doc_c0"], f"expected only the title chunk, got {_ids(hits)}"


@test
def test_expansion_pulls_in_the_adjacent_answer_chunk():
    """The fix: retrieving the title chunk now brings the next chunk with it."""
    retriever = Retriever().index(_corpus())
    hits = retriever.retrieve("Medication Treatment for Autism", k=2,
                              expand_neighbours=True)
    assert "doc_c1" in _ids(hits), \
        f"expected the answer chunk alongside the title chunk, got {_ids(hits)}"


@test
def test_expansion_respects_the_k_budget():
    """Expansion trades lower-ranked hits for context, it does not lengthen the prompt."""
    retriever = Retriever().index(_corpus())
    for k in (1, 2, 3, 4):
        hits = retriever.retrieve("autism", k=k, expand_neighbours=True)
        assert len(hits) <= k, f"k={k} returned {len(hits)} passages"


@test
def test_expansion_never_repeats_a_passage():
    """Adjacent hits share neighbours, so deduplication has to hold."""
    retriever = Retriever().index(_corpus())
    ids = _ids(retriever.retrieve("autism medication", k=5, expand_neighbours=True))
    assert len(ids) == len(set(ids)), f"duplicate passages returned: {ids}"


@test
def test_the_highest_ranked_hit_is_still_first():
    """Expansion must not displace the best match from the top of the context."""
    retriever = Retriever().index(_corpus())
    plain = _ids(retriever.retrieve("Medication Treatment for Autism", k=3))
    expanded = _ids(retriever.retrieve("Medication Treatment for Autism", k=3,
                                       expand_neighbours=True))
    assert expanded[0] == plain[0], \
        f"top hit changed from {plain[0]} to {expanded[0]}"


@test
def test_first_and_last_chunks_of_a_document_are_handled():
    """A chunk at either end of a document has only one neighbour, not zero results."""
    retriever = Retriever().index(_corpus())
    hits = retriever.retrieve("Providers usually prescribe on a trial basis", k=3,
                              expand_neighbours=True)
    assert "doc_c2" in _ids(hits), f"the matching last chunk vanished: {_ids(hits)}"
    assert "doc_c1" in _ids(hits), f"its one neighbour should be present: {_ids(hits)}"


@test
def test_neighbours_come_from_the_same_document_only():
    """Ids encode document membership; expansion must not cross a document boundary."""
    retriever = Retriever().index(_corpus())
    ids = _ids(retriever.retrieve("Screening is not diagnosis", k=2,
                                  expand_neighbours=True))
    assert all(i.startswith("other_") for i in ids), \
        f"expansion crossed into another document: {ids}"


@test
def test_expansion_is_off_by_default():
    """The Phase 2 runs did not use it, so the default has to reproduce them."""
    retriever = Retriever().index(_corpus())
    assert _ids(retriever.retrieve("autism", k=3)) == \
        _ids(retriever.retrieve("autism", k=3, expand_neighbours=False))


@test
def test_passages_with_unparseable_ids_are_left_alone():
    """A corpus not built by build_corpus.py must degrade, not crash."""
    passages = [Passage(passage_id="no_index_here", text="some text",
                        source_name="X", meta={"source_id": "x"})]
    retriever = Retriever().index(passages)
    hits = retriever.retrieve("text", k=3, expand_neighbours=True)
    assert _ids(hits) == ["no_index_here"], f"got {_ids(hits)}"


def main() -> None:
    """Run every registered test and report the tally."""
    for func in list(PASSED):
        try:
            func()
            logger.info("PASS  %s", func.__name__)
        except Exception as exc:
            FAILED.append(func.__name__)
            logger.error("FAIL  %s: %s", func.__name__, exc)
    logger.info("%d passed, %d failed", len(PASSED) - len(FAILED), len(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
