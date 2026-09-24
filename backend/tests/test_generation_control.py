"""
Tests for completion-scoped repetition control.

Two things have to be true at once, and they pull in opposite directions:

  1. Quoting a retrieved source must not be treated as repetition, because that
     is what broke the Phase 2 RAG runs.
  2. The model repeating itself must still be caught, because that is what the
     settings were added for in Phase 1.

The tests below check both, against the stock HuggingFace processors as the
reference for what the old behaviour was.

Run standalone, no pytest:

  python tests/test_generation_control.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from transformers.generation.logits_process import (  # noqa: E402
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

from src.generation_control import (  # noqa: E402
    CompletionOnlyNoRepeatNGram,
    CompletionOnlyRepetitionPenalty,
    build_logits_processors,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("tests.generation_control")

VOCAB = 50
PASSED = []
FAILED = []


def test(func):
    """Register a test function to be run by main()."""
    PASSED.append(func)
    return func


def _blank_scores(batch: int = 1) -> torch.FloatTensor:
    """A flat score vector, so any change is attributable to the processor."""
    return torch.zeros((batch, VOCAB))


def _banned(scores: torch.FloatTensor) -> set[int]:
    """The token ids a processor set to negative infinity."""
    return set(torch.nonzero(torch.isinf(scores[0])).flatten().tolist())


# --------------------------------------------------------------------------
# The behaviour that broke Phase 2
# --------------------------------------------------------------------------

@test
def test_stock_processor_bans_the_continuation_of_a_quoted_prompt_phrase():
    """Document the old behaviour: quoting the prompt gets the next token banned."""
    # Prompt contains the phrase 1 2 3 4 5 6. The completion has reproduced
    # 1 2 3 4 5 and the natural continuation is 6.
    prompt = [1, 2, 3, 4, 5, 6, 7]
    completion = [1, 2, 3, 4, 5]
    ids = torch.tensor([prompt + completion])
    banned = _banned(NoRepeatNGramLogitsProcessor(6)(ids, _blank_scores()))
    assert 6 in banned, f"expected the stock processor to ban token 6, banned {banned}"


@test
def test_completion_scoped_allows_the_continuation_of_a_quoted_prompt_phrase():
    """The fix: the same quotation is legal, because the prompt is not consulted."""
    prompt = [1, 2, 3, 4, 5, 6, 7]
    completion = [1, 2, 3, 4, 5]
    ids = torch.tensor([prompt + completion])
    processor = CompletionOnlyNoRepeatNGram(6, prompt_length=len(prompt))
    banned = _banned(processor(ids, _blank_scores()))
    assert 6 not in banned, f"token 6 should be allowed, banned {banned}"
    assert not banned, f"nothing should be banned here, banned {banned}"


@test
def test_completion_scoped_still_bans_the_models_own_repetition():
    """The protection Phase 1 added survives: a self-repeated n-gram is blocked."""
    prompt = [40, 41, 42]
    # The completion has written 1 2 3 4 5 6 and is back at 1 2 3 4 5.
    completion = [1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5]
    ids = torch.tensor([prompt + completion])
    processor = CompletionOnlyNoRepeatNGram(6, prompt_length=len(prompt))
    banned = _banned(processor(ids, _blank_scores()))
    assert 6 in banned, f"expected token 6 to be banned as self-repetition, got {banned}"


@test
def test_completion_scoped_matches_stock_when_the_prompt_is_empty():
    """With no prompt to exclude, the fix must be identical to the stock processor."""
    completion = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5]
    ids = torch.tensor([completion])
    stock = _banned(NoRepeatNGramLogitsProcessor(6)(ids, _blank_scores()))
    scoped = _banned(
        CompletionOnlyNoRepeatNGram(6, prompt_length=0)(ids, _blank_scores())
    )
    assert stock == scoped, f"stock banned {stock}, scoped banned {scoped}"


@test
def test_short_completions_ban_nothing():
    """A completion shorter than the n-gram width cannot have repeated anything."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1, 2]])
    processor = CompletionOnlyNoRepeatNGram(6, prompt_length=7)
    assert not _banned(processor(ids, _blank_scores()))


# --------------------------------------------------------------------------
# The repetition penalty, which has the same scope problem
# --------------------------------------------------------------------------

@test
def test_stock_penalty_punishes_vocabulary_that_only_appears_in_the_prompt():
    """Document the old behaviour: source terminology is penalised on sight."""
    prompt = [11, 12, 13]
    ids = torch.tensor([prompt + [20]])
    scores = torch.ones((1, VOCAB))
    out = RepetitionPenaltyLogitsProcessor(1.5)(ids, scores)
    assert out[0, 11] < 1.0, "expected a prompt-only token to be penalised"


@test
def test_completion_scoped_penalty_leaves_prompt_vocabulary_alone():
    """The fix: only what the model itself has written is penalised."""
    prompt = [11, 12, 13]
    ids = torch.tensor([prompt + [20]])
    scores = torch.ones((1, VOCAB))
    out = CompletionOnlyRepetitionPenalty(1.5, prompt_length=len(prompt))(ids, scores)
    assert out[0, 11] == 1.0, f"prompt token should be untouched, got {out[0, 11]}"
    assert out[0, 20] < 1.0, f"generated token should be penalised, got {out[0, 20]}"


@test
def test_completion_scoped_penalty_handles_negative_scores_like_huggingface():
    """Negative scores are multiplied, not divided, or the penalty inverts."""
    ids = torch.tensor([[5, 6, 7]])
    scores = torch.full((1, VOCAB), -2.0)
    out = CompletionOnlyRepetitionPenalty(1.5, prompt_length=0)(ids, scores)
    assert out[0, 5] == -3.0, f"expected -3.0, got {out[0, 5]}"


@test
def test_completion_scoped_penalty_matches_stock_when_the_prompt_is_empty():
    """With no prompt to exclude, the fix must be identical to the stock processor."""
    ids = torch.tensor([[5, 6, 7, 5]])
    a = RepetitionPenaltyLogitsProcessor(1.2)(ids, torch.ones((1, VOCAB)))
    b = CompletionOnlyRepetitionPenalty(1.2, 0)(ids, torch.ones((1, VOCAB)))
    assert torch.allclose(a, b), "scoped penalty diverges from stock on an empty prompt"


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

@test
def test_builder_returns_both_processors_for_a_full_config():
    """A config with both settings gets both processors, in a defined order."""
    config = {"no_repeat_ngram_size": 6, "repetition_penalty": 1.15}
    processors = build_logits_processors(config, prompt_length=10)
    assert len(processors) == 2, f"expected 2 processors, got {len(processors)}"
    assert isinstance(processors[0], CompletionOnlyNoRepeatNGram)
    assert isinstance(processors[1], CompletionOnlyRepetitionPenalty)


@test
def test_builder_returns_nothing_for_sequence_scope():
    """Under sequence scope the settings go to generate() instead, not here."""
    config = {"no_repeat_ngram_size": 6, "repetition_penalty": 1.15,
              "repetition_scope": "sequence"}
    assert len(build_logits_processors(config, 10)) == 0


@test
def test_builder_omits_processors_the_config_does_not_ask_for():
    """A config with neither setting reproduces unconstrained generation."""
    assert len(build_logits_processors({}, 10)) == 0
    assert len(build_logits_processors({"repetition_penalty": 1.1}, 10)) == 1


@test
def test_unknown_scope_names_the_valid_options():
    """An invalid scope must fail loudly, since the two scopes differ measurably."""
    try:
        build_logits_processors({"repetition_scope": "prompt"}, 10)
    except ValueError as exc:
        assert "completion" in str(exc) and "sequence" in str(exc), \
            f"error should list the valid scopes, said: {exc}"
    else:
        raise AssertionError("expected a ValueError for an unknown scope")


@test
def test_invalid_construction_arguments_are_rejected():
    """Bad widths and penalties are configuration errors, not silent no-ops."""
    for bad in (0, -1):
        try:
            CompletionOnlyNoRepeatNGram(bad, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ngram_size={bad} should raise")
    try:
        CompletionOnlyRepetitionPenalty(0.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("penalty=0 should raise")


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
