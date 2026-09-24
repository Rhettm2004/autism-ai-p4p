"""
Repetition control that does not punish a model for using its sources.

Phase 1 added `repetition_penalty` and `no_repeat_ngram_size` because two of 176
responses collapsed into repetition loops. Both settings work, and Phase 2 ran
with them. The problem is where HuggingFace applies them: both processors read
the entire sequence, and the prompt is part of the sequence.

That is harmless when the prompt is a question. It is destructive when the
prompt is a question plus five retrieved source passages, because then every
phrase the model quotes from a source is a repeated n-gram, and the token that
would finish the sentence is banned outright. The model, having no legal way to
continue the phrase it is copying, stops. Measured on the Phase 2 runs, 89% of
the responses graded as cut off mid-sentence stopped on a token the n-gram
constraint had banned (F-P2-010).

The degeneracy these settings exist to prevent is repetition inside the model's
own output. Repeating the source material is the intended behaviour of a
retrieval system. So the fix is not to weaken the constraints but to scope them
to the completion, which is what these two processors do.

Both take the prompt length and ignore everything before it. Everything after it
is constrained exactly as the stock processors would constrain it.
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList

# Scope names accepted in models.yaml. "sequence" is the HuggingFace default and
# reproduces the original Phase 1 and Phase 2 generation behaviour exactly.
VALID_SCOPES = ("completion", "sequence")


class CompletionOnlyNoRepeatNGram(LogitsProcessor):
    """
    Ban n-grams the model has already produced, ignoring n-grams in the prompt.

    Equivalent to no_repeat_ngram_size over the generated text alone. A phrase
    quoted from a retrieved passage is not blocked; a phrase the model has
    already written is.
    """

    def __init__(self, ngram_size: int, prompt_length: int):
        """Store the n-gram width and where the prompt ends in the sequence."""
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be at least 1, got {ngram_size}")
        if prompt_length < 0:
            raise ValueError(f"prompt_length cannot be negative, got {prompt_length}")
        self.ngram_size = ngram_size
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Set the score of any token that would repeat a generated n-gram to -inf."""
        scores = scores.clone()
        for batch in range(input_ids.shape[0]):
            generated = input_ids[batch, self.prompt_length:].tolist()
            if len(generated) < self.ngram_size:
                continue
            # Every n-gram the completion has produced so far, keyed by its first
            # n-1 tokens, so the banned continuations of the current prefix are a
            # single lookup.
            seen: dict[tuple[int, ...], set[int]] = {}
            for i in range(len(generated) - self.ngram_size + 1):
                window = generated[i:i + self.ngram_size]
                seen.setdefault(tuple(window[:-1]), set()).add(window[-1])
            prefix = tuple(generated[-(self.ngram_size - 1):]) if self.ngram_size > 1 \
                else ()
            for token in seen.get(prefix, ()):
                scores[batch, token] = -float("inf")
        return scores


class CompletionOnlyRepetitionPenalty(LogitsProcessor):
    """
    Penalise tokens the model has already produced, ignoring the prompt.

    The stock processor penalises every token present in the sequence. Under
    retrieval that penalises the source vocabulary itself, pushing the model away
    from the exact terminology it was given to be accurate about.
    """

    def __init__(self, penalty: float, prompt_length: int):
        """Store the penalty and where the prompt ends in the sequence."""
        if penalty <= 0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        if prompt_length < 0:
            raise ValueError(f"prompt_length cannot be negative, got {prompt_length}")
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Divide positive scores and multiply negative ones, as HuggingFace does."""
        scores = scores.clone()
        for batch in range(input_ids.shape[0]):
            generated = input_ids[batch, self.prompt_length:]
            if generated.numel() == 0:
                continue
            unique = torch.unique(generated)
            selected = scores[batch, unique]
            scores[batch, unique] = torch.where(
                selected > 0, selected / self.penalty, selected * self.penalty
            )
        return scores


def build_logits_processors(config: dict, prompt_length: int) -> LogitsProcessorList:
    """
    Build the completion-scoped processors a config asks for.

    Returns an empty list when the config sets repetition_scope to "sequence",
    because in that case the settings are passed to generate() as keyword
    arguments and handled by the stock processors instead.

    Raises ValueError on an unrecognised scope rather than silently choosing one,
    since the two scopes produce measurably different output.
    """
    scope = config.get("repetition_scope", "completion")
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"repetition_scope must be one of {VALID_SCOPES}, got {scope!r}. "
            f"Use 'completion' to constrain only the generated text, or "
            f"'sequence' to reproduce the original prompt-inclusive behaviour."
        )

    processors = LogitsProcessorList()
    if scope == "sequence":
        return processors

    ngram_size = config.get("no_repeat_ngram_size")
    if ngram_size is not None:
        processors.append(CompletionOnlyNoRepeatNGram(ngram_size, prompt_length))
    penalty = config.get("repetition_penalty")
    if penalty is not None:
        processors.append(CompletionOnlyRepetitionPenalty(penalty, prompt_length))
    return processors
