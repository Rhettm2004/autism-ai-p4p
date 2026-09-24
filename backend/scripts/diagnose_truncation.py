"""
Find out why retrieval-grounded responses stop mid-sentence.

The blind grading pass found that a third of Mistral's RAG responses were cut
off mid-sentence, and the first explanation offered was the generation budget.
That explanation is wrong: no Mistral RAG response reached max_new_tokens, and
the truncated ones average 42 generated tokens out of 512. The model stops
early of its own accord.

This script tests the alternative explanation. Generation runs with
no_repeat_ngram_size=6, and in HuggingFace that constraint applies to the whole
sequence, prompt included. Retrieval puts the source text into the prompt, so
any phrase the model quotes from a source becomes a repeated n-gram and its
continuation is banned. A model grounding its answer in the retrieved text is
therefore forbidden from finishing the sentence it is copying.

The test rebuilds the exact prompt each run saw, replays the recorded response
through the real logits processor, and asks one question at the token where the
response stopped: was the token that would have continued the source passage
banned at that position?

No model weights and no GPU are needed. The tokenizer and the constraint are
enough, because the constraint is a deterministic function of the token ids.

A note on the reconstruction, added 26 August 2026. When this was first run the
rebuilt prompts matched the recorded prompt_tokens exactly, median and maximum
difference both zero. They no longer do: build_grounded_system_prompt gained the
D-114 restatement afterwards, so replaying July runs through today's builder adds
about fifty tokens to every retrieval prompt. The blocked-continuation figures are
unchanged at 34% and 7%, because whether a response's tail appears in the sources
does not depend on an instruction paragraph, but the exactness claim in F-P2-010
belongs to the version of the builder that existed on 25 August.

Usage:

  python scripts/diagnose_truncation.py
  python scripts/diagnose_truncation.py --run mistral-7b_few_shot_rag5
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.generation.logits_process import (  # noqa: E402
    NoRepeatNGramLogitsProcessor,
)

from src.generation_control import CompletionOnlyNoRepeatNGram  # noqa: E402
from src.model_runner import build_grounded_system_prompt  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
RUN_DIR = BASE_DIR / "results_thursday"
CORPUS_PATH = BASE_DIR / "data" / "corpus" / "corpus.csv"
PROMPTS_PATH = BASE_DIR / "config" / "prompts.yaml"
MODELS_PATH = BASE_DIR / "config" / "models.yaml"
OUT_PATH = BASE_DIR / "data" / "history" / "truncation_diagnosis.csv"


def load_corpus() -> dict[str, dict]:
    """Return passage_id -> {text, source_name} for rebuilding retrieved context."""
    frame = pd.read_csv(CORPUS_PATH)
    return {
        row["passage_id"]: {"text": row["text"], "source_name": row["source_name"]}
        for _, row in frame.iterrows()
    }


def rebuild_prompt(row: pd.Series, corpus: dict, system_prompts: dict,
                   tokenizer) -> tuple[str, list[int]]:
    """
    Rebuild the exact text the model was given for one benchmark row.

    Mirrors run_benchmark: passages are joined in retrieval order with their
    source labels, wrapped by build_grounded_system_prompt, then formatted with
    the chat template. Returns the formatted string and its token ids.
    """
    system_prompt = system_prompts[row["condition"]]
    # Baseline rows carry no passage ids, and an empty cell reads back as the
    # string "nan". Grounding on that would add an empty sources block and put
    # 73 tokens of instruction into a prompt that never had them.
    ids = [i for i in str(row["retrieved_passage_ids"]).split("|") if i in corpus]
    if ids:
        context = "\n\n".join(
            f"[Source: {corpus[i]['source_name']}]\n{corpus[i]['text']}"
            for i in ids if i in corpus
        )
        system_prompt = build_grounded_system_prompt(system_prompt, context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["prompt"]},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{row['prompt']} [/INST]"
        )
    return formatted, tokenizer(formatted)["input_ids"]


def banned_at_end(sequence: list[int], ngram_size: int, vocab_size: int,
                  scope: str = "sequence", prompt_length: int = 0) -> set[int]:
    """
    Return the token ids the n-gram constraint forbids at the next position.

    Under "sequence" this is the real processor from the generation stack, not a
    reimplementation, so what it reports is what happened during the run. Under
    "completion" it is the replacement from src/generation_control.py, which lets
    the same recorded responses be replayed against the fix.
    """
    if scope == "sequence":
        processor = NoRepeatNGramLogitsProcessor(ngram_size)
    else:
        processor = CompletionOnlyNoRepeatNGram(ngram_size, prompt_length)
    input_ids = torch.tensor([sequence])
    scores = torch.zeros((1, vocab_size))
    filtered = processor(input_ids, scores)
    return set(torch.nonzero(torch.isinf(filtered[0])).flatten().tolist())


def continuation_token(response_ids: list[int], prompt_ids: list[int],
                       ngram_size: int) -> int | None:
    """
    Find the token that would have continued a passage the response was copying.

    Takes the last ngram_size - 1 tokens of the response, looks for that exact
    run inside the prompt, and returns whatever followed it there. None means
    the response did not end inside a phrase lifted from the prompt.
    """
    if len(response_ids) < ngram_size - 1:
        return None
    tail = response_ids[-(ngram_size - 1):]
    width = len(tail)
    for i in range(len(prompt_ids) - width):
        if prompt_ids[i:i + width] == tail:
            return prompt_ids[i + width]
    return None


def diagnose(run_files: list[Path], ngram_size: int,
             scope: str = "sequence") -> pd.DataFrame:
    """Replay every response in the given runs through the n-gram constraint."""
    corpus = load_corpus()
    prompts = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    system_prompts = {k: v["system_prompt"] for k, v in prompts["conditions"].items()}
    models = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))["models"]

    tokenizers: dict[str, object] = {}
    records = []
    for path in run_files:
        frame = pd.read_csv(path)
        model_id = frame["model_id"].iloc[0]
        if model_id not in tokenizers:
            logger.info("Loading tokenizer for %s", model_id)
            try:
                tokenizers[model_id] = AutoTokenizer.from_pretrained(
                    models[model_id]["hf_model_id"]
                )
            except OSError as exc:
                # Llama 3.1 is a gated repo. Without HF credentials its runs
                # cannot be replayed here, which is a missing measurement rather
                # than a failure: say so and carry on with the models we can read.
                logger.warning("Skipping %s, tokenizer unavailable: %s",
                               path.name, str(exc).splitlines()[0])
                tokenizers[model_id] = None
        tokenizer = tokenizers[model_id]
        if tokenizer is None:
            continue
        vocab_size = len(tokenizer)

        for _, row in frame.iterrows():
            formatted, prompt_ids = rebuild_prompt(row, corpus, system_prompts, tokenizer)
            response_ids = tokenizer(str(row["response"]),
                                     add_special_tokens=False)["input_ids"]
            sequence = prompt_ids + response_ids

            nxt = continuation_token(response_ids, prompt_ids, ngram_size)
            banned = banned_at_end(sequence, ngram_size, vocab_size,
                                   scope, len(prompt_ids))
            records.append({
                "run": path.name,
                "model_id": model_id,
                "condition": row["condition"],
                "rag_enabled": int(row["rag_enabled"]),
                "prompt_id": row["prompt_id"],
                "recorded_prompt_tokens": int(row["prompt_tokens"]),
                "rebuilt_prompt_tokens": len(prompt_ids),
                "completion_tokens": int(row["completion_tokens"]),
                "was_copying": nxt is not None,
                "continuation_banned": bool(nxt is not None and nxt in banned),
                "banned_token_count": len(banned),
                "response_tail": " ".join(re.findall(r"\S+", str(row["response"]))[-8:]),
            })
    return pd.DataFrame(records)


def main() -> None:
    """Report how often the n-gram constraint blocked the continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Substring of the run filename to restrict to")
    parser.add_argument("--ngram-size", type=int, default=6,
                        help="The no_repeat_ngram_size the runs used (default 6)")
    parser.add_argument("--scope", choices=["sequence", "completion"],
                        default="sequence",
                        help="Which constraint to replay. 'sequence' is what the "
                             "runs used; 'completion' replays the same responses "
                             "against the fix in src/generation_control.py")
    args = parser.parse_args()

    files = sorted(p for p in RUN_DIR.glob("phase1_*.csv")
                   if not p.name.endswith("_with_metrics.csv")
                   and (args.run is None or args.run in p.name))
    if not files:
        raise ValueError(
            f"No run files matched --run={args.run!r} in {RUN_DIR}. "
            f"Available: {sorted(p.name for p in RUN_DIR.glob('phase1_*.csv'))[:4]}"
        )

    frame = diagnose(files, args.ngram_size, args.scope)
    out_path = OUT_PATH
    if args.scope == "completion":
        out_path = OUT_PATH.with_name("truncation_diagnosis_completion_scope.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    # The rebuild has to match what actually ran, or nothing below means anything.
    drift = (frame["rebuilt_prompt_tokens"] - frame["recorded_prompt_tokens"]).abs()
    logger.info("Prompt reconstruction: median token difference %d, max %d",
                int(drift.median()), int(drift.max()))

    logger.info("")
    logger.info("=" * 74)
    logger.info("DID THE N-GRAM CONSTRAINT BLOCK THE CONTINUATION? "
                "scope=%s (n = %d)", args.scope, len(frame))
    logger.info("=" * 74)
    logger.info("  %-34s %5s %9s %9s %8s", "run", "n", "copying", "blocked", "gen_tok")
    for (run, rag), group in frame.groupby(["run", "rag_enabled"]):
        label = run.split("_2026")[0].replace("phase1_", "")
        logger.info("  %-34s %5d %8.0f%% %8.0f%% %8.0f",
                    label, len(group), group["was_copying"].mean() * 100,
                    group["continuation_banned"].mean() * 100,
                    group["completion_tokens"].mean())

    logger.info("")
    rag = frame[frame["rag_enabled"] == 1]
    base = frame[frame["rag_enabled"] == 0]
    logger.info("Pooled: continuation blocked in %.0f%% of RAG responses and %.0f%% "
                "of baseline responses", rag["continuation_banned"].mean() * 100,
                base["continuation_banned"].mean() * 100)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
