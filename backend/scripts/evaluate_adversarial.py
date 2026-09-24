"""
Evaluate the router against a held-out adversarial set.

The set in `data/benchmark/router_adversarial_v1.csv` is never trained on. The
router is fitted on the full extended prompt set and then asked to route twenty
turns it has never seen, every one of which is a request for a diagnostic
judgement phrased to avoid the vocabulary the rule patterns match on.

What this measures and what it does not. The prompts were written by the same
author as the rule patterns, with the pattern list in view and the known gaps
deliberately targeted. That makes the result a worst case rather than an
estimate: it is what a determined person who has read the source can do, not what
an ordinary caregiver would say. A genuinely blind set, written by someone who
has never seen the patterns, is a different and still-missing test.

Usage:

  python scripts/evaluate_adversarial.py
  python scripts/evaluate_adversarial.py --track topic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.router import ROUTE_SAFETY, is_diagnosis_request  # noqa: E402
from src.router_eval import (  # noqa: E402
    EVAL_DIR,
    get_config,
    load_labelled_prompts,
    mean_risk,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

ADVERSARIAL_PATH = Path(__file__).parent.parent / "data" / "benchmark" / "router_adversarial_v1.csv"
OUT_PATH = EVAL_DIR / "adversarial_holdout.csv"

CONFIGS = ["baseline", "embeddings_word", "classifier_only"]


def load_adversarial() -> pd.DataFrame:
    """Load the held-out adversarial set, checking it is entirely safety turns."""
    if not ADVERSARIAL_PATH.exists():
        raise FileNotFoundError(f"Adversarial set not found: {ADVERSARIAL_PATH}")
    frame = pd.read_csv(ADVERSARIAL_PATH, dtype=str, keep_default_na=False)
    if set(frame["route"]) != {ROUTE_SAFETY}:
        raise ValueError(
            f"Every adversarial prompt must be a safety turn, found: {set(frame['route'])}"
        )
    return frame


def main() -> None:
    """Fit each configuration on the extended set and route the adversarial set."""
    parser = argparse.ArgumentParser(description="Route a held-out adversarial set.")
    parser.add_argument("--track", "--taxonomy", dest="track", default="topic",
                        help="ground truth to train against (default: audited)")
    args = parser.parse_args()

    adversarial = load_adversarial()
    training = load_labelled_prompts(taxonomy=args.track, dataset="extended")

    overlap = set(adversarial["prompt"]) & set(training["prompt"])
    if overlap:
        raise ValueError(f"Adversarial prompts leaked into training: {sorted(overlap)[:3]}")

    rule_hits = adversarial["prompt"].apply(is_diagnosis_request)
    logger.info("=" * 74)
    logger.info("HELD-OUT ADVERSARIAL SET: %d prompts, none seen in training",
                len(adversarial))
    logger.info("=" * 74)
    logger.info("  Rule layer alone: %d of %d caught", int(rule_hits.sum()), len(adversarial))

    rows, per_prompt = [], adversarial.copy()
    for config_name in CONFIGS:
        config = get_config(config_name)
        router = config.build().fit(training["prompt"].tolist(), training["route"].tolist())
        predictions, stages = [], []
        for text in adversarial["prompt"]:
            route, stage = router.route(text)
            predictions.append(route)
            stages.append(stage)
        per_prompt[f"{config_name}_route"] = predictions
        reached = sum(1 for r in predictions if r == ROUTE_SAFETY)
        rows.append({
            "config": config_name,
            "track": args.track,
            "n": len(adversarial),
            "safety_reached": reached,
            "safety_recall": reached / len(adversarial),
            "decided_by_rule": sum(1 for s in stages if s == "rule"),
            "risk": mean_risk(adversarial["route"].tolist(), predictions),
        })
        logger.info("  %-18s %2d of %d reached safety (rules decided %d), risk %.2f",
                    config_name, reached, len(adversarial),
                    rows[-1]["decided_by_rule"], rows[-1]["risk"])

    logger.info("")
    logger.info("  Per prompt, adopted router (embeddings_word):")
    for _, row in per_prompt.iterrows():
        mark = "OK  " if row["embeddings_word_route"] == ROUTE_SAFETY else "MISS"
        logger.info("    %s [%-24s] %s", mark, row["attack_family"], row["prompt"][:60])

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    per_prompt.to_csv(EVAL_DIR / "adversarial_holdout_per_prompt.csv", index=False)
    logger.info("")
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
