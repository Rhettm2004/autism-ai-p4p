"""
Distribution-shift evaluation of the router's rule layer.

Cross-validation cannot test what the rules are for. Every fold trains the
classifier on adversarial prompts drawn from the same small pool it is then
tested on, so by the time the rules are consulted the classifier has already
learned the phrasings they exist to catch. Measured that way the rules look
almost worthless: 0.8pp of safety recall, not significant (F-P3-004).

The claim the rules actually make is about a turn resembling nothing in the
training data. This script tests that directly by removing whole families of
safety prompts from training and testing on the family that was removed.

The split is deterministic, so there is nothing to average and no confidence
interval to report. Held-out groups have three, six and nine prompts, and results
are reported as counts.

Usage:

  python scripts/evaluate_rule_shift.py
  python scripts/evaluate_rule_shift.py --scenario jailbreaks
  python scripts/evaluate_rule_shift.py --track topic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.router_eval import (  # noqa: E402
    EVAL_DIR,
    HOLDOUT_SCENARIOS,
    evaluate_holdout,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

OUT_PATH = EVAL_DIR / "rule_distribution_shift.csv"

# Each pair is (with the layer, without it), so the difference isolates what the
# component contributes under shift.
PAIRS = [
    ("baseline", "classifier_only"),
    ("cascade", "cascade_no_rules"),
]


def main() -> None:
    """Run every scenario for every configuration and report what the rules save."""
    parser = argparse.ArgumentParser(description="Distribution-shift test for the rule layer.")
    parser.add_argument("--scenario", help="run one scenario by name")
    parser.add_argument("--track", "--taxonomy", dest="track", default="audited",
                        help="ground truth to measure against (default: audited)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else list(HOLDOUT_SCENARIOS)
    for name in scenarios:
        if name not in HOLDOUT_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{name}'. Available: {sorted(HOLDOUT_SCENARIOS)}"
            )

    rows = []
    for scenario in scenarios:
        description, _ = HOLDOUT_SCENARIOS[scenario]
        logger.info("=" * 76)
        logger.info("SCENARIO: %s", scenario)
        logger.info("%s", description)
        logger.info("=" * 76)

        for with_layer, without_layer in PAIRS:
            for config in (with_layer, without_layer):
                result = evaluate_holdout(config, scenario, taxonomy=args.track)
                row = result.summary()
                rows.append(row)
                logger.info(
                    "  %-18s train %2d, test %2d | safety reached %d of %d | "
                    "rules decided %d | risk %.2f",
                    config, row["n_train"], row["n_test"], row["safety_reached"],
                    row["safety_in_test"], row["decided_by_rule"], row["risk_on_test"],
                )
                for missed in result.missed:
                    logger.info("      MISSED: %s", missed[:72])
        logger.info("")

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.scenario and OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        keep = existing[~((existing["scenario"] == args.scenario)
                          & (existing["track"] == args.track))]
        frame = pd.concat([keep, frame], ignore_index=True)
    frame.to_csv(OUT_PATH, index=False)
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
