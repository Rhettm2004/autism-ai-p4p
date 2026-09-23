"""
Benchmark schema definition and validation.

The Phase 1 benchmark is authored externally (currently in Apple Numbers) and
converted to CSV. Because a malformed benchmark would otherwise only surface
partway through a multi-hour GPU run, validation happens up front, both at
conversion time and immediately before a benchmark run starts.
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Canonical column order. Every benchmark CSV must carry these columns.
REQUIRED_COLUMNS = [
    "prompt_id",
    "category",
    "prompt",
    "reference_answer",
    "source_name",
    "source_url",
    "answer_origin",
]

# Columns that must be non-empty on every row for the run to be meaningful.
NON_EMPTY_COLUMNS = ["prompt_id", "category", "prompt", "reference_answer"]

# The reference answer and its provenance. Present on every benchmark meant for
# similarity scoring, and absent from prompt sets where no single correct answer
# can be written down. The adversarial safety set is the case in point: the right
# response to "just tell me if my child has autism" is a refusal, and there is no
# defensible single reference text for one. Those sets are still worth running,
# and are graded on the rubric alone, so validation can be asked to allow their
# absence explicitly rather than the schema being loosened for everyone.
REFERENCE_COLUMNS = ["reference_answer", "source_name", "source_url", "answer_origin"]


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark file does not satisfy the expected schema."""


def validate_benchmark_df(df: pd.DataFrame, source: str = "benchmark",
                          allow_missing_reference: bool = False) -> None:
    """
    Validate a benchmark DataFrame in place, raising on any structural problem.

    Checks performed:
      - all required columns present
      - at least one data row
      - prompt_id values unique
      - no empty values in the columns that must be populated

    Raises BenchmarkValidationError with a message naming the offending rows.
    """
    required = REQUIRED_COLUMNS
    non_empty = NON_EMPTY_COLUMNS
    if allow_missing_reference:
        required = [c for c in REQUIRED_COLUMNS if c not in REFERENCE_COLUMNS]
        non_empty = [c for c in NON_EMPTY_COLUMNS if c not in REFERENCE_COLUMNS]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BenchmarkValidationError(
            f"{source}: missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(required)}. "
            f"Found: {', '.join(df.columns)}"
        )

    if len(df) == 0:
        raise BenchmarkValidationError(f"{source}: benchmark contains no data rows.")

    duplicates = df["prompt_id"][df["prompt_id"].duplicated()].unique().tolist()
    if duplicates:
        raise BenchmarkValidationError(
            f"{source}: duplicate prompt_id value(s): {', '.join(map(str, duplicates))}. "
            "Every prompt_id must be unique so results can be joined back to references."
        )

    for col in non_empty:
        blank_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank_mask.any():
            bad_ids = df.loc[blank_mask, "prompt_id"].astype(str).tolist()
            raise BenchmarkValidationError(
                f"{source}: column '{col}' is empty for prompt_id(s): "
                f"{', '.join(bad_ids[:10])}"
                + (" ..." if len(bad_ids) > 10 else "")
            )

    sources = df["source_name"].nunique() if "source_name" in df.columns else 0
    logger.info(
        "%s: schema OK — %d prompts, %d categories, %d sources%s",
        source, len(df), df["category"].nunique(), sources,
        ", no reference answers" if allow_missing_reference else "",
    )


def validate_benchmark_csv(csv_path: str,
                           allow_missing_reference: bool = False) -> pd.DataFrame:
    """
    Load a benchmark CSV, validate it, and return the DataFrame.

    Use this anywhere the benchmark is read for a real run. When reference
    columns are allowed to be absent they are filled with empty strings, so every
    caller downstream sees the same shape whichever kind of prompt set it is.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    validate_benchmark_df(df, source=csv_path,
                          allow_missing_reference=allow_missing_reference)
    if allow_missing_reference:
        for column in REFERENCE_COLUMNS:
            if column not in df.columns:
                df[column] = ""
    return df


def summarise_benchmark(csv_path: str) -> dict:
    """
    Return a summary dict describing a benchmark CSV.

    Useful for reporting benchmark composition without re-deriving it by hand.
    """
    df = validate_benchmark_csv(csv_path)
    return {
        "path": csv_path,
        "n_prompts": len(df),
        "n_categories": int(df["category"].nunique()),
        "n_sources": int(df["source_name"].nunique()),
        "categories": df["category"].value_counts().to_dict(),
        "sources": df["source_name"].value_counts().to_dict(),
        "answer_origins": df["answer_origin"].value_counts().to_dict(),
    }
