"""
Convert an Apple Numbers (.numbers) benchmark file into the CSV format the
Phase 1 pipeline expects.

Rhett authors the benchmark in Numbers; the pipeline reads CSV. This script is
the bridge, so the conversion is reproducible rather than a manual export.

Usage:

  # Convert to the default benchmark location
  python scripts/convert_numbers_benchmark.py path/to/benchmark.numbers

  # Convert to an explicit output path
  python scripts/convert_numbers_benchmark.py input.numbers -o data/benchmark/my_benchmark.csv

  # Inspect the file's sheets/tables without writing anything
  python scripts/convert_numbers_benchmark.py input.numbers --list

Requires: pip install numbers-parser
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.benchmark_schema import REQUIRED_COLUMNS, validate_benchmark_df
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT = "data/benchmark/phase1_baseline_benchmark.csv"


def load_numbers_table(numbers_path: str, sheet: str | None = None, table: str | None = None) -> pd.DataFrame:
    """
    Read one table out of a .numbers document and return it as a DataFrame.

    The first row of the table is treated as the header. If sheet/table are not
    given, the first table of the first sheet is used.
    """
    try:
        from numbers_parser import Document
    except ImportError as exc:
        raise ImportError(
            "numbers-parser is required to read .numbers files. "
            "Install it with: pip install numbers-parser"
        ) from exc

    doc = Document(numbers_path)

    sheets = doc.sheets
    if sheet is not None:
        matching = [s for s in sheets if s.name == sheet]
        if not matching:
            available = ", ".join(s.name for s in sheets)
            raise ValueError(f"Sheet '{sheet}' not found. Available sheets: {available}")
        target_sheet = matching[0]
    else:
        target_sheet = sheets[0]

    tables = target_sheet.tables
    if table is not None:
        matching = [t for t in tables if t.name == table]
        if not matching:
            available = ", ".join(t.name for t in tables)
            raise ValueError(f"Table '{table}' not found. Available tables: {available}")
        target_table = matching[0]
    else:
        target_table = tables[0]

    logger.info(
        "Reading sheet='%s' table='%s' (%d rows x %d cols)",
        target_sheet.name,
        target_table.name,
        target_table.num_rows,
        target_table.num_cols,
    )

    rows = target_table.rows(values_only=True)
    if not rows:
        raise ValueError("The selected table is empty.")

    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)

    # Numbers leaves trailing blank rows; drop rows with no prompt_id.
    if "prompt_id" in df.columns:
        df = df[df["prompt_id"].notna() & (df["prompt_id"].astype(str).str.strip() != "")]

    # Normalise cell values to trimmed strings so downstream CSV parsing is clean.
    for col in df.columns:
        df[col] = df[col].apply(lambda v: "" if v is None else str(v).strip())

    return df.reset_index(drop=True)


def list_contents(numbers_path: str) -> None:
    """Print every sheet and table in the document, then exit."""
    from numbers_parser import Document

    doc = Document(numbers_path)
    for sheet in doc.sheets:
        print(f"Sheet: {sheet.name}")
        for tbl in sheet.tables:
            print(f"  Table: {tbl.name}  ({tbl.num_rows} rows x {tbl.num_cols} cols)")


def main() -> None:
    """Parse arguments, convert the .numbers file, and write validated CSV."""
    parser = argparse.ArgumentParser(
        description="Convert a .numbers benchmark into pipeline-ready CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the .numbers file")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--sheet", help="Sheet name (default: first sheet)")
    parser.add_argument("--table", help="Table name (default: first table)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List sheets and tables in the file, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args()

    if args.list:
        list_contents(args.input)
        return

    df = load_numbers_table(args.input, sheet=args.sheet, table=args.table)

    # Fail loudly on a malformed benchmark rather than midway through a GPU run.
    validate_benchmark_df(df, source=args.input)

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {out_path}. Re-run with --force to overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the canonical column order; carry any extra columns along at the end.
    ordered = [c for c in REQUIRED_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    logger.info("Categories: %d distinct", df["category"].nunique())
    logger.info("Sources: %d distinct", df["source_name"].nunique())


if __name__ == "__main__":
    main()
