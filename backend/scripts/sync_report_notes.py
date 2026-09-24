"""
Export the project notes to the report writing folder.

The notes, the exported numbers and the figures are version controlled in this
repository alongside the code that produced them. Writing happens elsewhere, in
`final report/notes/`, so this script copies a current snapshot there.

The snapshot is disposable. Nothing should ever be edited in the destination:
the next sync overwrites it. Edit the originals under `docs/` and sync again.

Usage examples:

  python scripts/sync_report_notes.py
  python scripts/sync_report_notes.py --dest "C:/some/other/folder"
  python scripts/sync_report_notes.py --dry-run
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
DEFAULT_DEST = BASE_DIR.parent / "final report" / "notes"

# Copied as they are, preserving the directory layout. The figure patterns are
# recursive because figures live in one folder per stage of the project, and the
# folder is what tells two versions of the same chart apart.
PATTERNS = ["*.md", "evidence/*.md", "report/*.md", "report/numbers/*.csv",
            "figures/*.md", "figures/**/*.md", "figures/**/*.png"]


def _git_commit() -> str:
    """Return the short hash of HEAD, or 'unknown' outside a git checkout."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _write_index(dest: Path, copied: list[Path]) -> None:
    """Write a short orientation file at the top of the snapshot."""
    lines = [
        "# Project notes (generated snapshot)",
        "",
        f"Exported from the `autism-ai` repository at commit `{_git_commit()}` on "
        f"{datetime.now().strftime('%d %B %Y, %H:%M')}.",
        "",
        "**Do not edit anything in this folder.** It is overwritten by",
        "`python scripts/sync_report_notes.py`. Edit the originals under `docs/` in",
        "the repository, then sync again.",
        "",
        "## Start here",
        "",
        "- `DATA_GUIDE.md` - what every dataset is, what each column means, and which",
        "  report claim it supports. Read this first.",
        "- `report/NUMBERS.md` - every table the report needs, with the CSVs beside it",
        "  in `report/numbers/` for building figures.",
        "- `report/requirements.md` - length, format, structure and deadlines.",
        "- `report/outline.md` - the argument, section by section.",
        "- `evidence/` - the append-only findings log for each phase.",
        "- `decisions.md` - design decisions with their justification.",
        "- `contributions.md` - who did what, for the Statement of Contribution.",
        "- `figures/` - built figures, one folder per stage of the project, with",
        "  `figures/MANIFEST.md` listing every one. Folder 05 supersedes folder 03:",
        "  they are the same charts of two different grading passes.",
        "",
        "## Files in this snapshot",
        "",
    ]
    for path in sorted(copied):
        lines.append(f"- `{path.as_posix()}`")
    (dest / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync(dest: Path = DEFAULT_DEST, dry_run: bool = False) -> list[Path]:
    """Copy the notes, numbers and figures into dest, returning the relative paths."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"No docs directory at {DOCS_DIR}")

    sources: list[Path] = []
    for pattern in PATTERNS:
        sources.extend(sorted(DOCS_DIR.glob(pattern)))

    # The snapshot is a mirror, not an accumulation: anything no longer in docs
    # is removed, or a figure that moved folders would appear in both places.
    if not dry_run and dest.exists():
        keep = {(dest / s.relative_to(DOCS_DIR)).resolve() for s in sources}
        keep.add((dest / "INDEX.md").resolve())
        for existing in sorted(dest.rglob("*"), reverse=True):
            if existing.is_file() and existing.resolve() not in keep:
                existing.unlink()
            elif existing.is_dir() and not any(existing.iterdir()):
                existing.rmdir()

    copied: list[Path] = []
    for source in sources:
        relative = source.relative_to(DOCS_DIR)
        copied.append(relative)
        if dry_run:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    if not dry_run:
        _write_index(dest, copied)
    return copied


def main() -> None:
    """Parse arguments and export the notes snapshot."""
    parser = argparse.ArgumentParser(description="Export project notes for report writing.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"destination folder (default: {DEFAULT_DEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be copied without writing anything")
    args = parser.parse_args()

    copied = sync(args.dest, dry_run=args.dry_run)
    for relative in copied:
        logger.info("  %s", relative.as_posix())
    if args.dry_run:
        logger.info("Dry run: %d file(s) would be copied to %s", len(copied), args.dest)
    else:
        logger.info("Copied %d file(s) to %s", len(copied), args.dest)


if __name__ == "__main__":
    main()
