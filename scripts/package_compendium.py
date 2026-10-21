#!/usr/bin/env python3
"""Package the current checkout without local runtime state (standard library only)."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "autism_ai_research_compendium"
EXCLUDED_PARTS = {
    ".git", ".idea", ".claude", ".codex", ".agents", ".aws",
    ".venv", ".venv-eaip", "venv", "__pycache__", ".pytest_cache",
    ".dart_tool", ".cache", ".tools", "node_modules", "dist",
}
EXCLUDED_PREFIXES = (
    "build/", "tmp/", ".github/modernize/", "backend/logs/",
    "backend/data/corpus/raw/", "backend/data/results/",
    "backend/data/vectorstore/",
)


def excluded(name):
    path = PurePosixPath(name)
    return (
        bool(set(path.parts) & EXCLUDED_PARTS)
        or name.startswith(EXCLUDED_PREFIXES)
        or path.name in {".DS_Store", ".flutter-plugins-dependencies"}
        or (path.name.startswith(".env") and not path.name.endswith(".example"))
        or path.suffix.lower() in {
            ".pyc", ".log", ".gguf", ".pth", ".pt", ".h5", ".keras",
            ".safetensors", ".jks", ".keystore", ".pem", ".key",
        }
        or (name.startswith("backend/data/corpus/") and
            (path.name.startswith("corpus") or path.name == "readiness.json"))
        or (name.startswith("backend/data/grading") and path.name == "blind_map.csv")
    )


def main():
    # Include current edits and non-ignored new files, not just the last commit.
    # Git's nested ignore rules also exclude generated Android/iOS files.
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    ).decode().split("\0")
    files = []
    for name in sorted(set(filter(None, paths))):
        if excluded(name):
            continue
        path = ROOT / name
        if path.is_symlink():
            raise SystemExit(f"Refusing to package symlink: {name}")
        if path.is_file():
            files.append((name, path.read_bytes(), path.stat().st_mode & 0o777))

    manifest = {
        "format": 1,
        "description": "Current source checkout; local state and external models excluded.",
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data, _ in files
        },
    }
    files.append(("ARCHIVE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o644))
    destination = ROOT / "dist" / f"{PREFIX}.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in files:
            info = zipfile.ZipInfo(f"{PREFIX}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise SystemExit("Archive integrity check failed")
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{checksum}  {destination.name}\n")
    print(f"Created {destination.relative_to(ROOT)} ({len(files)} files, {destination.stat().st_size:,} bytes)")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
