"""
Build the RAG corpus from the source manifest.

Reads data/corpus/sources.yaml, fetches every enabled source, extracts the
readable text, chunks it, and writes data/corpus/corpus.csv for the retriever to
index.

Raw downloads are cached in data/corpus/raw/ so re-chunking at a different size
does not re-fetch. Delete that directory to force a refresh.

Usage:

  pip install requests beautifulsoup4 pypdf     # one time

  python scripts/build_corpus.py                # fetch and build
  python scripts/build_corpus.py --no-fetch     # rebuild from cache only
  python scripts/build_corpus.py --chunk-size 60 --overlap 15
  python scripts/build_corpus.py --list         # show the manifest, fetch nothing

Chunk size defaults to 120 words. At 60 words the CDC prevalence table split
across chunks and the passage carrying the current "1 in 31" figure fell to
rank 5, outside a top-3 retrieval window, even though the fact was indexed.
120 words keeps tabular facts intact. See scoring/evaluate_corpus_coverage.py.

Note on scope. The corpus is deliberately built from the full source pages rather
than from the benchmark's extracted reference answers. Indexing the answers
themselves would leak: retrieval would return the answer, generation would
paraphrase it, and the similarity metrics would rise without the system becoming
more reliable.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import chunk_text  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
MANIFEST = CORPUS_DIR / "sources.yaml"
RAW_DIR = CORPUS_DIR / "raw"
OUT_CSV = CORPUS_DIR / "corpus.csv"

# cdc.gov returns 403 to non-browser user agents, so send a standard one.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def load_boilerplate() -> list[str]:
    """Return the boilerplate line patterns declared in sources.yaml."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return manifest.get("boilerplate", []) or []


def load_manifest() -> list[dict]:
    """Read the manifest and return the enabled sources."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with open(MANIFEST, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sources = data.get("sources") or []
    enabled = [s for s in sources if s.get("enabled", True)]
    logger.info("Manifest: %d sources, %d enabled", len(sources), len(enabled))
    return enabled


def cache_path(source: dict) -> Path:
    """Return the cache file path for a source."""
    key = source.get("url") or source.get("local", "")
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return RAW_DIR / f"{source['id']}_{digest}.txt"


def html_to_text(html: str) -> str:
    """Strip a page down to readable body text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html, "html.parser")
    # Remove chrome that would otherwise dominate the chunks.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "form", "noscript", "svg", "button"]):
        tag.decompose()

    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    text = main.get_text(separator="\n")

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 2]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_boilerplate(text: str, patterns: list[str]) -> str:
    """
    Drop navigation lines that survived HTML extraction.

    Scraped pages start with chrome that is not prose: breadcrumbs, social
    links, language switches. It is a small share of the corpus but it sits in
    the first chunk of each page, next to the page title, which makes that chunk
    a magnet for any query phrased like a title. P025 asks about medication
    treatment and retrieved three title chunks instead of the passage stating
    which medications are FDA approved (F-P2-011).

    Matching is per line and case-insensitive, and a line is dropped only if it
    is boilerplate in its entirety, so prose mentioning one of these words in
    passing is kept.
    """
    if not patterns:
        return text
    compiled = [re.compile(rf"^\s*{p}\s*$", re.IGNORECASE) for p in patterns]
    kept = [ln for ln in text.splitlines()
            if not any(rx.match(ln) for rx in compiled)]
    return "\n".join(kept)


def pdf_to_text(raw: bytes) -> str:
    """Extract text from a PDF."""
    import io

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF sources: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def fetch_source(source: dict, use_cache: bool = True) -> str:
    """Fetch one source and return its extracted text, using the cache when present."""
    cache = cache_path(source)
    if use_cache and cache.exists():
        logger.info("cached  %s", source["id"])
        return cache.read_text(encoding="utf-8")

    if source.get("local"):
        path = CORPUS_DIR / source["local"]
        if not path.exists():
            raise FileNotFoundError(f"Local source missing: {path}")
        raw = path.read_bytes()
        text = pdf_to_text(raw) if path.suffix.lower() == ".pdf" else raw.decode("utf-8", "ignore")
    else:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required: pip install requests") from exc

        url = source["url"]
        logger.info("fetch   %s  %s", source["id"], url)
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
            text = pdf_to_text(resp.content)
        else:
            text = html_to_text(resp.text)

    if len(text) < 200:
        logger.warning("%s produced only %d characters, check the extraction",
                       source["id"], len(text))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    """Fetch every enabled source and write the chunked corpus."""
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build the RAG corpus from sources.yaml")
    parser.add_argument("--chunk-size", type=int, default=120, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="Word overlap between chunks")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached downloads only")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch")
    parser.add_argument("--list", action="store_true", help="List the manifest and exit")
    parser.add_argument("--keep-boilerplate", action="store_true",
                        help="Do not strip navigation lines. Reproduces the corpus "
                             "the Phase 2 runs used.")
    parser.add_argument("--out", default=str(OUT_CSV),
                        help="Where to write the corpus, for building a variant "
                             "without replacing the one in use")
    args = parser.parse_args()

    sources = load_manifest()
    boilerplate = load_boilerplate()

    if args.list:
        for s in sources:
            target = s.get("url") or f"local:{s.get('local')}"
            print(f"  [{s['authority']}] {s['id']:<26} {s['name']}")
            print(f"      {target}")
        return

    rows = []
    failed = []
    for s in sources:
        try:
            # local: sources are read from disk, so --no-fetch must not skip them.
            if args.no_fetch and not s.get("local") and not cache_path(s).exists():
                logger.warning("skip    %s (no cache, --no-fetch set)", s["id"])
                continue
            text = fetch_source(s, use_cache=not args.refresh)
        except Exception as exc:
            logger.error("FAILED  %s: %s", s["id"], exc)
            failed.append((s["id"], str(exc)))
            continue

        if not args.keep_boilerplate:
            text = strip_boilerplate(text, boilerplate)
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        for j, ch in enumerate(chunks):
            rows.append({
                "passage_id": f"{s['id']}_c{j}",
                "text": ch,
                "source_id": s["id"],
                "source_name": s["name"],
                "source_url": s.get("url", s.get("local", "")),
                "authority": s.get("authority", 3),
            })
        logger.info("chunked %s into %d passages", s["id"], len(chunks))

    if not rows:
        raise RuntimeError("No passages produced. Check the manifest and network access.")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 70)
    print("CORPUS BUILT")
    print("=" * 70)
    print(f"  Sources fetched : {df['source_id'].nunique()} of {len(sources)}")
    print(f"  Passages        : {len(df)}")
    print(f"  Chunk size      : {args.chunk_size} words, overlap {args.overlap}")
    print(f"  Mean words      : {df['text'].str.split().str.len().mean():.0f}")
    print(f"  Written to      : {out_csv}")
    print("\n  Passages per source")
    for sid, n in df["source_id"].value_counts().items():
        print(f"    {n:>5}  {sid}")
    if failed:
        print("\n  FAILED SOURCES")
        for sid, err in failed:
            print(f"    {sid}: {err[:90]}")
        print("\n  A failed fetch is usually a blocked user agent or a moved page.")
        print("  Download the page manually into data/corpus/manual/ and switch the")
        print("  entry in sources.yaml from url: to local:.")


if __name__ == "__main__":
    main()
