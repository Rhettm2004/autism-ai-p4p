"""Validate locally prepared evidence without substituting another corpus."""
import hashlib
import json
from pathlib import Path
import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_corpus(root: Path):
    corpus = root / 'data/corpus/corpus.csv'
    manifest = root / 'data/corpus/sources.yaml'
    report_path = root / 'data/corpus/readiness.json'
    if not corpus.is_file() or not report_path.is_file():
        raise ValueError('corpus_not_prepared')
    report = json.loads(report_path.read_text())
    if not report.get('complete'):
        raise ValueError('corpus_incomplete')
    if report.get('corpus_sha256') != digest(corpus) or report.get('manifest_sha256') != digest(manifest):
        raise ValueError('corpus_fingerprint_mismatch')
    from src.retrieval import load_corpus
    passages = load_corpus(str(corpus))
    enabled = {s['id'] for s in yaml.safe_load(manifest.read_text())['sources'] if s.get('enabled', True)}
    actual = {p.meta.get('source_id') for p in passages}
    if not passages or enabled != actual or any(not p.text.strip() for p in passages):
        raise ValueError('corpus_source_mismatch')
    return passages, report['corpus_sha256']
