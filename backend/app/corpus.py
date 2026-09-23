"""Validate locally prepared evidence without substituting another corpus."""
import hashlib
import json
from pathlib import Path
import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_scope(root: Path):
    manifest = root / 'data/corpus/sources.yaml'
    sources = yaml.safe_load(manifest.read_text())['sources']
    enabled = {s['id'] for s in sources if s.get('enabled', True)}
    policy_path = root / 'config/corpus_policy.yaml'
    excluded = set()
    if policy_path.is_file():
        policy = yaml.safe_load(policy_path.read_text()) or {}
        if policy.get('mode') != 'explicit_exclusions':
            raise ValueError('corpus_policy_invalid')
        entries = policy.get('excluded_sources') or []
        excluded = {entry['id'] for entry in entries}
        if len(excluded) != len(entries) or not excluded.issubset(enabled):
            raise ValueError('corpus_policy_invalid')
    return enabled - excluded, excluded, policy_path


def verify_corpus(root: Path):
    corpus = root / 'data/corpus/corpus.csv'
    manifest = root / 'data/corpus/sources.yaml'
    report_path = root / 'data/corpus/readiness.json'
    if not corpus.is_file() or not report_path.is_file():
        raise ValueError('corpus_not_prepared')
    report = json.loads(report_path.read_text())
    if not report.get('complete'):
        raise ValueError('corpus_incomplete')
    expected, excluded, policy_path = corpus_scope(root)
    policy_hash = digest(policy_path) if policy_path.is_file() else None
    if (report.get('corpus_sha256') != digest(corpus)
            or report.get('manifest_sha256') != digest(manifest)
            or report.get('policy_sha256') != policy_hash
            or set(report.get('excluded_sources') or []) != excluded):
        raise ValueError('corpus_fingerprint_mismatch')
    from src.retrieval import load_corpus
    passages = load_corpus(str(corpus))
    actual = {p.meta.get('source_id') for p in passages}
    if not passages or expected != actual or any(not p.text.strip() for p in passages):
        raise ValueError('corpus_source_mismatch')
    return passages, report['corpus_sha256']
