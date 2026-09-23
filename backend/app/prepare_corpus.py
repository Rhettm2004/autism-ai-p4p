"""Run Rayaan's builder, then attest completeness. Never invent missing evidence.

Run: python -m app.prepare_corpus [--verify-existing]
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from app.settings import ROOT
from app.corpus import corpus_scope, digest


def inspect(root):
    manifest = root / 'data/corpus/sources.yaml'
    expected, excluded, policy_path = corpus_scope(root)
    corpus = root / 'data/corpus/corpus.csv'
    rows = []
    if corpus.exists():
        with corpus.open(newline='') as f:
            rows = list(csv.DictReader(f))
    actual = {r.get('source_id') for r in rows}
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    empty = sum(not r.get('text', '').strip() for r in rows)
    return {'complete': bool(rows) and not missing and not unexpected and not empty,
            'missing_sources': missing, 'unexpected_sources': unexpected,
            'empty_passages': empty, 'passages': len(rows), 'source_count': len(actual),
            'manifest_sha256': digest(manifest),
            'policy_sha256': digest(policy_path) if policy_path.is_file() else None,
            'excluded_sources': sorted(excluded),
            'corpus_sha256': digest(corpus) if corpus.exists() else None,
            'prepared_at': datetime.now(timezone.utc).isoformat(),
            'provenance': 'local_build_or_supplied_snapshot_not_verified_as_research_snapshot'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    report_path = ROOT / 'data/corpus/readiness.json'
    report_path.unlink(missing_ok=True)
    exit_code = 0
    if not args.verify_existing:
        logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
        with (logs / 'corpus-build.log').open('w') as log:
            result = subprocess.run([sys.executable, 'scripts/build_corpus.py',
                '--chunk-size', '120', '--overlap', '30'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        exit_code = result.returncode
    report = inspect(ROOT)
    report['builder_exit_code'] = exit_code
    report['complete'] = report['complete'] and exit_code == 0
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
