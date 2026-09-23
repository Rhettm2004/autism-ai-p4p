import csv
import json
import pytest
from app.prepare_corpus import inspect
from app.corpus import verify_corpus


def test_partial_or_tampered_corpus_never_ready(tmp_path):
    data = tmp_path / 'data/corpus'
    data.mkdir(parents=True)
    (data / 'sources.yaml').write_text('sources:\n  - {id: one, enabled: true}\n  - {id: two, enabled: true}\n')
    path = data / 'corpus.csv'
    def write(ids):
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['passage_id', 'text', 'source_name', 'source_id'])
            writer.writeheader()
            for i in ids: writer.writerow(dict(passage_id=i+'_c0', text='fixture only', source_name=i, source_id=i))
    write(['one'])
    report = inspect(tmp_path)
    assert report['missing_sources'] == ['two'] and not report['complete']
    (data / 'readiness.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='incomplete'): verify_corpus(tmp_path)
    write(['one', 'two'])
    report = inspect(tmp_path)
    (data / 'readiness.json').write_text(json.dumps(report))
    assert len(verify_corpus(tmp_path)[0]) == 2
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='fingerprint'): verify_corpus(tmp_path)
