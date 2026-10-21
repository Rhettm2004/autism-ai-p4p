# Autism AI research compendium

This archive contains the Flutter screening prototype, its Python integration,
and the preserved router/RAG research code and evaluation records. Start with
[README.md](README.md) for the application and
[backend/docs/README.md](backend/docs/README.md) for the research evidence.

## Contents

| Location | Purpose |
| --- | --- |
| `lib/`, `test/`, `assets/` | Flutter source, tests, and bundled fonts |
| `android/`, `ios/`, `web/` | Platform source configuration |
| `backend/app/`, `backend/app_tests/` | FastAPI integration and tests |
| `backend/eaip_service/` | Adapter service for the external EAIP-DARV bundle |
| `backend/src/`, `backend/scripts/`, `backend/tests/` | Original research pipeline and evaluation tools |
| `backend/data/`, `backend/results_*/` | Benchmarks, annotations, and historical experimental outputs |
| `backend/docs/` | Methods, evidence, references, figures, and report tables |
| `docs/`, `output/` | Integration research record and existing report/poster artifacts |
| `pubspec.lock`, `backend/requirements*.txt` | Recorded dependency specifications |

Historical evaluation records are retained as evidence; packaging does not rerun
experiments or establish that every historical result can be reproduced exactly.
The PDF under `output/pdf/` is an existing preview artifact, not a new study result.

## External prerequisites and reproduction

Use Flutter/Dart and Python 3.11. See
[backend/DEVELOPMENT.md](backend/DEVELOPMENT.md) for dependency installation,
corpus preparation, service startup, and configuration. `pubspec.lock` records
Dart dependencies; Python requirements record the supplied specifications rather
than a complete platform-independent environment lock.

The following are deliberately excluded and must be supplied separately:

- Llama/Mistral GGUF weights and the `llama-server` executable.
- The authorized `EAIP ASD project.zip` model bundle, including its trained assets.
- Downloaded Hugging Face model caches and local Python environments.
- Extracted RAG corpus text, raw downloaded sources, and readiness/cache files.
  The upstream ignore policy excludes restricted source text from redistribution;
  the source manifest and manual reference notes remain included.
- Local `.env` configuration, logs, browser session state, IDE files, Git history,
  build output, and agent tooling.

Environment templates are provided at `backend/.env.example` and
`backend/.env.app.example`. Full inference requires the external assets above;
`./run_app.sh mock` provides the explicit UI demonstration mode.

After installing dependencies, run the existing verification commands:

```sh
(cd backend && .venv/bin/python -m pytest app_tests tests)
flutter analyze
flutter test
```

The current model feature-contract limitations, including answer encoding and age
units, are documented in [README.md](README.md) and
[backend/IMPLEMENTATION_STATUS.md](backend/IMPLEMENTATION_STATUS.md).

### Packaging validation (19 October 2026)

ZIP integrity, every manifest hash, launcher executable permissions, and identical
checksums across repeated builds were verified. A scan for common private-key and
API-token patterns found no matches; this is not a comprehensive content audit.

The existing backend suite reached 76 passing tests and one collection/setup error:
`backend/tests/test_chat.py::test` requests a missing `func` fixture. The run was
then interrupted while waiting in Hugging Face HTTP code. A complete backend pass,
Flutter checks, and live model inference were not established by this packaging
work. Application and research behavior were not changed.

## Provenance and reuse

See [backend/UPSTREAM.md](backend/UPSTREAM.md),
[backend/docs/contributions.md](backend/docs/contributions.md), and
[backend/docs/references.md](backend/docs/references.md) for attribution and source
records. Existing font licensing is retained in `assets/fonts/Roboto_LICENSE.txt`.
This packaging step grants no new license to the source, questionnaires, model
bundles, or third-party materials; their existing permissions still apply.

## Build and check the submission archive

From the Git checkout, using the Python standard library only:

```sh
python3 scripts/package_compendium.py
```

Submit `dist/autism_ai_research_compendium.zip` with its `.zip.sha256` checksum.
The packager includes current tracked files and non-ignored new files, applies
additional exclusions, preserves executable permissions, and checks ZIP integrity.
It leaves local environments and caches in place for continued development.
The superseded root `README 2.md` has been removed to avoid conflicting setup advice.

The ZIP contains `ARCHIVE_MANIFEST.json`, which records each packaged file's size
and SHA-256 (excluding the manifest itself). Archive timestamps are fixed so the
same file contents and permissions produce the same ZIP. Rebuild the archive after
any source changes. The packaging script requires Git and a source checkout;
Git history itself is never included in the submission.
