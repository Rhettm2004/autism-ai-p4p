# Autism AI implementation status

Last verified: 2 October 2026 (Pacific/Auckland).

## Implemented

- Imported Rayaan Rajabally's ZIP under `backend/` without moving or rewriting
  its research source, evaluation data, tests, documentation, or results.
- Added provenance in `UPSTREAM.md` using archive commit marker
  `036151917f928ec462b1a8918ea6ea0bd2f0e56d` and ZIP SHA-256
  `4fbee83d8cc4a5559d5faf788dc98f2ea47fbe93f9b894cccd2498451264d652`.
- Added a FastAPI service with typed `/health` and `/chat`, explicit development
  CORS, request-size/history limits, structured errors, lifecycle cleanup,
  one-generation-at-a-time capacity protection, and one typed
  `start_screening` action proposal validated by Flutter.
- Reused `src.chat.prepare_turn`, router A (`embeddings_word`, `topic`,
  `extended`), the original rule layer, prompt builders, TF-IDF retriever,
  source formatter, and five-passage RAG configuration. The active application
  profile uses `chat.py`'s default of neighbour expansion off.
- The active `app_context_v1` profile reuses Rayaan's `prepare_turn()` assembly,
  routing, grounding, and route guidance with a separately versioned neutral
  application base prompt and constraints. It adds managed conversation history,
  current screening state, and the active questionnaire as read-only context.
  Rayaan's exact single-turn CLI profile remains unchanged.
- Added one llama.cpp adapter for backend-controlled `mistral` and `llama`
  aliases. It rejects tool calls, empty replies, and visible reasoning markers.
- Added `AutismAiBackendChatService`, typed replies/sources, explicit provider
  selection, source display/links, provider-neutral failures, persisted source
  metadata, version-1 session migration, and late-response protection in Flutter.
- Refactored the default Flutter interface into one chat-first workspace with
  natural-language screening start, inline stage cards, collapsed completed
  steps, review editing, inline disclaimer/result/report, and an explicit flag
  for the previous split workspace.
- Reused Rayaan's `src.chat.parse_command` and `config/demo.yaml` for Flutter
  `/help`, `/examples`, `/ex N`, `/prompt`, `/cite`, `/concise`, `/router`,
  `/rag`, `/start`, `/quit`, and `/exit` support. Command settings are explicit request
  data. Router/RAG/concise and inline citations start on in the integrated app.
- Applied Rayaan's original citation renumbering and source-ordering functions
  to API answers. Uncited answers are returned without a rewrite, only cited
  sources are exposed to Flutter, and invalid source numbers are reported in
  response metadata.
- Integrated the supplied three-module EAIP-DARV bundle behind typed
  `/screening/health` and `/screening/predict` endpoints. It runs in an isolated
  environment and never falls back to a mock.
- Added deterministic mapping from each selected answer to binary `Q1`–`Q10`,
  submission of all required EAIP fields, explicit loading/error states, result
  persistence, and late-response protection.
- Replaced the prototype result and report with EAIP-DARV output. The UI and PDF
  show the screening flag, DARV probability, disagreement, confidence, per-module
  values, original answers, and exact submitted model fields.
- Kept explicit mock prediction and chat providers for tests and UI-only work,
  plus the direct local chat provider.

## Verification completed

- `flutter analyze`: clean.
- `flutter test`: 55 passed.
- Backend API/integration suite: 43 passed (one dependency deprecation warning).
- Original research suites run successfully:
  - interactive chat: 21 passed;
  - retrieval expansion: 9 passed;
  - route prompts: 15 passed;
  - generation control: 14 passed;
  - router evaluation: 45 passed, 10 upstream-declared skips because the v4
    label audit is absent;
  - LLM router: 32 passed;
  - safety grading: 17 passed;
  - scoring arms: 5 passed;
  - follow-up conformance: 14 passed;
  - generation comparison: 14 passed.
- Live Mistral v0.3 GGUF request through the implemented llama.cpp request shape:
  HTTP 200, alias `mistral`, visible response returned.
- Live Llama 3 GGUF request through the same shape: HTTP 200, alias `llama`,
  visible response returned.
- Live reduced-corpus backend initialization: router, prompts, corpus, and
  selected Mistral server ready. `/health` returned HTTP 200 and disclosed
  `reduced_corpus_10_sources_excluded`.
- Live end-to-end `/chat`: HTTP 200 through router A, five-passage TF-IDF RAG,
  the FastAPI adapter, and Mistral. The response used the
  `general_knowledge` route, included supporting-source metadata, and returned
  `action: null`.
- Live Flutter-shaped history check: HTTP 200 after normalizing the initial
  assistant greeting and consecutive user turns for Mistral's strict
  alternating-role chat template.
- Live `/help` command check: HTTP 200 with all original demo commands and the
  current command settings in the typed response.
- The earlier exact-prompt comparison for `what is asd` remains recorded as a
  research-fidelity baseline. The active chat-first profile now intentionally
  adds application context and therefore is not byte-identical to that baseline.
- `git diff --check`: clean. The temporary EAIP-DARV verification service was
  stopped after the live prediction check.

The API tests use explicit fixture passages and fake generation. They verify
orchestration, conversation context, questionnaire context, action validation,
and failure behaviour but do not count as live corpus readiness or model-quality
evaluation. Live generation with the new `app_context_v1` profile remains to be
checked when a local model server is running.

## EAIP-DARV integration notes

- Supplied ZIP SHA-256:
  `268c329a840ec70d283fce12b8a68785536bc8a878a3ff475b1d56ae39dbaadc`.
- Supplied model hashes: Module 1
  `01ea3b2275fab0874cf29176844d0cb871a86bae2a7d9852687d304943f5fa16`,
  Module 2
  `03ef054980458469d4c5eb0e3c7414287a8558db79b4387441862420f50227ba`,
  and Module 3
  `e995db8511ae9e7a62c7a6c65ce8b8cfe1a2c40b6da193d4cfdd894ee178bf3d`.
- `setup_eaip.sh` extracts the supplied ZIP to ignored local cache and installs
  its runtime separately from the chat backend.
- The supplied contracts contain absolute Windows paths. The service rewrites
  those paths in temporary runtime contracts without modifying the source ZIP.
- The supplied Keras 2.13 files contain Windows-authored nested weight-group names
  that macOS loaders cannot associate with layers automatically. The compatibility
  loader rebuilds the exact architectures declared in the supplied training scripts
  and maps the supplied tensors by layer. It does not retrain, substitute, or alter
  model weights.
- A live structured request loaded all three supplied modules and returned HTTP
  200 with raw and calibrated probabilities, agreement, EAIP/DARV probability,
  disagreement, confidence, thresholds, and classifications.
- The supplied artifacts do not include an explicit document mapping the four
  questionnaire answer scales to binary `Q1`–`Q10`. The integration applies the
  established per-item scoring keys already implemented for those question banks.
  This is deterministic and tested, but the model owner should confirm that it
  matches the encoding used to train the bundle before research results are
  interpreted or published.

## Research-fidelity limitations

### Corpus scope reduced by explicit project decision

Rayaan's documented `build_corpus.py --list` and refresh build were rerun on
23 September 2026. The project owner then supplied the corpus snapshot used for
the `chat.py` comparison. An official KidsHealth New Zealand jaundice page was
subsequently added through Rayaan's builder to support the application's existing
background question. The resulting corpus contains 1,689 passages from 52 of the
62 enabled manifest sources. `config/corpus_policy.yaml` records the addition,
ten exclusions, resulting SHA-256, and reasons. The readiness validator accepts
this 52-source corpus and health reports the reduced scope. The base snapshot
reproduces the five retrieved passages in the supplied
"what is asd" prompt when used with neighbour expansion off, but it does not
contain every source in the current manifest.

Missing source IDs:

- `atherapysolutions_addressing_picky_eating_in` — HTTP 403
- `autismcare_faq` — HTTP 403
- `camhs_frequently_asked_questions` — DNS resolution failure
- `cdc_index_html` — HTTP 403
- `chop_augmentative_and_alternative_communicat` — HTTP 403
- `raisingchildren_conditions_that_occur_with_a` — HTTP 403
- `raisingchildren_pecs` — absent from the supplied snapshot
- `raisingchildren_sleep_problems_children_with` — HTTP 403
- `raisingchildren_speech_generating_devices` — HTTP 403
- `sunhealthcares_index` — HTTP 403

Accepted reduced-corpus fingerprints:

- manifest SHA-256:
  `e8099d006a2debc35c62e04e87f67dd25927788d7778ce631771b74b4c8b9df0`
- corpus SHA-256:
  `540364bca8c95f78e206a275ae14d90afd96699573e4ac81b342a70d20608ef7`

The corpus and readiness JSON remain ignored because source redistribution and
generated-data rules in Rayaan's repository require that. Exact failures are in
the ignored local `logs/corpus-build.log`.

To restore full research scope, obtain Rayaan's permitted original corpus
snapshot and hash, or manually obtain the exact configured documents under
their permitted terms and update `sources.yaml` to point at those local copies.
Then remove the explicit exclusions and run
`python -m app.prepare_corpus --verify-existing`. No replacement documents or
benchmark-answer corpus were invented.

### Research limitations retained

- The v4 route-label audit is still absent. Ten of Rayaan's router tests skip for
  that documented reason; the integration continues to use the recorded
  `topic` track and does not fabricate audited labels.
- The app models differ from evaluated models: local Mistral v0.3 GGUF versus
  research HF Mistral v0.1, and local Llama 3 GGUF versus research HF Llama 3.1.
  Transport works, but research-equivalent generation has not been claimed.
- The live Router + RAG + Mistral verification applies to the accepted
  52-source application corpus, not the full manifest used to define the
  intended research scope.
