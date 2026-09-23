# A–G implementation status

Last verified: 23 September 2026 (Pacific/Auckland).

## Implemented

- Imported Rayaan Rajabally's ZIP under `backend/` without moving or rewriting
  its research source, evaluation data, tests, documentation, or results.
- Added provenance in `UPSTREAM.md` using archive commit marker
  `036151917f928ec462b1a8918ea6ea0bd2f0e56d` and ZIP SHA-256
  `4fbee83d8cc4a5559d5faf788dc98f2ea47fbe93f9b894cccd2498451264d652`.
- Added a FastAPI service with typed `/health` and `/chat`, explicit development
  CORS, request-size/history limits, structured errors, lifecycle cleanup,
  one-generation-at-a-time capacity protection, and no application actions.
- Reused `src.chat.prepare_turn`, router A (`embeddings_word`, `topic`,
  `extended`), the original rule layer, prompt builders, TF-IDF retriever,
  source formatter, and five-passage neighbour-expanded RAG configuration.
- Preserved Rayaan's prompts unchanged. Added separately versioned application
  constraints in `config/application_prompts.yaml`; concise and inline citation
  additions are disabled.
- Added one llama.cpp adapter for backend-controlled `mistral` and `llama`
  aliases. It rejects tool calls, empty replies, and visible reasoning markers.
- Added `AutismAiBackendChatService`, typed replies/sources, explicit provider
  selection, source display/links, provider-neutral failures, persisted source
  metadata, version-1 session migration, and late-response protection in Flutter.
- Kept the screening UI, stage flow, question banks, scoring, validation, report,
  mock prediction, direct local chat provider, and mock chat provider in place.

## Verification completed

- `flutter analyze`: clean.
- `flutter test`: 45 passed.
- New backend API/integration suite: 20 passed (one dependency deprecation warning).
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
- Real backend initialization: router ready, prompts ready, selected Mistral
  server ready, corpus unavailable; `/health` returned 503 and `/chat` refused
  inference with `runtime_unavailable`, as designed.
- `git diff --check`: clean. Temporary model/backend processes were stopped.

The API tests use explicit fixture passages and fake generation. They verify
orchestration and failure behaviour but do not count as live corpus readiness or
model-quality evaluation.

## Unresolved blockers

### Corpus is incomplete

Rayaan's documented `build_corpus.py --list` and refresh build were rerun on
23 September 2026. The builder completed with exit code 0 and produced 1,605
passages from 52 enabled sources, but 9 of the 61 enabled sources were missing.
The readiness validator therefore recorded `complete: false`; live integrated
chat remains deliberately unavailable.

Missing source IDs:

- `atherapysolutions_addressing_picky_eating_in` — HTTP 403
- `autismcare_faq` — HTTP 403
- `camhs_frequently_asked_questions` — DNS resolution failure
- `cdc_index_html` — HTTP 403
- `chop_augmentative_and_alternative_communicat` — HTTP 403
- `raisingchildren_conditions_that_occur_with_a` — HTTP 403
- `raisingchildren_sleep_problems_children_with` — HTTP 403
- `raisingchildren_speech_generating_devices` — HTTP 403
- `sunhealthcares_index` — HTTP 403

Local partial-build fingerprints:

- manifest SHA-256:
  `9868aa4fd0ac1bcec54a9ee1343f2cfa514174a1ef3d036462eaabc917a615b7`
- corpus SHA-256:
  `44164ff9ae0f823c14f890223c3c6c029442628388f4903934d072f04680bf6a`

The corpus and readiness JSON remain ignored because source redistribution and
generated-data rules in Rayaan's repository require that. Exact failures are in
the ignored local `logs/corpus-build.log`.

Required next step: obtain Rayaan's permitted original corpus snapshot and hash,
or manually obtain the exact configured documents under their permitted terms
and update `sources.yaml` to point at those local copies. Then run
`python -m app.prepare_corpus --verify-existing`. No replacement documents or
benchmark-answer corpus were invented.

### Research limitations retained

- The v4 route-label audit is still absent. Ten of Rayaan's router tests skip for
  that documented reason; the integration continues to use the recorded
  `topic` track and does not fabricate audited labels.
- The app models differ from evaluated models: local Mistral v0.3 GGUF versus
  research HF Mistral v0.1, and local Llama 3 GGUF versus research HF Llama 3.1.
  Transport works, but research-equivalent generation has not been claimed.
- No full Router + real RAG + real model response was generated because doing so
  would require accepting the partial corpus. Model-quality and safety review of
  integrated live answers therefore remains outstanding after corpus resolution.
