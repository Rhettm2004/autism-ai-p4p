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
  source formatter, and five-passage RAG configuration. The active exact-chat
  profile uses `chat.py`'s default of neighbour expansion off.
- The active `rayaan_chat_exact` profile sends Rayaan's `prepare_turn()` system
  prompt and exact question byte for byte, with independent turns as in
  `scripts/chat.py`. The separately versioned application constraints remain
  available as an inactive profile.
- Added one llama.cpp adapter for backend-controlled `mistral` and `llama`
  aliases. It rejects tool calls, empty replies, and visible reasoning markers.
- Added `AutismAiBackendChatService`, typed replies/sources, explicit provider
  selection, source display/links, provider-neutral failures, persisted source
  metadata, version-1 session migration, and late-response protection in Flutter.
- Reused Rayaan's `src.chat.parse_command` and `config/demo.yaml` for Flutter
  `/help`, `/examples`, `/ex N`, `/prompt`, `/cite`, `/concise`, `/router`,
  `/rag`, `/quit`, and `/exit` support. Command settings are explicit request
  data. Router/RAG/concise and inline citations start on in the integrated app.
- Applied Rayaan's original citation renumbering and source-ordering functions
  to API answers. As in Rayaan's terminal demo, uncited answers are returned
  without a rewrite, sources are marked cited or uncited, and invalid source
  numbers are reported in response metadata.
- Kept the screening UI, stage flow, question banks, scoring, validation, report,
  mock prediction, direct local chat provider, and mock chat provider in place.

## Verification completed

- `flutter analyze`: clean.
- `flutter test`: 46 passed.
- New backend API/integration suite: 29 passed (one dependency deprecation warning).
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
- Live exact-prompt check for `what is asd`: HTTP 200; the system prompt had no
  application constraints or screening context, citation and concise blocks
  matched the selected commands, the question was byte-exact, and the five
  retrieved passage IDs matched the supplied `chat.py` transcript.
- `git diff --check`: clean. The temporary backend process was stopped; the
  pre-existing local Mistral server was left running.

The API tests use explicit fixture passages and fake generation. They verify
orchestration and failure behaviour but do not count as live corpus readiness or
model-quality evaluation.

## Research-fidelity limitations

### Corpus scope reduced by explicit project decision

Rayaan's documented `build_corpus.py --list` and refresh build were rerun on
23 September 2026. The project owner then supplied the corpus snapshot used for
the `chat.py` comparison. It contains 1,670 passages from 51 of the 61 enabled
manifest sources. `config/corpus_policy.yaml` records the ten exclusions,
snapshot SHA-256, and reasons without changing Rayaan's upstream manifest. The
readiness validator accepts this 51-source snapshot and health reports the
reduced scope. It reproduces the five retrieved passages in the supplied
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
  `9868aa4fd0ac1bcec54a9ee1343f2cfa514174a1ef3d036462eaabc917a615b7`
- corpus SHA-256:
  `c185a35adfef3ebe971ac64018d6550ee4ecd9513515c210ca3b48f1976db713`

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
  51-source application corpus, not the full manifest used to define the
  intended research scope.
