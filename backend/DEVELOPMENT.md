# Integrated Autism AI backend (stages A–G)

Flutter stays at the repository root. The screening UI, question banks, scoring,
validation and PDF reporting remain local and unchanged. Backend `/chat` cannot
mutate screening state and always returns `action: null`.

## Setup

Use Python 3.11, not the macOS system Python 3.9. From the repository root:

```sh
python3.11 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements-runtime.txt -r backend/requirements-dev.txt
cp backend/.env.app.example backend/.env
flutter pub get
```

The `.env.app.example` is new integration configuration. The original research
`.env.example`, `requirements.txt`, `README.md`, modules and results are preserved.
The MiniLM router needs its original Hugging Face model cache; first initialization
can download it. `HF_HOME` can point to an existing university cache. No lexical
fallback router is selected if MiniLM is unavailable.

### Prepare the real corpus

From `backend/`:

```sh
.venv/bin/python -m app.prepare_corpus
```

This invokes Rayaan's existing builder at 120 words / 30 overlap, records its output
in `logs/corpus-build.log`, and writes `data/corpus/readiness.json`. The report lists
missing sources and source/corpus hashes. The API refuses an incomplete corpus,
missing readiness report, or fingerprint mismatch. There is no benchmark-answer
fallback. Source material and downloads remain ignored, as upstream requires.

If Rayaan provides an authorized built corpus snapshot, place it at
`backend/data/corpus/corpus.csv` and run:

```sh
.venv/bin/python -m app.prepare_corpus --verify-existing
```

This checks source coverage; it does **not** establish that bytes match the research
snapshot or independently validate the extracted text. Retain his provenance/hash
separately. Review source extraction as well as the machine checks. A partial build
is not ready merely because the original builder exits successfully.

## Run locally

Terminal 1, using the model location found on this machine:

```sh
llama-server \
  -m "$HOME/Desktop/Sem 2 2026/SOFTENG 700A/Models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
  --alias mistral --host 127.0.0.1 --port 8080 --ctx-size 8192
```

Terminal 2, from `backend/`:

```sh
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 3, from the Flutter repository root:

```sh
flutter run -d chrome --web-hostname localhost --web-port 3000 \
  --dart-define=CHAT_PROVIDER=backend \
  --dart-define=AUTISM_AI_BACKEND_URL=http://127.0.0.1:8000 \
  --dart-define=AUTISM_AI_MODEL=mistral
```

`GET http://127.0.0.1:8000/health` reports router, corpus, prompts, and default model
readiness; unavailable dependencies return HTTP 503. Startup attempts all components
independently. Restart the backend after preparing missing assets. `/docs` exposes
the exact typed contract. Full prompts, answers and respondent data are not logged
by the integration. Original CLI transcript logging is not enabled by the API.

For Llama, use this server and `AUTISM_AI_MODEL=llama` in Flutter:

```sh
llama-server \
  -m "$HOME/Desktop/Sem 2 2026/SOFTENG 700A/Models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" \
  --alias llama --host 127.0.0.1 --port 8081 --ctx-size 8192
```

Set `AUTISM_AI_DEFAULT_MODEL=llama` in backend `.env` if Llama is your sole server;
health checks the configured default. Both aliases work through the same `/chat`
contract. Do not run both simultaneously if your machine lacks memory.

Development alternatives remain explicit:

```sh
flutter run -d chrome --dart-define=CHAT_PROVIDER=local --dart-define=LOCAL_LLM_BASE_URL=http://localhost:8080
flutter run -d chrome --dart-define=USE_MOCK_CHAT=true
```

Neither is an automatic fallback. Local mode retains the original Flutter prompt
and bypasses the integrated research pipeline. Mock prediction stays mocked in all
chat modes. URL changes for GPU hosting do not change the Flutter API; remote
hosting/authentication is outside these stages.

## Runtime contract

`POST /chat` accepts API version 1, request/session IDs, a nonblank message of at
most 4,000 characters, up to 12 previous user/assistant messages, model alias, and
read-only screening context. Request bodies are limited to 128 KiB. The current
message is not duplicated in history. Error bubbles are excluded from history.
Results contain distinct classical and prediction fields; no answer map or identity
fields are transmitted automatically.

Replies include text, route, model, sources and provenance hashes; `action` is always
null. Flutter rejects non-null actions and mismatched correlation IDs. Source links
only open HTTP(S) URLs. They identify retrieved supporting material, not verified
sentence-level citations. Tier-5 sources are visibly labelled commercial/blog.

Errors have `{error: {code, message, retryable, request_id}}`. Validation is 422,
oversized input 413, unavailable/capacity 503, timeout 504, invalid model reply 502,
and unexpected errors 500. Failed request bodies are not echoed. One inference runs
at a time per backend worker; additional requests return busy. Use one worker for
local development. History is bounded, but a sufficiently long prompt may still
exceed a model's context; this returns an explicit upstream error rather than
silently deleting safety instructions or retrieved evidence.

## Research fidelity

Runtime uses the original `src.chat.prepare_turn`, rule-first `embeddings_word`
router on `topic`/`extended`, and TF-IDF retrieval. Retrieval runs on every turn,
including safety routes: five passages, neighbour expansion enabled, original soft
per-source cap of two. No reranker, threshold, vector store or replacement router
has been introduced. Expansion can exceed the per-source cap. Concise and inline
citation instructions are off. Original few-shot/route prompt blocks are unchanged;
`config/application_prompts.yaml` adds separately versioned application constraints.
History/context are new app inputs. The original single-turn benchmark runner stays
unchanged. Research no-RAG behaviour is tested, but live app mode requires RAG.

Key differences from evaluated generation:

| Setting | Research | Local integration |
|---|---|---|
| Mistral | HF Mistral v0.1 | Mistral v0.3 Q4_K_M GGUF |
| Llama | HF ID is Llama **3.1** 8B | Local filename identifies Llama **3** 8B |
| Inference | Transformers / BitsAndBytes | llama.cpp OpenAI-compatible HTTP |
| Repetition | Completion-only custom logits processors | repeat penalty disabled; no claimed equivalent |
| Prompt | Single-turn measured blocks | Those blocks plus app constraints, history and read-only context |
| Corpus | Research snapshot | Locally prepared/supplied corpus with explicit fingerprint |

The adapter fixes temperature 0.1, max tokens 512, top-k 50, top-p 1, min-p 0 and
repeat/presence/frequency penalties 1/0/0. These are explicit app settings, not a
claim of HF sampling parity. Model aliases are backend configuration, not arbitrary
user-controlled model URLs. The local GGUF chat template is used by llama.cpp.
Original HF loading and completion-scoped generation controls remain available in
`src/model_runner.py` and `src/generation_control.py`; an HF API adapter is deferred.

`UPSTREAM.md` records ZIP identity. Research evidence and historical experiments
remain under their existing directories; cascade and LLM routers are not selected
at runtime. A benchmark prompt routed by a model fitted on the full training data
is not an unbiased evaluation. Keep out-of-fold and live measurements distinct.

## Tests

From `backend/`:

```sh
.venv/bin/python -m pytest app_tests -q
.venv/bin/python tests/test_chat.py
.venv/bin/python tests/test_retrieval_expansion.py
.venv/bin/python tests/test_route_prompts.py
.venv/bin/python tests/test_generation_control.py
.venv/bin/python tests/test_router_eval.py
.venv/bin/python tests/test_llm_router.py
```

Other original `tests/test_*.py` also have standalone runners. Several use custom
test registration, so do not assume pytest discovery runs them. Router tests need
the MiniLM cache; generation-control tests need torch, but not a 7B model. New API
tests use explicit fixture passages and fake generation; they never satisfy live
corpus readiness. Do not run `run_all.py` as a read-only test: it regenerates
research outputs.

From the Flutter root:

```sh
flutter analyze
flutter test
```

Live model verification is separate from mocked tests and from full RAG readiness.
See `IMPLEMENTATION_STATUS.md` for actual checks and unresolved issues.
