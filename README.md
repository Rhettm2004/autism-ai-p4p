# Autism AI

A local Flutter prototype of the Autism AI screening flow. The app uses one
persistent workspace: the current screening card changes above an in-memory
chat history and a fixed chat input. The current session is serialized as one
JSON document in local shared preferences so it can survive an app restart or
browser refresh.

## Easiest way to run

From the project root, run:

```sh
./run_app.sh
```

Choose **1** for the fastest UI test with mock chat, or **2** to have the
launcher start the local Mistral model and connect Flutter to it. The launcher
stops any server it started when you quit Flutter with `Ctrl+C`.

You can skip the menu with `./run_app.sh mock` or `./run_app.sh mistral`.
`./run_app.sh backend` runs the complete research pipeline, but it will explain
and stop if the required verified corpus is unavailable.

## Included

- Age-based routing for Q-CHAT-10, AQ-10 Child, AQ-10 Adolescent, and AQ-10 Adult
- The official wording and answer options for all four 10-item questionnaires
- Respondent and background setup, answer review/editing, and disclaimer
- Deterministic mock screening result, research validation, and downloadable PDF report
- Persistent chat through the Python research backend, with explicit local and mock alternatives
- Continue-or-restart prompt when a saved local session is found
- Local validation and responsive, accessible Material UI

Supported routing is 18 to under 36 months for Q-CHAT-10, 3–11 years for
AQ-10 Child, 12–15 years for AQ-10 Adolescent, and 16–80 years for AQ-10 Adult.

## Local LLM Models

The Python backend talks to local llama.cpp servers. The legacy local chat provider can also connect directly. Each developer must download
the expected GGUF model files separately and should store them in `~/Models/`:

- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
- `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`

Start the local servers with:

```sh
llama-server -m ~/Models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --port 8080
```

```sh
llama-server -m ~/Models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --port 8081
```

Model weights are intentionally not stored in GitHub. Avoid committing `.gguf`
files because they are several GB each; `.gitignore` excludes them.

## Direct local Mistral comparison mode

Start the llama.cpp OpenAI-compatible server on `http://localhost:8080`, then
select the explicit direct-provider comparison mode. Override the URL at build time when needed:

```sh
flutter run -d chrome \
  --dart-define=CHAT_PROVIDER=local \
  --dart-define=LOCAL_LLM_BASE_URL=http://localhost:8080
```

Switch back to deterministic mock chat without changing code:

```sh
flutter run -d chrome --dart-define=USE_MOCK_CHAT=true
```

For Flutter Web, the local server must allow the app's origin through CORS and
support the browser's OPTIONS preflight request. An HTTPS-hosted app cannot call
an HTTP model server, and `localhost` always refers to the browser user's
device.

The Python backend integrates Rayaan’s routing, prompts and RAG. No CNN, vector
database, analytics, cloud storage, account system, or production data storage is
implemented. The prediction result remains mock data; classical questionnaire
scoring remains separate and local. Local shared preferences provide prototype session
restoration and are not storage for critical or production health data.

## Run and verify

```sh
flutter run
flutter analyze
flutter test
```

## Integrated Python backend (A–G)

The default chat provider is now the Python Autism AI backend. The screening UI,
questionnaire scoring, validation and PDF flow remain local. Rayaan's original
research code and artifacts are preserved under `backend/`.

See [backend setup and exact run commands](backend/DEVELOPMENT.md),
[research import provenance](backend/UPSTREAM.md), and
[verification results and blockers](backend/IMPLEMENTATION_STATUS.md).

Use `CHAT_PROVIDER=local` for the original direct llama.cpp comparison mode, or
`USE_MOCK_CHAT=true` for deterministic mock chat. The backend never silently falls
back when its corpus, router or model is unavailable. The CNN remains unimplemented
and the prediction result remains explicitly mocked.
