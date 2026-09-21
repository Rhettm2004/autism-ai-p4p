# Autism AI Flutter Shell

A local Flutter prototype of the Autism AI screening flow. The app uses one
persistent workspace: the current screening card changes above an in-memory
chat history and a fixed chat input. The current session is serialized as one
JSON document in local shared preferences so it can survive an app restart or
browser refresh.

## Included

- Age-based routing for Q-CHAT-10, AQ-10 Child, AQ-10 Adolescent, and AQ-10 Adult
- The official wording and answer options for all four 10-item questionnaires
- Respondent and background setup, answer review/editing, and disclaimer
- Deterministic mock screening result, research validation, and downloadable PDF report
- Persistent chat backed by a local Mistral server, with a selectable mock implementation
- Continue-or-restart prompt when a saved local session is found
- Local validation and responsive, accessible Material UI

Supported routing is 18 to under 36 months for Q-CHAT-10, 3–11 years for
AQ-10 Child, 12–15 years for AQ-10 Adolescent, and 16–80 years for AQ-10 Adult.

## Local LLM Models

The app currently talks to local llama.cpp servers. Each developer must download
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

## Local Mistral assistant

Start the llama.cpp OpenAI-compatible server on `http://localhost:8080`, then
run the web app normally. Override the URL at build time when needed:

```sh
flutter run -d chrome \
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

No RAG, agent routing, CNN/backend, vector database, analytics, cloud storage,
account system, or production data storage is implemented. The screening result
remains mock data. Local shared preferences provide prototype session
restoration and are not storage for critical or production health data.

## Run and verify

```sh
flutter run
flutter analyze
flutter test
```
