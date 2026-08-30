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
- Deterministic mock screening result, research validation, and report summary
- Persistent local mock chat and complete screening reset
- Continue-or-restart prompt when a saved local session is found
- Local validation and responsive, accessible Material UI

Supported routing is 18 to under 36 months for Q-CHAT-10, 3–11 years for
AQ-10 Child, 12–15 years for AQ-10 Adolescent, and 16–80 years for AQ-10 Adult.

## Not connected in this phase

No LLM, RAG, agent routing, CNN/backend, vector database, external API, PDF
generation, analytics, cloud storage, account system, or production data
storage is implemented. Chat replies and the screening result are clearly
marked mock data. Local shared preferences are prototype session restoration,
not storage for critical or production health data.

## Run and verify

```sh
flutter run
flutter analyze
flutter test
```
