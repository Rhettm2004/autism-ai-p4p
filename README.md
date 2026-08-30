# Autism AI Flutter Shell

A local Flutter prototype of the Autism AI screening flow. The app uses one
persistent workspace: the current screening card changes above an in-memory
chat history and a fixed chat input.

## Included

- Age-based routing for Q-CHAT-10, AQ-10 Child, AQ-10 Adolescent, and AQ-10 Adult
- The official wording and answer options for all four 10-item questionnaires
- Respondent and background setup, answer review/editing, and disclaimer
- Deterministic mock screening result, research validation, and report summary
- Persistent local mock chat and complete screening reset
- Local validation and responsive, accessible Material UI

Age routing for 3-year-old respondents is intentionally left unassigned pending
confirmation.

## Not connected in this phase

No LLM, RAG, agent routing, CNN/backend, vector database, external API, PDF
generation, analytics, or production data storage is implemented. Chat replies
and the screening result are clearly marked mock data.

## Run and verify

```sh
flutter run
flutter analyze
flutter test
```
