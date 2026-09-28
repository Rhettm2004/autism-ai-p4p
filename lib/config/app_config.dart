abstract final class AppConfig {
  static const String chatProvider = String.fromEnvironment(
    'CHAT_PROVIDER',
    defaultValue: 'backend',
  );
  static const String backendBaseUrl = String.fromEnvironment(
    'AUTISM_AI_BACKEND_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );
  static const String backendModel = String.fromEnvironment(
    'AUTISM_AI_MODEL',
    defaultValue: 'mistral',
  );

  static const String localLlmBaseUrl = String.fromEnvironment(
    'LOCAL_LLM_BASE_URL',
    defaultValue: 'http://localhost:8080',
  );

  static const bool useMockChat = bool.fromEnvironment(
    'USE_MOCK_CHAT',
    defaultValue: false,
  );

  static const bool chatFirstScreening = bool.fromEnvironment(
    'CHAT_FIRST_SCREENING',
    defaultValue: true,
  );
}
