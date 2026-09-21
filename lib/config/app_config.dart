abstract final class AppConfig {
  static const String localLlmBaseUrl = String.fromEnvironment(
    'LOCAL_LLM_BASE_URL',
    defaultValue: 'http://localhost:8080',
  );

  static const bool useMockChat = bool.fromEnvironment(
    'USE_MOCK_CHAT',
    defaultValue: false,
  );
}
