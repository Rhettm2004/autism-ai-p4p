import 'autism_ai_backend_chat_service.dart';
import '../config/app_config.dart';
import 'chat_service.dart';
import 'local_llm_chat_service.dart';
import 'mock_services.dart';

ChatService createConfiguredChatService() {
  if (AppConfig.useMockChat) return MockChatService();
  return switch (AppConfig.chatProvider) {
    'backend' => AutismAiBackendChatService(
      baseUrl: AppConfig.backendBaseUrl,
      model: AppConfig.backendModel,
    ),
    'local' => LocalLlmChatService(baseUrl: AppConfig.localLlmBaseUrl),
    'mock' => MockChatService(),
    _ => throw StateError(
      'Unsupported CHAT_PROVIDER: ${AppConfig.chatProvider}',
    ),
  };
}
