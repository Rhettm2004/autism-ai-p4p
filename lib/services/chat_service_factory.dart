import '../config/app_config.dart';
import 'chat_service.dart';
import 'local_llm_chat_service.dart';
import 'mock_services.dart';

ChatService createConfiguredChatService() {
  if (AppConfig.useMockChat) return MockChatService();
  return LocalLlmChatService(baseUrl: AppConfig.localLlmBaseUrl);
}
