import '../models/screening_models.dart';

abstract class ChatService {
  String get displayName;

  Future<String> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  });

  void dispose() {}
}

enum ChatFailureType { timeout, connection, server, invalidResponse }

class ChatServiceException implements Exception {
  const ChatServiceException(this.type, {this.statusCode});

  final ChatFailureType type;
  final int? statusCode;

  String get userMessage => switch (type) {
    ChatFailureType.timeout => 'Local Mistral took too long to respond. Please try sending your message again.',
    ChatFailureType.connection => 'I could not reach the local Mistral server. Check that it is running, then try again.',
    ChatFailureType.server =>
      'Local Mistral returned an error. Please try sending your message again.',
    ChatFailureType.invalidResponse =>
      'I could not read the response from local Mistral. Please try again.',
  };

  @override
  String toString() {
    final status = statusCode == null ? '' : ' (HTTP $statusCode)';
    return 'ChatServiceException: ${type.name}$status';
  }
}
