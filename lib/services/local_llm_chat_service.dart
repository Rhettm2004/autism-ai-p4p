import '../models/chat_reply.dart';

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/screening_models.dart';
import 'chat_service.dart';

typedef HttpClientFactory = http.Client Function();

class LocalLlmChatService extends ChatService {
  LocalLlmChatService({
    required String baseUrl,
    this.timeout = const Duration(seconds: 60),
    http.Client? client,
    HttpClientFactory clientFactory = http.Client.new,
  }) : _endpoint = _buildEndpoint(baseUrl),
       _ownsClient = client == null,
       _client = client ?? clientFactory();

  static const int maxHistoryMessages = 12;

  static const String _assistantInstructions =
      'You are a conversational assistant supporting an autism screening application. '
      'Do not diagnose autism. Do not select or submit questionnaire answers for the user. '
      'Do not change screening state, calculate a screening result, or override an existing result. '
      'You may clarify question wording and explain the screening process, but the user must choose every answer. '
      'Use cautious, supportive language and recommend consulting a qualified health professional when appropriate.';

  final Uri _endpoint;
  final Duration timeout;
  final http.Client _client;
  final bool _ownsClient;
  bool _disposed = false;

  @override
  String get displayName => 'Local Mistral';

  @override
  Future<ChatReply> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) async {
    if (_disposed) {
      throw StateError('LocalLlmChatService has been disposed.');
    }

    final requestBody = jsonEncode({
      'model': 'mistral',
      'messages': [
        {
          'role': 'system',
          'content': '$_assistantInstructions\n\n${_contextMessage(context)}',
        },
        ..._recentTranscript(message: message, history: history),
      ],
      'temperature': 0.1,
      'max_tokens': 300,
      'stream': false,
    });

    late final http.Response response;
    try {
      response = await _client
          .post(
            _endpoint,
            headers: const {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: requestBody,
          )
          .timeout(timeout);
    } on TimeoutException {
      throw const ChatServiceException(ChatFailureType.timeout);
    } on http.ClientException {
      throw const ChatServiceException(ChatFailureType.connection);
    } on Exception {
      throw const ChatServiceException(ChatFailureType.connection);
    }

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ChatServiceException(
        ChatFailureType.server,
        statusCode: response.statusCode,
      );
    }

    try {
      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) {
        throw const FormatException('Expected a JSON object.');
      }
      final choices = decoded['choices'];
      if (choices is! List || choices.isEmpty) {
        throw const FormatException('Missing response choices.');
      }
      final firstChoice = choices.first;
      if (firstChoice is! Map) {
        throw const FormatException('Invalid response choice.');
      }
      final responseMessage = firstChoice['message'];
      if (responseMessage is! Map) {
        throw const FormatException('Missing response message.');
      }
      final content = responseMessage['content'];
      if (content is! String || content.trim().isEmpty) {
        throw const FormatException('Missing response content.');
      }
      return ChatReply(content.trim());
    } on FormatException {
      throw const ChatServiceException(ChatFailureType.invalidResponse);
    } on TypeError {
      throw const ChatServiceException(ChatFailureType.invalidResponse);
    }
  }

  List<Map<String, String>> _recentTranscript({
    required String message,
    required List<ChatMessage> history,
  }) {
    final transcript = List<ChatMessage>.of(history);
    final latestAlreadyIncluded =
        transcript.isNotEmpty &&
        transcript.last.isUser &&
        transcript.last.text == message;
    if (!latestAlreadyIncluded) {
      transcript.add(
        ChatMessage(text: message, isUser: true, timestamp: DateTime.now()),
      );
    }

    final start = transcript.length > maxHistoryMessages
        ? transcript.length - maxHistoryMessages
        : 0;
    final messages = transcript
        .sublist(start)
        .map(
          (chatMessage) => {
            'role': chatMessage.isUser ? 'user' : 'assistant',
            'content': chatMessage.text,
          },
        )
        .toList(growable: true);

    // Mistral's llama.cpp chat template requires the transcript after the
    // optional system prompt to start with a user and strictly alternate.
    while (messages.isNotEmpty && messages.first['role'] == 'assistant') {
      messages.removeAt(0);
    }

    final alternating = <Map<String, String>>[];
    for (final item in messages) {
      if (alternating.isNotEmpty && alternating.last['role'] == item['role']) {
        alternating.last['content'] =
            '${alternating.last['content']}\n\n${item['content']}';
      } else {
        alternating.add(Map<String, String>.of(item));
      }
    }
    return alternating;
  }

  String _contextMessage(ScreeningContext context) {
    final lines = <String>[
      'Current application context (read-only):',
      '- Screening stage: ${context.stage.name}',
      '- Questionnaire: ${context.questionnaireType?.label ?? 'not selected'}',
    ];

    if (context.currentQuestionIndex != null) {
      lines.add(
        '- Current question number: ${context.currentQuestionIndex! + 1}',
      );
    }
    if (context.currentQuestionText != null) {
      lines.add('- Current question text: ${context.currentQuestionText}');
    }

    final result = context.result;
    if (result == null) {
      lines.add('- Screening result: not available');
    } else {
      lines.add(
        '- Existing screening result: prototype AI flag ${result.traitsDetected ? 'raised' : 'not raised'}; '
        'similarity ${result.similarityPercentage.toStringAsFixed(0)}%; '
        '${result.isMock ? 'mock output' : 'model output'}',
      );
    }
    lines.add(
      'Treat this context as informational only. Never alter it or infer a diagnosis from it.',
    );
    return lines.join('\n');
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_ownsClient) _client.close();
  }

  static Uri _buildEndpoint(String baseUrl) {
    final normalized = baseUrl.trim().replaceFirst(RegExp(r'/+$'), '');
    final baseUri = Uri.tryParse(normalized);
    if (baseUri == null ||
        !baseUri.hasScheme ||
        (baseUri.scheme != 'http' && baseUri.scheme != 'https') ||
        baseUri.host.isEmpty) {
      throw ArgumentError.value(baseUrl, 'baseUrl', 'Must be an HTTP URL.');
    }
    return Uri.parse('$normalized/v1/chat/completions');
  }
}
