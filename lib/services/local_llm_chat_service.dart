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

  static const int maxHistoryMessages = 60;

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
    if (context.stage == ScreeningStage.welcome &&
        _isStartRequest(message, history)) {
      return ChatReply(
        'Of course. Let’s begin with a few details to select the appropriate questionnaire.',
        route: 'screening_guidance',
        model: 'mistral',
        action: ChatAction(
          type: ChatActionType.startScreening,
          expectedContextRevision: context.revision,
        ),
      );
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

    final managed = _managedTranscript(transcript);
    final messages = managed
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

  List<ChatMessage> _managedTranscript(List<ChatMessage> transcript) {
    final totalCharacters = transcript.fold<int>(
      0,
      (total, message) => total + message.text.length,
    );
    if (transcript.length <= maxHistoryMessages && totalCharacters <= 12000) {
      return transcript;
    }

    final recent = <ChatMessage>[];
    var recentCharacters = 0;
    for (final message in transcript.reversed) {
      if (recent.length >= maxHistoryMessages - 2 ||
          (recent.isNotEmpty &&
              recentCharacters + message.text.length > 9000)) {
        break;
      }
      recent.insert(0, message);
      recentCharacters += message.text.length;
    }
    final older = transcript.take(transcript.length - recent.length);
    final summary = StringBuffer(
      'Earlier conversation excerpts. U means user and A means assistant. '
      'Assistant statements are not user-provided facts.\n',
    );
    for (final item in older) {
      final normalized = item.text.replaceAll(RegExp(r'\s+'), ' ').trim();
      final excerpt = normalized.length > 220
          ? '${normalized.substring(0, 220)}…'
          : normalized;
      final line = '${item.isUser ? 'U' : 'A'}: $excerpt\n';
      if (summary.length + line.length > 3000) break;
      summary.write(line);
    }
    return [
      ChatMessage(
        text: summary.toString().trim(),
        isUser: true,
        timestamp: transcript.first.timestamp,
      ),
      ...recent,
    ];
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
    if (context.questionnaireQuestions.isNotEmpty) {
      lines.add('- Active questionnaire questions (read-only):');
      for (final question in context.questionnaireQuestions) {
        lines.add('  ${question.number}. ${question.text}');
      }
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

  bool _isStartRequest(String message, List<ChatMessage> history) {
    final text = message.toLowerCase().trim().replaceAll(RegExp(r'[.!?]'), '');
    if (text == '/start') {
      return true;
    }
    if (const [
      "don't start",
      'do not start',
      'not ready',
      'no screening',
      "don't want",
      'do not want',
    ].any(text.contains)) {
      return false;
    }
    if (RegExp(
          r'\b(start|begin|take|do|complete)\b.{0,24}\b(screening|questionnaire|test)\b|\b(screening|questionnaire|test)\b.{0,24}\b(start|begin|take|do|complete)\b',
        ).hasMatch(text) ||
        RegExp(
          r"\b(?:i am|i['’]?m|im|we are|we['’]?re)?\s*ready\b.{0,32}\b(?:start|begin|screening|questionnaire|test|get\s+started)\b|\blet['’]?s\s+(?:start|get\s+started)\b",
        ).hasMatch(text)) {
      return true;
    }
    if (!const {
      'yes',
      'yes please',
      'yeah',
      'yep',
      'sure',
      'okay',
      'ok',
      "let's start",
      'lets start',
      "let's get started",
      'lets get started',
      'ready',
      "i'm ready",
      'im ready',
      'i am ready',
    }.contains(text)) {
      return false;
    }
    final assistantHistory = history
        .where((entry) => !entry.isUser)
        .map((entry) => entry.text.toLowerCase())
        .join(' ');
    return const [
      'start a screening',
      'start screening',
      'begin a screening',
      'ready to start',
      'ready to get started',
      'say "yes"',
      "say 'yes'",
    ].any(assistantHistory.contains);
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
