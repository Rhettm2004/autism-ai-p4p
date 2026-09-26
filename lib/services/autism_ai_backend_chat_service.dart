import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/chat_reply.dart';
import '../models/screening_models.dart';
import 'chat_service.dart';

class AutismAiBackendChatService extends ChatService {
  AutismAiBackendChatService({
    required String baseUrl,
    this.model = 'mistral',
    this.timeout = const Duration(seconds: 70),
    http.Client? client,
  }) : _endpoint = _endpointFor(baseUrl),
       _client = client ?? http.Client(),
       _ownsClient = client == null {
    if (!const ['mistral', 'llama'].contains(model)) {
      throw ArgumentError.value(model, 'model');
    }
  }
  final String model;
  final Duration timeout;
  final Uri _endpoint;
  final http.Client _client;
  final bool _ownsClient;
  bool _disposed = false;
  int _sequence = 0;
  Map<String, bool> _options = {
    'router': true,
    'rag': true,
    'cite': true,
    'concise': true,
  };
  @override
  String get displayName =>
      model == 'llama' ? 'Autism AI · Llama' : 'Autism AI · Mistral';

  @override
  Future<ChatReply> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) async {
    if (_disposed) throw StateError('Chat service disposed');
    final requestId = '${DateTime.now().microsecondsSinceEpoch}-${_sequence++}';
    final transcript = history.where((m) => !m.isError).toList();
    if (transcript.isNotEmpty &&
        transcript.last.isUser &&
        transcript.last.text == message) {
      transcript.removeLast();
    }
    final managedHistory = _managedTranscript(transcript);
    final classical = context.classicalResult;
    final prediction = context.result;
    final body = {
      'api_version': 1,
      'request_id': requestId,
      'session_id': context.sessionId,
      'message': message,
      'model': model,
      'options': _options,
      'history': managedHistory,
      'screening_context': {
        'revision': context.revision,
        'stage': context.stage.name,
        'screening_active':
            context.stage != ScreeningStage.welcome &&
            context.stage != ScreeningStage.report,
        'questionnaire': context.questionnaireType?.name,
        'current_question': context.currentQuestionId == null
            ? null
            : {
                'id': context.currentQuestionId,
                'number': context.currentQuestionIndex! + 1,
                'text': context.currentQuestionText,
              },
        'questionnaire_questions': context.questionnaireQuestions
            .map(
              (question) => {
                'id': question.id,
                'number': question.number,
                'text': question.text,
              },
            )
            .toList(),
        'classical_result': classical == null
            ? null
            : {
                'questionnaire': classical.questionnaireType.name,
                'score': classical.score,
                'referral_threshold': classical.referralThreshold,
                'threshold_met': classical.thresholdMet,
              },
        'prediction_result': prediction == null
            ? null
            : {
                'traits_detected': prediction.traitsDetected,
                'similarity_percentage': prediction.similarityPercentage,
                'is_mock': prediction.isMock,
              },
      },
    };
    late http.Response response;
    try {
      response = await _client
          .post(
            _endpoint,
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: jsonEncode(body),
          )
          .timeout(timeout);
    } on TimeoutException {
      throw const ChatServiceException(ChatFailureType.timeout);
    } on http.ClientException {
      throw const ChatServiceException(ChatFailureType.connection);
    }
    if (response.statusCode == 504) {
      throw const ChatServiceException(ChatFailureType.timeout);
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ChatServiceException(
        ChatFailureType.server,
        statusCode: response.statusCode,
      );
    }
    try {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      if (data['api_version'] != 1 ||
          data['request_id'] != requestId ||
          data['session_id'] != context.sessionId ||
          data['context_revision'] != context.revision ||
          data['model'] != model ||
          !data.containsKey('action')) {
        throw const FormatException('Unsupported or mismatched response');
      }
      final text = data['response'] as String;
      final route = data['route'] as String;
      if (text.trim().isEmpty ||
          !const {
            'safety_deflect',
            'misinformation_correction',
            'screening_guidance',
            'result_explanation',
            'referral',
            'general_knowledge',
            'caregiver_support',
          }.contains(route)) {
        throw const FormatException('Invalid reply');
      }
      final sources = (data['sources'] as List).map(
        (s) => ChatSource.fromJson(s as Map<String, dynamic>),
      );
      final returnedOptions = data['options'];
      if (returnedOptions is Map<String, dynamic>) {
        _options = {
          for (final key in const ['router', 'rag', 'cite', 'concise'])
            key: returnedOptions[key] as bool,
        };
      }
      final command = data['command'];
      final executedQuestion = command is Map<String, dynamic>
          ? command['executed_question'] as String?
          : null;
      final actionData = data['action'];
      final action = actionData == null
          ? null
          : ChatAction.fromJson(actionData as Map<String, dynamic>);
      return ChatReply(
        executedQuestion == null
            ? text.trim()
            : 'Demo question: $executedQuestion\n\n${text.trim()}',
        route: route,
        model: model,
        sources: List.unmodifiable(sources),
        action: action,
      );
    } on FormatException {
      throw const ChatServiceException(ChatFailureType.invalidResponse);
    } on TypeError {
      throw const ChatServiceException(ChatFailureType.invalidResponse);
    }
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_ownsClient) _client.close();
  }

  static Uri _endpointFor(String base) {
    final uri = Uri.tryParse(base.trim().replaceFirst(RegExp(r'/+$'), ''));
    if (uri == null ||
        !const ['http', 'https'].contains(uri.scheme) ||
        uri.host.isEmpty ||
        uri.hasQuery ||
        uri.hasFragment) {
      throw ArgumentError.value(base, 'baseUrl', 'Expected HTTP(S) base URL');
    }
    return Uri.parse('$uri/chat');
  }

  static List<Map<String, String>> _managedTranscript(
    List<ChatMessage> transcript,
  ) {
    const maxMessages = 60;
    const maxCharacters = 12000;
    final totalCharacters = transcript.fold<int>(
      0,
      (total, message) => total + message.text.length,
    );
    if (transcript.length <= maxMessages && totalCharacters <= maxCharacters) {
      return transcript
          .map(
            (message) => {
              'role': message.isUser ? 'user' : 'assistant',
              'content': message.text,
            },
          )
          .toList(growable: false);
    }

    final recent = <ChatMessage>[];
    var recentCharacters = 0;
    for (final message in transcript.reversed) {
      if (recent.length >= 58 ||
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
    for (final message in older) {
      final normalized = message.text.replaceAll(RegExp(r'\s+'), ' ').trim();
      final excerpt = normalized.length > 220
          ? '${normalized.substring(0, 220)}…'
          : normalized;
      final line = '${message.isUser ? 'U' : 'A'}: $excerpt\n';
      if (summary.length + line.length > 3000) break;
      summary.write(line);
    }
    return [
      {'role': 'user', 'content': summary.toString().trim()},
      ...recent.map(
        (message) => {
          'role': message.isUser ? 'user' : 'assistant',
          'content': message.text,
        },
      ),
    ];
  }
}
