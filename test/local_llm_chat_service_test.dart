import 'dart:async';
import 'dart:convert';

import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/services/chat_service.dart';
import 'package:autism_ai/services/local_llm_chat_service.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  group('LocalLlmChatService request', () {
    test('sends OpenAI-compatible history and read-only app context', () async {
      late http.Request capturedRequest;
      late Map<String, dynamic> capturedBody;
      var requestCount = 0;
      final client = MockClient((request) async {
        requestCount += 1;
        capturedRequest = request;
        capturedBody = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode({
            'choices': [
              {
                'message': {'role': 'assistant', 'content': '  Be helpful.  '},
              },
            ],
          }),
          200,
        );
      });
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080/',
        client: client,
      );
      addTearDown(() {
        service.dispose();
        client.close();
      });
      final history = [
        _message('Welcome', isUser: false, minute: 0),
        _message('What does this mean?', isUser: true, minute: 1),
        _message('It means...', isUser: false, minute: 2),
        _message('Can you clarify?', isUser: true, minute: 3),
      ];

      final response = await service.sendMessage(
        message: 'Can you clarify?',
        history: history,
        context: _contextWithResult(),
      );
      await service.sendMessage(
        message: 'Can you clarify?',
        history: history,
        context: _contextWithResult(),
      );

      expect(response, 'Be helpful.');
      expect(requestCount, 2);
      expect(
        capturedRequest.url,
        Uri.parse('http://localhost:8080/v1/chat/completions'),
      );
      expect(capturedRequest.method, 'POST');
      expect(capturedRequest.headers['content-type'], 'application/json');
      expect(capturedRequest.headers['accept'], 'application/json');
      expect(capturedBody['model'], 'mistral-7b-instruct');
      expect(capturedBody['temperature'], 0.1);
      expect(capturedBody['max_tokens'], 300);
      expect(capturedBody['stream'], isFalse);

      final messages = (capturedBody['messages'] as List)
          .cast<Map<String, dynamic>>();
      expect(messages.map((item) => item['role']), [
        'system',
        'system',
        'assistant',
        'user',
        'assistant',
        'user',
      ]);
      expect(
        messages.first['content'],
        allOf(
          contains('Do not diagnose autism'),
          contains('Do not select or submit questionnaire answers'),
          contains('Do not change screening state'),
        ),
      );
      expect(
        messages[1]['content'],
        allOf(
          contains('Screening stage: behaviouralQuestions'),
          contains('Questionnaire: Q-CHAT-10'),
          contains('Current question number: 3'),
          contains('Current question text: Example screening question?'),
          contains('prototype AI flag not raised'),
          contains('similarity 24%'),
          contains('mock output'),
        ),
      );
      expect(
        messages.where((item) => item['content'] == 'Can you clarify?'),
        hasLength(1),
      );
    });

    test('limits the transcript to the latest 12 messages', () async {
      late List<dynamic> capturedMessages;
      final client = MockClient((request) async {
        final body = jsonDecode(request.body) as Map<String, dynamic>;
        capturedMessages = body['messages'] as List<dynamic>;
        return _successResponse('Done');
      });
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        client: client,
      );
      addTearDown(() {
        service.dispose();
        client.close();
      });
      final history = List.generate(
        14,
        (index) =>
            _message('message-$index', isUser: index.isEven, minute: index),
      );

      await service.sendMessage(
        message: 'latest-message',
        history: history,
        context: const ScreeningContext(stage: ScreeningStage.welcome),
      );

      expect(capturedMessages, hasLength(14));
      expect(capturedMessages[2]['content'], 'message-3');
      expect(capturedMessages.last['role'], 'user');
      expect(capturedMessages.last['content'], 'latest-message');
    });
  });

  group('LocalLlmChatService failures', () {
    test('maps request timeout to a friendly typed failure', () async {
      final client = MockClient((request) async {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        return _successResponse('Too late');
      });
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        client: client,
        timeout: const Duration(milliseconds: 1),
      );
      addTearDown(() {
        service.dispose();
        client.close();
      });

      await expectLater(
        _sendBasic(service),
        throwsA(_failureWithType(ChatFailureType.timeout)),
      );
    });

    test('maps client connection failures', () async {
      final client = MockClient((request) async {
        throw http.ClientException('Connection refused', request.url);
      });
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        client: client,
      );
      addTearDown(() {
        service.dispose();
        client.close();
      });

      await expectLater(
        _sendBasic(service),
        throwsA(_failureWithType(ChatFailureType.connection)),
      );
    });

    test('reports non-success status codes', () async {
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        client: MockClient((request) async => http.Response('busy', 503)),
      );
      addTearDown(service.dispose);

      await expectLater(
        _sendBasic(service),
        throwsA(
          isA<ChatServiceException>()
              .having((error) => error.type, 'type', ChatFailureType.server)
              .having((error) => error.statusCode, 'statusCode', 503),
        ),
      );
    });

    test('rejects malformed or incomplete response bodies', () async {
      final invalidBodies = [
        'not-json',
        jsonEncode({'choices': []}),
        jsonEncode({
          'choices': [
            {
              'message': {'content': '   '},
            },
          ],
        }),
      ];

      for (final body in invalidBodies) {
        final service = LocalLlmChatService(
          baseUrl: 'http://localhost:8080',
          client: MockClient((request) async => http.Response(body, 200)),
        );
        await expectLater(
          _sendBasic(service),
          throwsA(_failureWithType(ChatFailureType.invalidResponse)),
          reason: body,
        );
        service.dispose();
      }
    });
  });

  group('HTTP client ownership', () {
    test('reuses and does not close a caller-provided client', () async {
      final client = _CloseTrackingClient();
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        client: client,
      );

      await _sendBasic(service);
      await _sendBasic(service);
      expect(client.requestCount, 2);
      expect(client.wasClosed, isFalse);

      service.dispose();
      expect(client.wasClosed, isFalse);
      client.close();
    });

    test('closes only a client created by its client factory', () {
      final client = _CloseTrackingClient();
      final service = LocalLlmChatService(
        baseUrl: 'http://localhost:8080',
        clientFactory: () => client,
      );

      service.dispose();
      service.dispose();

      expect(client.wasClosed, isTrue);
      expect(client.closeCount, 1);
    });
  });

  group('ScreeningController chat behavior', () {
    test('disposes its chat service exactly once', () {
      final chatService = _ControlledChatService();
      final controller = ScreeningController(chatService: chatService);

      controller.dispose();
      controller.dispose();

      expect(chatService.disposeCount, 1);
    });

    test(
      'blocks duplicate sends while preserving loading and context',
      () async {
        final chatService = _ControlledChatService();
        final controller = ScreeningController(chatService: chatService);
        addTearDown(controller.dispose);
        controller.startScreening();

        final firstSend = controller.sendChatMessage('hello');
        final duplicateSend = controller.sendChatMessage('hello');

        expect(controller.isSendingChat, isTrue);
        expect(chatService.callCount, 1);
        expect(
          controller.chatMessages.where((message) => message.isUser),
          hasLength(1),
        );
        expect(chatService.history.last.text, 'hello');
        expect(chatService.context.stage, ScreeningStage.toddlerCheck);

        await duplicateSend;
        chatService.completer.complete('Mistral reply');
        await firstSend;

        expect(controller.isSendingChat, isFalse);
        expect(controller.chatMessages.last.text, 'Mistral reply');
        expect(controller.stage, ScreeningStage.toddlerCheck);
      },
    );

    test('shows a friendly error and recovers on the next send', () async {
      final chatService = _FailOnceChatService();
      final controller = ScreeningController(chatService: chatService);
      addTearDown(controller.dispose);
      controller.startScreening();

      await controller.sendChatMessage('first');

      expect(controller.isSendingChat, isFalse);
      expect(controller.chatMessages.last.text, contains('too long'));
      expect(controller.chatMessages.last.text, contains('try'));
      expect(controller.stage, ScreeningStage.toddlerCheck);

      await controller.sendChatMessage('second');

      expect(controller.isSendingChat, isFalse);
      expect(controller.chatMessages.last.text, 'Recovered response');
      expect(chatService.callCount, 2);
    });
  });
}

Future<String> _sendBasic(LocalLlmChatService service) {
  return service.sendMessage(
    message: 'Hello',
    history: [_message('Hello', isUser: true, minute: 0)],
    context: const ScreeningContext(stage: ScreeningStage.welcome),
  );
}

Matcher _failureWithType(ChatFailureType type) =>
    isA<ChatServiceException>().having((error) => error.type, 'type', type);

http.Response _successResponse(String content) {
  return http.Response(
    jsonEncode({
      'choices': [
        {
          'message': {'role': 'assistant', 'content': content},
        },
      ],
    }),
    200,
  );
}

ChatMessage _message(String text, {required bool isUser, required int minute}) {
  return ChatMessage(
    text: text,
    isUser: isUser,
    timestamp: DateTime(2026, 9, 6, 12, minute),
  );
}

ScreeningContext _contextWithResult() {
  return const ScreeningContext(
    stage: ScreeningStage.behaviouralQuestions,
    questionnaireType: QuestionnaireType.qchat10,
    currentQuestionIndex: 2,
    currentQuestionText: 'Example screening question?',
    result: ScreeningResult(
      traitsDetected: false,
      similarityPercentage: 24,
      isMock: true,
    ),
  );
}

class _CloseTrackingClient extends http.BaseClient {
  int requestCount = 0;
  int closeCount = 0;
  bool wasClosed = false;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    requestCount += 1;
    return http.StreamedResponse(
      Stream.value(utf8.encode(_successResponse('OK').body)),
      200,
      headers: const {'content-type': 'application/json'},
    );
  }

  @override
  void close() {
    closeCount += 1;
    wasClosed = true;
  }
}

class _ControlledChatService extends ChatService {
  final Completer<String> completer = Completer<String>();
  int callCount = 0;
  int disposeCount = 0;
  late List<ChatMessage> history;
  late ScreeningContext context;

  @override
  String get displayName => 'Controlled';

  @override
  Future<String> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) {
    callCount += 1;
    this.history = history;
    this.context = context;
    return completer.future;
  }

  @override
  void dispose() {
    disposeCount += 1;
  }
}

class _FailOnceChatService extends ChatService {
  int callCount = 0;

  @override
  String get displayName => 'Fail once';

  @override
  Future<String> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) async {
    callCount += 1;
    if (callCount == 1) {
      throw const ChatServiceException(ChatFailureType.timeout);
    }
    return 'Recovered response';
  }
}
