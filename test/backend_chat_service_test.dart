import 'dart:async';
import 'dart:convert';

import 'package:autism_ai/models/chat_reply.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/models/screening_session.dart';
import 'package:autism_ai/services/autism_ai_backend_chat_service.dart';
import 'package:autism_ai/services/chat_service.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:autism_ai/widgets/persistent_chat_panel.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

Map<String, dynamic> reply(Map<String, dynamic> request) => {
  'api_version': 1,
  'request_id': request['request_id'],
  'session_id': request['session_id'],
  'context_revision': request['screening_context']['revision'],
  'response': 'A neutral explanation.',
  'route': 'screening_guidance',
  'model': request['model'],
  'action': null,
  'sources': [
    {
      'number': 1,
      'title': 'Instrument documentation',
      'url': 'https://example.org/source',
      'authority': '3',
      'low_authority': false,
      'passage_ids': ['instrument_c1'],
    },
  ],
};
ChatMessage message(String text, {bool isUser = true, bool isError = false}) =>
    ChatMessage(
      text: text,
      isUser: isUser,
      isError: isError,
      timestamp: DateTime(2026),
    );
const context = ScreeningContext(
  stage: ScreeningStage.behaviouralQuestions,
  sessionId: 'session-1',
  revision: 5,
  questionnaireType: QuestionnaireType.qchat10,
  currentQuestionId: 'qchat10_q4',
  currentQuestionIndex: 3,
  currentQuestionText: 'Official wording',
);

void main() {
  for (final model in ['mistral', 'llama']) {
    test('$model sends bounded history once and read-only context', () async {
      late Map<String, dynamic> captured;
      final client = MockClient((request) async {
        expect(request.url.toString(), 'http://localhost:8000/chat');
        captured = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(jsonEncode(reply(captured)), 200);
      });
      final service = AutismAiBackendChatService(
        baseUrl: 'http://localhost:8000/',
        model: model,
        client: client,
      );
      addTearDown(client.close);
      final result = await service.sendMessage(
        message: 'Explain',
        context: context,
        history: [
          ...List.generate(20, (i) => message('Earlier $i', isUser: i.isEven)),
          message('Service error', isUser: false, isError: true),
          message('Explain'),
        ],
      );
      expect(captured['history'], hasLength(12));
      expect(
        (captured['history'] as List).any((m) => m['content'] == 'Explain'),
        isFalse,
      );
      expect(captured['model'], model);
      expect(
        captured['screening_context']['current_question']['id'],
        'qchat10_q4',
      );
      expect(captured['screening_context'].containsKey('answers'), isFalse);
      expect(
        result.sources.single.link.toString(),
        'https://example.org/source',
      );
    });
  }

  test('rejects mismatched, malformed and action-bearing replies', () async {
    for (final change in <void Function(Map<String, dynamic>)>[
      (r) => r['action'] = {'type': 'set_answer', 'answer': 'Always'},
      (r) => r['session_id'] = 'other',
      (r) => r['request_id'] = 'other',
      (r) => r['context_revision'] = 99,
      (r) => r['model'] = 'llama',
      (r) => r['route'] = 'invented',
      (r) => r['sources'] = 'invalid',
      (r) => r['response'] = '',
    ]) {
      final client = MockClient((request) async {
        final data = reply(jsonDecode(request.body));
        change(data);
        return http.Response(jsonEncode(data), 200);
      });
      final service = AutismAiBackendChatService(
        baseUrl: 'http://localhost:8000',
        client: client,
      );
      await expectLater(
        service.sendMessage(message: 'Explain', history: [], context: context),
        throwsA(
          isA<ChatServiceException>().having(
            (e) => e.type,
            'type',
            ChatFailureType.invalidResponse,
          ),
        ),
      );
      client.close();
    }
  });

  test('unavailable backend surfaces failure without fallback', () async {
    var calls = 0;
    final client = MockClient((r) async {
      calls++;
      return http.Response('{}', 503);
    });
    final service = AutismAiBackendChatService(
      baseUrl: 'http://localhost:8000',
      client: client,
    );
    addTearDown(client.close);
    await expectLater(
      service.sendMessage(message: 'Explain', history: [], context: context),
      throwsA(isA<ChatServiceException>()),
    );
    expect(calls, 1);
  });

  test(
    'clarification never changes screening state and sources survive restore',
    () async {
      final client = MockClient(
        (r) async => http.Response(jsonEncode(reply(jsonDecode(r.body))), 200),
      );
      final controller = ScreeningController(
        chatService: AutismAiBackendChatService(
          baseUrl: 'http://localhost:8000',
          client: client,
        ),
      );
      addTearDown(() {
        controller.dispose();
        client.close();
      });
      controller.chooseToddlerPath(true);
      controller.submitRespondentDetails(
        gender: 'Female',
        ethnicity: 'Asian',
        ageText: '24',
      );
      controller.setJaundice(false);
      controller.setFamilyHistory(false);
      controller.setCompletedBy('Parent');
      controller.submitBackgroundDetails();
      controller.answerCurrentQuestion(
        controller.currentQuestion!.options.first,
      );
      final before = controller.sessionSnapshot.toJson();
      await controller.sendChatMessage('Explain this question');
      final after = controller.sessionSnapshot.toJson();
      for (final key in [
        'respondent',
        'background',
        'questionnaireType',
        'currentQuestionIndex',
        'behaviouralAnswers',
        'stage',
        'result',
        'validation',
      ]) {
        expect(after[key], before[key], reason: key);
      }
      final restored = ScreeningSession.fromJson(after);
      expect(
        restored.chatMessages.last.sources.single.title,
        'Instrument documentation',
      );
      final legacy = Map<String, dynamic>.from(before)..['version'] = 1;
      for (final item in legacy['chatMessages']) {
        (item as Map).remove('sources');
        item.remove('route');
        item.remove('model');
        item.remove('isError');
      }
      expect(ScreeningSession.fromJson(legacy).stage, controller.stage);
    },
  );

  test('restart drops old reply without clearing a newer request', () async {
    final service = Controlled();
    final controller = ScreeningController(chatService: service);
    addTearDown(controller.dispose);
    final old = controller.sendChatMessage('Old');
    await controller.restart();
    final current = controller.sendChatMessage('New');
    service.pending[0].complete(const ChatReply('Old reply'));
    await old;
    expect(controller.isSendingChat, isTrue);
    expect(controller.chatMessages.any((m) => m.text == 'Old reply'), isFalse);
    service.pending[1].complete(const ChatReply('New reply'));
    await current;
    expect(controller.chatMessages.last.text, 'New reply');
    expect(controller.isSendingChat, isFalse);
  });

  test('dispose ignores late failures and replies', () async {
    final service = Controlled();
    final controller = ScreeningController(chatService: service);
    final request = controller.sendChatMessage('Hi');
    final count = controller.chatMessages.length;
    controller.dispose();
    service.pending.single.completeError(
      const ChatServiceException(ChatFailureType.connection),
    );
    await request;
    expect(controller.chatMessages.length, count);
  });

  test('source URLs only allow HTTP(S)', () {
    const source = ChatSource(
      number: 1,
      title: 'Unsafe',
      url: 'javascript:alert(1)',
      authority: '5',
      lowAuthority: true,
      passageIds: [],
    );
    expect(source.link, isNull);
  });

  testWidgets(
    'chat displays source links and authority without replacing screening',
    (tester) async {
      final focus = FocusNode();
      addTearDown(focus.dispose);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PersistentChatPanel(
              messages: [
                ChatMessage(
                  text: 'Explanation',
                  isUser: false,
                  timestamp: DateTime(2026),
                  sources: const [
                    ChatSource(
                      number: 1,
                      title: 'Source title',
                      url: 'https://example.org',
                      authority: '5',
                      lowAuthority: true,
                      passageIds: ['x'],
                    ),
                  ],
                ),
              ],
              onSend: (_) async {},
              enabled: true,
              isSending: false,
              serviceLabel: 'Autism AI',
              inputFocusNode: focus,
            ),
          ),
        ),
      );
      expect(find.text('Retrieved supporting sources'), findsOneWidget);
      expect(find.text('1. Source title'), findsOneWidget);
      expect(find.textContaining('Commercial or blog source'), findsOneWidget);
    },
  );
}

class Controlled extends ChatService {
  final pending = <Completer<ChatReply>>[];
  @override
  String get displayName => 'Controlled';
  @override
  Future<ChatReply> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) {
    final completer = Completer<ChatReply>();
    pending.add(completer);
    return completer.future;
  }
}
