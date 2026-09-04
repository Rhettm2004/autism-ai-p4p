import 'package:autism_ai/app.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/memory_screening_session_store.dart';

void main() {
  testWidgets('welcome and persistent mock chat are available', (tester) async {
    final controller = ScreeningController();
    addTearDown(controller.dispose);
    await tester.pumpWidget(MyApp(controller: controller));
    await tester.pumpAndSettle();

    expect(find.text('Autism AI Assistant'), findsOneWidget);
    expect(find.byKey(const Key('chat-input')), findsOneWidget);
    expect(
      find.textContaining('Ask me about the screening process'),
      findsOneWidget,
    );

    await tester.enterText(
      find.byKey(const Key('chat-input')),
      'Can you explain this question?',
    );
    await tester.testTextInput.receiveAction(TextInputAction.send);
    await tester.pump();
    await tester.pumpAndSettle();

    expect(find.text('Can you explain this question?'), findsOneWidget);
    await controller.flushPersistence();
  });

  testWidgets('persistent chatbot shell remains visible across stages', (
    tester,
  ) async {
    final controller = ScreeningController();
    addTearDown(controller.dispose);
    await tester.pumpWidget(MyApp(controller: controller));
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const Key('start-screening')));
    await tester.pumpAndSettle();

    expect(
      find.text(
        'Are you taking the test for a toddler less than 36 months old?',
      ),
      findsOneWidget,
    );
    expect(find.byKey(const Key('chat-input')), findsOneWidget);

    await tester.tap(find.text('No'));
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    expect(find.text('Respondent details'), findsWidgets);
    expect(find.byKey(const Key('age-field')), findsOneWidget);
    expect(find.byKey(const Key('chat-input')), findsOneWidget);
  });

  testWidgets('no restore prompt appears without a saved session', (
    tester,
  ) async {
    final store = MemoryScreeningSessionStore();
    final controller = ScreeningController(sessionStore: store);
    addTearDown(controller.dispose);

    await tester.pumpWidget(MyApp(controller: controller));
    await tester.pumpAndSettle();

    expect(find.text('Continue where you left off?'), findsNothing);
    expect(find.text('Autism AI Assistant'), findsOneWidget);
  });

  testWidgets('Continue restores the saved stage from the startup prompt', (
    tester,
  ) async {
    final store = MemoryScreeningSessionStore();
    final source = ScreeningController(sessionStore: store);
    await source.initializeSession();
    source.startScreening();
    source.chooseToddlerPath(false);
    await source.flushPersistence();
    source.dispose();

    final restored = ScreeningController(sessionStore: store);
    addTearDown(restored.dispose);
    await tester.pumpWidget(MyApp(controller: restored));
    await tester.pumpAndSettle();

    expect(find.text('Continue where you left off?'), findsOneWidget);
    await tester.tap(find.byKey(const Key('continue-saved-session')));
    await tester.pumpAndSettle();

    expect(find.text('Respondent details'), findsWidgets);
    expect(find.byKey(const Key('chat-input')), findsOneWidget);
  });
}
