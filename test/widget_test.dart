import 'package:autism_ai/app.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('welcome and persistent mock chat are available', (tester) async {
    await tester.pumpWidget(const MyApp());

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
  });

  testWidgets('screening uses one shell while the current stage changes', (
    tester,
  ) async {
    await tester.pumpWidget(const MyApp());

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
}
