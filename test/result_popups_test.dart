import 'package:autism_ai/data/question_banks.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/screens/screening_page.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('popups close to the same result with three unchanged actions', (
    tester,
  ) async {
    final controller = ScreeningController();
    addTearDown(controller.dispose);
    controller.stage = ScreeningStage.result;
    controller.questionnaireType = QuestionnaireType.aq10Adult;
    controller.respondent
      ..age = 21
      ..isToddler = false
      ..gender = 'Male'
      ..ethnicity = 'Asian';
    controller.background
      ..jaundice = false
      ..familyAutismHistory = false;
    for (final q in questionBanks[QuestionnaireType.aq10Adult]!) {
      controller.behaviouralAnswers[q.id] = q.options.first;
    }
    controller.result = const ScreeningResult(
      traitsDetected: true,
      similarityPercentage: 53.4,
      isMock: false,
      submission: {
        'features': {'Q1': 1, 'Q9': 0, 'Q10': 1, 'Age': 21},
        'answers': [],
      },
    );
    await tester.pumpWidget(
      MaterialApp(home: ScreeningPage(controller: controller)),
    );
    await tester.pumpAndSettle();
    for (final label in ['View EAIP inputs', 'View / download report']) {
      await tester.ensureVisible(find.text(label));
      await tester.pumpAndSettle();
      await tester.tap(find.text(label));
      await tester.pumpAndSettle();
      expect(find.byType(Dialog), findsOneWidget);
      expect(controller.stage, ScreeningStage.result);
      expect(find.text('Continue Conversation'), findsNothing);
      if (label.contains('report')) {
        expect(find.text('EAIP-DARV model input record'), findsNothing);
        expect(find.text('Download report'), findsOneWidget);
      }
      await tester.tap(find.byTooltip('Close'));
      await tester.pumpAndSettle();
      expect(find.byType(Dialog), findsNothing);
      expect(find.text('Next steps'), findsOneWidget);
      expect(find.byKey(const Key('new-screening-header')), findsOneWidget);
    }
  });
}
