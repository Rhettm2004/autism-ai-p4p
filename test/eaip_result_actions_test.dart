import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:autism_ai/widgets/stage_cards.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('result offers input inspection and report without validation', (
    tester,
  ) async {
    final controller = ScreeningController();
    addTearDown(controller.dispose);
    controller.stage = ScreeningStage.result;
    controller.result = const ScreeningResult(
      traitsDetected: false,
      similarityPercentage: 31.6,
      isMock: false,
    );
    var inspected = false;
    var reportOpened = false;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: ResultCard(
              controller: controller,
              onViewAnswers: () => inspected = true,
              onViewReport: () => reportOpened = true,
            ),
          ),
        ),
      ),
    );
    expect(find.text('View EAIP inputs'), findsOneWidget);
    expect(find.text('Next steps'), findsOneWidget);
    await tester.ensureVisible(find.text('View EAIP inputs'));
    await tester.tap(find.text('View EAIP inputs'));
    expect(inspected, isTrue);
    await tester.tap(find.text('View / download report'));
    expect(reportOpened, isTrue);
    expect(controller.stage, ScreeningStage.result);
    expect(controller.validation.assessmentStatus, isNull);
  });
}
