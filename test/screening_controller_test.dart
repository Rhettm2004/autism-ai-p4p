import 'package:autism_ai/data/question_banks.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('official questionnaire data', () {
    test('contains four complete 10-question banks', () {
      expect(questionBanks.length, 4);
      for (final bank in questionBanks.values) {
        expect(bank.length, 10);
      }
    });

    test('preserves Q-CHAT-10 question-specific answer scales', () {
      expect(qchat10Questions.first.options, [
        'Always',
        'Usually',
        'Sometimes',
        'Rarely',
        'Never',
      ]);
      expect(qchat10Questions[1].options, [
        'Very easy',
        'Quite easy',
        'Quite difficult',
        'Very difficult',
        'Impossible',
      ]);
      expect(qchat10Questions[7].options.last, 'My child doesn’t speak');
    });
  });

  group('age routing', () {
    test('routes age 3 to AQ-10 Child', () {
      expect(_routeYears(3), QuestionnaireType.aq10Child);
    });

    test('routes the age 11 and 12 boundary', () {
      expect(_routeYears(11), QuestionnaireType.aq10Child);
      expect(_routeYears(12), QuestionnaireType.aq10Adolescent);
    });

    test('routes the age 15 and 16 boundary', () {
      expect(_routeYears(15), QuestionnaireType.aq10Adolescent);
      expect(_routeYears(16), QuestionnaireType.aq10Adult);
    });

    test('validates toddler month boundaries', () {
      final belowRange = _submitAge(age: 17, isToddler: true);
      final lowerBoundary = _submitAge(age: 18, isToddler: true);
      final upperBoundary = _submitAge(age: 35, isToddler: true);
      final aboveRange = _submitAge(age: 36, isToddler: true);

      expect(belowRange.questionnaireType, isNull);
      expect(lowerBoundary.questionnaireType, QuestionnaireType.qchat10);
      expect(upperBoundary.questionnaireType, QuestionnaireType.qchat10);
      expect(aboveRange.questionnaireType, isNull);
      expect(belowRange.errorMessage, contains('18 to under 36 months'));
      expect(aboveRange.errorMessage, contains('18 to under 36 months'));
    });

    test('limits the adult pathway to age 80', () {
      expect(_routeYears(80), QuestionnaireType.aq10Adult);
      expect(_routeYears(81), isNull);
    });
  });

  test('formal assessment controls the diagnostic-technique follow-up', () {
    final notAssessed = ScreeningController()
      ..setAssessmentStatus(assessmentStatuses.first);
    expect(notAssessed.requiresDiagnosticTechnique, isFalse);
    expect(notAssessed.submitValidation(), isTrue);

    for (final assessedStatus in assessmentStatuses.skip(1)) {
      final assessed = ScreeningController()
        ..setAssessmentStatus(assessedStatus);
      expect(assessed.requiresDiagnosticTechnique, isTrue);
      expect(assessed.submitValidation(), isFalse);

      assessed.setDiagnosticTechnique(diagnosticTechniques.first);
      expect(assessed.submitValidation(), isTrue);
    }
  });

  test('restart clears screening and chat state', () async {
    final controller = ScreeningController();
    controller.chooseToddlerPath(false);
    controller.submitRespondentDetails(
      gender: 'Male',
      ethnicity: 'Asian',
      ageText: '20',
    );
    await controller.sendChatMessage('hello');

    expect(controller.chatMessages, hasLength(3));
    expect(controller.chatMessages.last.isUser, isFalse);
    expect(
      controller.chatMessages.last.text,
      contains('connected in a later phase'),
    );

    await controller.restart();

    expect(controller.stage, ScreeningStage.welcome);
    expect(controller.questionnaireType, isNull);
    expect(controller.behaviouralAnswers, isEmpty);
    expect(controller.chatMessages, hasLength(1));
    expect(controller.chatMessages.single.isUser, isFalse);
  });

  test('completes the local screening flow through the report', () async {
    final controller = ScreeningController();
    controller.chooseToddlerPath(false);
    expect(
      controller.submitRespondentDetails(
        gender: 'Female',
        ethnicity: 'Pacifica',
        ageText: '16',
      ),
      isTrue,
    );
    controller.setJaundice(false);
    controller.setFamilyHistory(true);
    controller.setCompletedBy('Myself');
    expect(controller.submitBackgroundDetails(), isTrue);

    for (var index = 0; index < controller.questions.length; index++) {
      controller.answerCurrentQuestion(
        controller.currentQuestion!.options.first,
      );
      expect(controller.nextQuestion(), isTrue);
    }
    expect(controller.stage, ScreeningStage.review);

    controller.submitScreening();
    expect(controller.stage, ScreeningStage.disclaimer);
    await controller.acknowledgeDisclaimer();
    expect(controller.stage, ScreeningStage.result);
    expect(controller.result?.isMock, isTrue);

    controller.startValidation();
    controller.setAssessmentStatus(assessmentStatuses.first);
    expect(controller.submitValidation(), isTrue);
    expect(controller.stage, ScreeningStage.report);
  });
}

QuestionnaireType? _routeYears(int years) {
  return _submitAge(age: years, isToddler: false).questionnaireType;
}

ScreeningController _submitAge({required int age, required bool isToddler}) {
  final controller = ScreeningController();
  controller.chooseToddlerPath(isToddler);
  controller.submitRespondentDetails(
    gender: 'Female',
    ethnicity: 'White European',
    ageText: '$age',
  );
  return controller;
}
