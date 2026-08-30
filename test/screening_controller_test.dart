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
    test('routes the supported age bands', () {
      expect(_routeToddler(24), QuestionnaireType.qchat10);
      expect(_routeYears(4), QuestionnaireType.aq10Child);
      expect(_routeYears(11), QuestionnaireType.aq10Child);
      expect(_routeYears(12), QuestionnaireType.aq10Adolescent);
      expect(_routeYears(15), QuestionnaireType.aq10Adolescent);
      expect(_routeYears(16), QuestionnaireType.aq10Adult);
      expect(_routeYears(42), QuestionnaireType.aq10Adult);
    });

    test('does not guess age-three routing', () {
      final controller = ScreeningController();
      controller.chooseToddlerPath(false);

      final accepted = controller.submitRespondentDetails(
        gender: 'Female',
        ethnicity: 'Pacific',
        ageText: '3',
      );

      expect(accepted, isFalse);
      expect(
        controller.errorMessage,
        'Age routing for 3-year-old respondents is pending confirmation.',
      );
      expect(controller.questionnaireType, isNull);
    });
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

    controller.restart();

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
        ethnicity: 'Pacific',
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

QuestionnaireType? _routeToddler(int months) {
  final controller = ScreeningController();
  controller.chooseToddlerPath(true);
  controller.submitRespondentDetails(
    gender: 'Male',
    ethnicity: 'Other',
    ageText: '$months',
  );
  return controller.questionnaireType;
}

QuestionnaireType? _routeYears(int years) {
  final controller = ScreeningController();
  controller.chooseToddlerPath(false);
  controller.submitRespondentDetails(
    gender: 'Female',
    ethnicity: 'European',
    ageText: '$years',
  );
  return controller.questionnaireType;
}
