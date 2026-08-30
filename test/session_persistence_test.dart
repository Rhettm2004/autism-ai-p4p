import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/memory_screening_session_store.dart';

void main() {
  test('saves and restores respondent and background state', () async {
    final store = MemoryScreeningSessionStore();
    final source = await _controllerWith(store);
    addTearDown(source.dispose);

    source.chooseToddlerPath(false);
    source.setRespondentGender('Female');
    source.setRespondentEthnicity('Maori');
    source.setRespondentAge(12);
    expect(
      source.submitRespondentDetails(
        gender: 'Female',
        ethnicity: 'Maori',
        ageText: '12',
      ),
      isTrue,
    );
    source.setJaundice(true);
    source.setFamilyHistory(false);
    source.setCompletedBy('Parent');
    await source.flushPersistence();

    final restored = await _controllerWith(store);
    addTearDown(restored.dispose);
    expect(restored.hasRestorableSession, isTrue);
    expect(restored.stage, ScreeningStage.welcome);
    restored.continueSavedSession();

    expect(restored.stage, ScreeningStage.backgroundQuestions);
    expect(restored.respondent.isToddler, isFalse);
    expect(restored.respondent.age, 12);
    expect(restored.respondent.ageUnit, 'years');
    expect(restored.respondent.gender, 'Female');
    expect(restored.respondent.ethnicity, 'Maori');
    expect(restored.questionnaireType, QuestionnaireType.aq10Adolescent);
    expect(restored.background.jaundice, isTrue);
    expect(restored.background.familyAutismHistory, isFalse);
    expect(restored.background.completedBy, 'Parent');
  });

  test('restores the current behavioural question and answers', () async {
    final store = MemoryScreeningSessionStore();
    final source = await _adultAtBehaviouralQuestions(store);
    addTearDown(source.dispose);

    final firstQuestion = source.currentQuestion!;
    source.answerCurrentQuestion(firstQuestion.options.first);
    source.nextQuestion();
    final secondQuestion = source.currentQuestion!;
    source.answerCurrentQuestion(secondQuestion.options.last);
    await source.flushPersistence();

    final restored = await _controllerWith(store);
    addTearDown(restored.dispose);
    restored.continueSavedSession();

    expect(restored.stage, ScreeningStage.behaviouralQuestions);
    expect(restored.currentQuestionIndex, 1);
    expect(restored.currentQuestion?.id, secondQuestion.id);
    expect(
      restored.behaviouralAnswers[firstQuestion.id],
      firstQuestion.options.first,
    );
    expect(
      restored.behaviouralAnswers[secondQuestion.id],
      secondQuestion.options.last,
    );
  });

  test('restores mock chat history', () async {
    final store = MemoryScreeningSessionStore();
    final source = await _controllerWith(store);
    addTearDown(source.dispose);
    source.startScreening();
    await source.sendChatMessage('Can you explain this question?');
    await source.flushPersistence();

    final restored = await _controllerWith(store);
    addTearDown(restored.dispose);
    restored.continueSavedSession();

    expect(restored.chatMessages, hasLength(3));
    expect(restored.chatMessages[1].text, 'Can you explain this question?');
    expect(restored.chatMessages[1].isUser, isTrue);
    expect(restored.chatMessages.last.isUser, isFalse);
  });

  test('restores mock result and validation state', () async {
    final store = MemoryScreeningSessionStore();
    final source = await _adultAtBehaviouralQuestions(store);
    addTearDown(source.dispose);

    for (var index = 0; index < source.questions.length; index++) {
      source.answerCurrentQuestion(source.currentQuestion!.options.first);
      source.nextQuestion();
    }
    source.submitScreening();
    await source.acknowledgeDisclaimer();
    source.startValidation();
    source.setAssessmentStatus(assessmentStatuses.last);
    source.setDiagnosticTechnique(diagnosticTechniques[2]);
    await source.flushPersistence();

    final restored = await _controllerWith(store);
    addTearDown(restored.dispose);
    restored.continueSavedSession();

    expect(restored.stage, ScreeningStage.validation);
    expect(restored.result?.isMock, isTrue);
    expect(restored.result?.traitsDetected, isFalse);
    expect(restored.result?.similarityPercentage, 24);
    expect(restored.validation.assessmentStatus, assessmentStatuses.last);
    expect(restored.validation.diagnosticTechnique, diagnosticTechniques[2]);
  });

  test('Restart clears the saved session', () async {
    final store = MemoryScreeningSessionStore();
    final controller = await _controllerWith(store);
    addTearDown(controller.dispose);
    controller.startScreening();
    await controller.flushPersistence();
    expect(store.session, isNotNull);

    await controller.restart();

    expect(store.session, isNull);
    expect(controller.stage, ScreeningStage.welcome);
    expect(controller.chatMessages, hasLength(1));

    final nextLaunch = await _controllerWith(store);
    addTearDown(nextLaunch.dispose);
    expect(nextLaunch.hasRestorableSession, isFalse);
  });
}

Future<ScreeningController> _controllerWith(
  MemoryScreeningSessionStore store,
) async {
  final controller = ScreeningController(sessionStore: store);
  await controller.initializeSession();
  return controller;
}

Future<ScreeningController> _adultAtBehaviouralQuestions(
  MemoryScreeningSessionStore store,
) async {
  final controller = await _controllerWith(store);
  controller.chooseToddlerPath(false);
  controller.submitRespondentDetails(
    gender: 'Male',
    ethnicity: 'Asian',
    ageText: '24',
  );
  controller.setJaundice(false);
  controller.setFamilyHistory(false);
  controller.setCompletedBy('Myself');
  controller.submitBackgroundDetails();
  return controller;
}
