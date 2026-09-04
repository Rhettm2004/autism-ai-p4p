import 'dart:convert';

import 'package:autism_ai/data/question_banks.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/models/screening_session.dart';
import 'package:autism_ai/services/questionnaire_scoring_service.dart';
import 'package:autism_ai/services/report_service.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('classical questionnaire scoring', () {
    test('applies the official Q-CHAT-10 key and threshold', () {
      final result = const QuestionnaireScoringService().calculate(
        questionnaireType: QuestionnaireType.qchat10,
        answers: _qchatAnswersForScoreFour(),
      );

      expect(result.questionnaireName, 'Q-CHAT-10');
      expect(result.score, 4);
      expect(result.referralThreshold, 3);
      expect(result.thresholdMet, isTrue);
      expect(
        result.thresholdStatement,
        'The conventional screening threshold was met.',
      );
    });

    test('applies each official AQ-10 version key and threshold', () {
      for (final type in [
        QuestionnaireType.aq10Child,
        QuestionnaireType.aq10Adolescent,
        QuestionnaireType.aq10Adult,
      ]) {
        final result = const QuestionnaireScoringService().calculate(
          questionnaireType: type,
          answers: _aqAnswersScoringEveryItem(type),
        );

        expect(result.score, 10, reason: type.name);
        expect(result.referralThreshold, 6, reason: type.name);
        expect(result.thresholdMet, isTrue, reason: type.name);
      }
    });

    test('rejects incomplete questionnaires', () {
      expect(
        () => const QuestionnaireScoringService().calculate(
          questionnaireType: QuestionnaireType.qchat10,
          answers: const {},
        ),
        throwsFormatException,
      );
    });
  });

  group('screening report data', () {
    test('includes respondent details, all answers, AI, classical, and validation data', () {
      final session = _completedQchatSession(
        assessmentStatus: assessmentStatuses.last,
        diagnosticTechnique: diagnosticTechniques[2],
      );
      final report = ScreeningReportData.fromSession(
        session,
        generatedAt: DateTime(2026, 8, 28, 14, 30),
      );

      expect(report.sessionId, '21303');
      expect(report.formattedGenerationDate, '28 August 2026');
      expect(report.filename, 'Autism_AI_Screening_Report_21303.pdf');
      expect(report.ageText, '24 months');
      expect(report.gender, 'Male');
      expect(report.ethnicity, 'Asian');
      expect(report.jaundice, isFalse);
      expect(report.familyAutismHistory, isTrue);
      expect(report.completedBy, 'Family Member');
      expect(report.questionnaireLabel, 'Q-CHAT-10 (Toddler)');

      expect(report.questionsAndAnswers, hasLength(10));
      for (var index = 0; index < qchat10Questions.length; index++) {
        final item = report.questionsAndAnswers[index];
        final sourceQuestion = qchat10Questions[index];
        expect(item.number, index + 1);
        expect(item.question, sourceQuestion.text);
        expect(item.answer, session.behaviouralAnswers[sourceQuestion.id]);
      }

      expect(report.aiResult.isMock, isTrue);
      expect(report.aiResult.traitsDetected, isFalse);
      expect(report.aiResult.similarityPercentage, 24);
      expect(report.classicalResult.score, 4);
      expect(report.classicalResult.referralThreshold, 3);
      expect(report.classicalResult.thresholdMet, isTrue);
      expect(report.assessmentStatus, assessmentStatuses.last);
      expect(report.includeDiagnosticTechnique, isTrue);
      expect(report.diagnosticTechnique, diagnosticTechniques[2]);
    });

    test('omits diagnostic technique when it is not applicable', () {
      final report = ScreeningReportData.fromSession(
        _completedQchatSession(
          assessmentStatus: assessmentStatuses.first,
          diagnosticTechnique: null,
        ),
      );

      expect(report.assessmentStatus, assessmentStatuses.first);
      expect(report.includeDiagnosticTechnique, isFalse);
      expect(report.diagnosticTechnique, isNull);
    });

    test('generates a real PDF document', () async {
      final report = ScreeningReportData.fromSession(
        _completedQchatSession(
          assessmentStatus: assessmentStatuses.first,
          diagnosticTechnique: null,
        ),
        generatedAt: DateTime(2026, 8, 28),
      );

      final bytes = await const ReportService().generatePdf(report);

      expect(bytes.length, greaterThan(10000));
      expect(ascii.decode(bytes.take(5).toList()), '%PDF-');
    });
  });
}

Map<String, String> _qchatAnswersForScoreFour() {
  return {
    for (var index = 0; index < qchat10Questions.length; index++)
      qchat10Questions[index].id: switch (index) {
        0 || 1 || 2 => qchat10Questions[index].options[2],
        _ => qchat10Questions[index].options.first,
      },
  };
}

Map<String, String> _aqAnswersScoringEveryItem(QuestionnaireType type) {
  final agreeScoredItems = switch (type) {
    QuestionnaireType.aq10Child => const {1, 5, 7, 10},
    QuestionnaireType.aq10Adolescent => const {1, 5, 8, 10},
    QuestionnaireType.aq10Adult => const {1, 7, 8, 10},
    QuestionnaireType.qchat10 => throw ArgumentError.value(type),
  };
  final questions = questionBanks[type]!;
  return {
    for (var index = 0; index < questions.length; index++)
      questions[index].id: agreeScoredItems.contains(index + 1)
          ? questions[index].options.first
          : questions[index].options.last,
  };
}

ScreeningSession _completedQchatSession({
  required String assessmentStatus,
  required String? diagnosticTechnique,
}) {
  return ScreeningSession(
    sessionId: '21303',
    startedAt: DateTime(2026, 8, 28, 14),
    stage: ScreeningStage.report,
    isToddler: true,
    age: 24,
    ageUnit: 'months',
    gender: 'Male',
    ethnicity: 'Asian',
    jaundice: false,
    familyAutismHistory: true,
    completedBy: 'Family Member',
    questionnaireType: QuestionnaireType.qchat10,
    currentQuestionIndex: 9,
    editingFromReview: false,
    behaviouralAnswers: _qchatAnswersForScoreFour(),
    chatMessages: const [],
    result: const ScreeningResult(
      traitsDetected: false,
      similarityPercentage: 24,
      isMock: true,
    ),
    assessmentStatus: assessmentStatus,
    diagnosticTechnique: diagnosticTechnique,
  );
}
