import 'dart:convert';

import 'package:autism_ai/data/question_banks.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/models/screening_session.dart';
import 'package:autism_ai/services/questionnaire_response_encoder.dart';
import 'package:autism_ai/services/report_service.dart';
import 'package:autism_ai/state/screening_controller.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('EAIP questionnaire encoding', () {
    test('encodes agreement uniformly for every age-specific AQ item', () {
      for (final type in [
        QuestionnaireType.aq10Child,
        QuestionnaireType.aq10Adolescent,
        QuestionnaireType.aq10Adult,
      ]) {
        final questions = questionBanks[type]!;
        for (var option = 0; option < aq10Options.length; option++) {
          final result = const QuestionnaireResponseEncoder().encodeModelItems(
            questionnaireType: type,
            answers: {for (final q in questions) q.id: q.options[option]},
          );
          expect(result.keys, [for (var i = 1; i <= 10; i++) 'Q$i']);
          expect(
            result.values,
            everyElement(option < 2 ? 1 : 0),
            reason: '${type.name}: ${aq10Options[option]}',
          );
        }
      }
    });

    test('non-agreement toddler responses encode as zero', () {
      final result = const QuestionnaireResponseEncoder().encodeModelItems(
        questionnaireType: QuestionnaireType.qchat10,
        answers: _qchatAnswersForScoreFour(),
      );
      expect(result.values, everyElement(0));
    });

    test('rejects responses outside the selected question options', () {
      final questions = questionBanks[QuestionnaireType.aq10Adult]!;
      expect(
        () => const QuestionnaireResponseEncoder().encodeModelItems(
          questionnaireType: QuestionnaireType.aq10Adult,
          answers: {for (final q in questions) q.id: 'Unknown'},
        ),
        throwsFormatException,
      );
    });

    test('rejects incomplete questionnaires', () {
      expect(
        () => const QuestionnaireResponseEncoder().encodeModelItems(
          questionnaireType: QuestionnaireType.qchat10,
          answers: const {},
        ),
        throwsFormatException,
      );
    });
  });

  group('screening report data', () {
    test(
      'includes respondent details, answers, EAIP input, and validation data',
      () {
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
        expect(
          report.questionnaireLabel,
          'Toddler screening — 18 to under 36 months',
        );

        expect(report.questionsAndAnswers, hasLength(10));
        for (var index = 0; index < qchat10Questions.length; index++) {
          final item = report.questionsAndAnswers[index];
          final sourceQuestion = qchat10Questions[index];
          expect(item.number, index + 1);
          expect(item.question, sourceQuestion.text);
          expect(item.answer, session.behaviouralAnswers[sourceQuestion.id]);
        }

        expect(report.aiResult.isMock, isFalse);
        expect(report.aiResult.traitsDetected, isFalse);
        expect(report.aiResult.similarityPercentage, 24);
        expect(report.assessmentStatus, assessmentStatuses.last);
        expect(report.includeDiagnosticTechnique, isTrue);
        expect(report.diagnosticTechnique, diagnosticTechniques[2]);
        expect(report.eaipInput.rawQuestionAnswers, hasLength(10));
        expect(report.eaipInput.questionEncodingConfirmed, isTrue);
        expect(report.eaipInput.age, 24);
        expect(report.eaipInput.ageUnit, 'months');
        expect(report.eaipInput.sex, 'm');
        expect(report.eaipInput.ethnicity, 'Asian');
        expect(report.eaipInput.jauntice, 'no');
        expect(report.eaipInput.familyAsdHistory, 'yes');
        expect(report.eaipInput.autismAgeCategory, 'chat');
        expect(report.eaipInput.toModelPayload(), containsPair('Q1', 0));
        expect(report.eaipInput.toModelPayload(), containsPair('Q10', 0));
      },
    );

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
      isMock: false,
      disagreement: 0.08,
      confidence: 0.92,
      tunedThreshold: 0.52,
    ),
    assessmentStatus: assessmentStatus,
    diagnosticTechnique: diagnosticTechnique,
  );
}
