import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import '../data/question_banks.dart';
import '../models/screening_models.dart';
import '../models/screening_session.dart';
import 'questionnaire_scoring_service.dart';

class ReportQuestionAnswer {
  const ReportQuestionAnswer({
    required this.number,
    required this.question,
    required this.answer,
  });

  final int number;
  final String question;
  final String answer;
}

class ScreeningReportData {
  const ScreeningReportData({
    required this.sessionId,
    required this.generatedAt,
    required this.age,
    required this.ageUnit,
    required this.gender,
    required this.ethnicity,
    required this.jaundice,
    required this.familyAutismHistory,
    required this.completedBy,
    required this.questionnaireType,
    required this.questionsAndAnswers,
    required this.aiResult,
    required this.classicalResult,
    required this.assessmentStatus,
    required this.diagnosticTechnique,
  });

  factory ScreeningReportData.fromSession(
    ScreeningSession session, {
    DateTime? generatedAt,
    QuestionnaireScoringService scoringService =
        const QuestionnaireScoringService(),
  }) {
    final questionnaireType = session.questionnaireType;
    final aiResult = session.result;
    if (questionnaireType == null || aiResult == null) {
      throw StateError(
        'The screening must have a questionnaire and AI result before a report can be generated.',
      );
    }

    final questions = questionBanks[questionnaireType]!;
    final questionsAndAnswers = <ReportQuestionAnswer>[];
    for (var index = 0; index < questions.length; index++) {
      final question = questions[index];
      final answer = session.behaviouralAnswers[question.id];
      if (answer == null) {
        throw StateError('Missing answer for ${question.id}.');
      }
      questionsAndAnswers.add(
        ReportQuestionAnswer(
          number: index + 1,
          question: question.text,
          answer: answer,
        ),
      );
    }

    return ScreeningReportData(
      sessionId: session.sessionId,
      generatedAt: generatedAt ?? DateTime.now(),
      age: session.age,
      ageUnit: session.ageUnit,
      gender: session.gender,
      ethnicity: session.ethnicity,
      jaundice: session.jaundice,
      familyAutismHistory: session.familyAutismHistory,
      completedBy: session.completedBy,
      questionnaireType: questionnaireType,
      questionsAndAnswers: List.unmodifiable(questionsAndAnswers),
      aiResult: aiResult,
      classicalResult: scoringService.calculate(
        questionnaireType: questionnaireType,
        answers: session.behaviouralAnswers,
      ),
      assessmentStatus: session.assessmentStatus,
      diagnosticTechnique: session.diagnosticTechnique,
    );
  }

  static const String screeningDisclaimer =
      'This report summarises a screening result only. It is not a diagnosis and cannot confirm or rule out autism. If you have concerns, discuss them with a qualified health professional.';

  static const String finalDisclaimer =
      'This prototype is intended for research and screening support. The AI result is mocked and is separate from the locally calculated conventional questionnaire score. Neither result replaces a formal clinical assessment.';

  final String sessionId;
  final DateTime generatedAt;
  final int? age;
  final String ageUnit;
  final String? gender;
  final String? ethnicity;
  final bool? jaundice;
  final bool? familyAutismHistory;
  final String? completedBy;
  final QuestionnaireType questionnaireType;
  final List<ReportQuestionAnswer> questionsAndAnswers;
  final ScreeningResult aiResult;
  final ClassicalScreeningResult classicalResult;
  final String? assessmentStatus;
  final String? diagnosticTechnique;

  String get filename => 'Autism_AI_Screening_Report_$sessionId.pdf';

  String get formattedGenerationDate {
    const months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
    ];
    return '${generatedAt.day} ${months[generatedAt.month - 1]} ${generatedAt.year}';
  }

  String get ageText => age == null ? 'Not provided' : '$age $ageUnit';

  String get questionnaireLabel => switch (questionnaireType) {
    QuestionnaireType.qchat10 => 'Q-CHAT-10 (Toddler)',
    QuestionnaireType.aq10Child => 'AQ-10 (Child)',
    QuestionnaireType.aq10Adolescent => 'AQ-10 (Adolescent)',
    QuestionnaireType.aq10Adult => 'AQ-10 (Adult)',
  };

  bool get includeDiagnosticTechnique =>
      diagnosticTechnique != null && diagnosticTechnique!.trim().isNotEmpty;
}

class ReportService {
  const ReportService();

  Future<Uint8List> generatePdf(ScreeningReportData data) async {
    final regularFont = pw.Font.ttf(
      await rootBundle.load('assets/fonts/Roboto-Regular.ttf'),
    );
    final boldFont = pw.Font.ttf(
      await rootBundle.load('assets/fonts/Roboto-Bold.ttf'),
    );
    final document = pw.Document(
      title: 'Autism AI Screening Report - ${data.sessionId}',
      author: 'Autism AI',
      subject: 'Autism screening report',
      creator: 'Autism AI Flutter application',
    );

    const navy = PdfColor.fromInt(0xFF0B1D51);
    const blue = PdfColor.fromInt(0xFF0B5DD7);
    const orange = PdfColor.fromInt(0xFFF05A1A);
    const paleBlue = PdfColor.fromInt(0xFFEEF5FF);
    const paleOrange = PdfColor.fromInt(0xFFFFF2EC);
    const border = PdfColor.fromInt(0xFFD7DFEC);
    const body = PdfColor.fromInt(0xFF26324B);
    const muted = PdfColor.fromInt(0xFF667085);

    final baseStyle = pw.TextStyle(
      font: regularFont,
      fontSize: 10,
      lineSpacing: 2.2,
      color: body,
    );
    final boldStyle = baseStyle.copyWith(font: boldFont);

    document.addPage(
      pw.MultiPage(
        pageTheme: pw.PageTheme(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.fromLTRB(42, 38, 42, 42),
          theme: pw.ThemeData(defaultTextStyle: baseStyle),
        ),
        header: (context) => pw.Padding(
          padding: const pw.EdgeInsets.only(bottom: 16),
          child: pw.Row(
            crossAxisAlignment: pw.CrossAxisAlignment.center,
            children: [
              pw.Container(
                width: 34,
                height: 34,
                alignment: pw.Alignment.center,
                decoration: pw.BoxDecoration(
                  color: blue,
                  borderRadius: pw.BorderRadius.circular(9),
                ),
                child: pw.Text(
                  'AI',
                  style: pw.TextStyle(
                    font: boldFont,
                    fontSize: 13,
                    color: PdfColors.white,
                  ),
                ),
              ),
              pw.SizedBox(width: 12),
              pw.Expanded(
                child: pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Text(
                      'Autism AI Screening Report',
                      style: pw.TextStyle(
                        font: boldFont,
                        fontSize: 18,
                        color: navy,
                      ),
                    ),
                    pw.SizedBox(height: 2),
                    pw.Text(
                      'Screening summary - not a diagnosis',
                      style: baseStyle.copyWith(fontSize: 8.5, color: muted),
                    ),
                  ],
                ),
              ),
              pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.end,
                children: [
                  pw.Text(
                    data.formattedGenerationDate,
                    style: boldStyle.copyWith(fontSize: 9, color: navy),
                  ),
                  pw.SizedBox(height: 3),
                  pw.Text(
                    'Session ID: ${data.sessionId}',
                    style: baseStyle.copyWith(fontSize: 8.5, color: muted),
                  ),
                ],
              ),
            ],
          ),
        ),
        footer: (context) => pw.Padding(
          padding: const pw.EdgeInsets.only(top: 12),
          child: pw.Row(
            children: [
              pw.Expanded(
                child: pw.Text(
                  'Autism AI - confidential screening summary',
                  style: baseStyle.copyWith(fontSize: 7.5, color: muted),
                ),
              ),
              pw.Text(
                'Page ${context.pageNumber} of ${context.pagesCount}',
                style: baseStyle.copyWith(fontSize: 7.5, color: muted),
              ),
            ],
          ),
        ),
        build: (context) => [
          _noticeBox(
            text: ScreeningReportData.screeningDisclaimer,
            background: paleOrange,
            borderColor: orange,
            textStyle: baseStyle,
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('AI Screening Result', blue, boldStyle),
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(14),
            decoration: pw.BoxDecoration(
              color: paleBlue,
              border: pw.Border.all(color: border),
              borderRadius: pw.BorderRadius.circular(8),
            ),
            child: pw.Column(
              crossAxisAlignment: pw.CrossAxisAlignment.start,
              children: [
                pw.Row(
                  children: [
                    pw.Container(
                      padding: const pw.EdgeInsets.symmetric(
                        horizontal: 7,
                        vertical: 3,
                      ),
                      decoration: pw.BoxDecoration(
                        color: orange,
                        borderRadius: pw.BorderRadius.circular(10),
                      ),
                      child: pw.Text(
                        'MOCK AI OUTPUT',
                        style: boldStyle.copyWith(
                          color: PdfColors.white,
                          fontSize: 7.5,
                        ),
                      ),
                    ),
                  ],
                ),
                pw.SizedBox(height: 9),
                pw.Text(
                  data.aiResult.traitsDetected
                      ? 'The prototype AI screening flag was raised for the submitted responses.'
                      : 'The prototype AI screening flag was not raised for the submitted responses.',
                  style: boldStyle.copyWith(fontSize: 12, color: navy),
                ),
                pw.SizedBox(height: 7),
                pw.Text(
                  'Mock similarity percentage: ${data.aiResult.similarityPercentage.toStringAsFixed(0)}%',
                  style: boldStyle.copyWith(color: blue),
                ),
                pw.SizedBox(height: 5),
                pw.Text(
                  'This mocked output is reserved for later replacement by the CNN model and is not used to calculate the conventional questionnaire score below.',
                  style: baseStyle.copyWith(fontSize: 9),
                ),
              ],
            ),
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('Respondent and Session Details', blue, boldStyle),
          _detailsBox(
            rows: [
              ('Respondent age', data.ageText),
              ('Gender', data.gender ?? 'Not provided'),
              ('Ethnicity', data.ethnicity ?? 'Not provided'),
              ('Jaundice', _yesNo(data.jaundice)),
              ('Family autism history', _yesNo(data.familyAutismHistory)),
              ('Test completed by', data.completedBy ?? 'Not provided'),
              ('Questionnaire used internally', data.questionnaireLabel),
            ],
            baseStyle: baseStyle,
            boldStyle: boldStyle,
            borderColor: border,
            labelColor: muted,
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('Behavioural Questions and Answers', blue, boldStyle),
          ...data.questionsAndAnswers.map(
            (item) => pw.Container(
              padding: const pw.EdgeInsets.symmetric(vertical: 8),
              decoration: const pw.BoxDecoration(
                border: pw.Border(
                  bottom: pw.BorderSide(color: border, width: 0.7),
                ),
              ),
              child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text(
                    'Q${item.number}. ${item.question}',
                    style: boldStyle.copyWith(fontSize: 9.5, color: navy),
                  ),
                  pw.SizedBox(height: 3),
                  pw.Text(
                    'Selected answer: ${item.answer}',
                    style: boldStyle.copyWith(fontSize: 9.5, color: blue),
                  ),
                ],
              ),
            ),
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('Classical Screening Result', orange, boldStyle),
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(14),
            decoration: pw.BoxDecoration(
              color: paleOrange,
              border: pw.Border.all(color: border),
              borderRadius: pw.BorderRadius.circular(8),
            ),
            child: pw.Column(
              crossAxisAlignment: pw.CrossAxisAlignment.start,
              children: [
                pw.Text(
                  '${data.classicalResult.questionnaireName} score: ${data.classicalResult.score} / 10',
                  style: boldStyle.copyWith(fontSize: 12, color: navy),
                ),
                pw.SizedBox(height: 5),
                pw.Text(
                  'Referral threshold: ${data.classicalResult.referralThreshold}',
                  style: boldStyle.copyWith(color: navy),
                ),
                pw.SizedBox(height: 5),
                pw.Text(
                  data.classicalResult.thresholdStatement,
                  style: boldStyle.copyWith(color: orange),
                ),
                pw.SizedBox(height: 7),
                pw.Text(
                  'This score was calculated locally using the conventional scoring key for ${data.questionnaireLabel}. It is separate from the mock AI screening result.',
                  style: baseStyle.copyWith(fontSize: 9),
                ),
              ],
            ),
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('Formal Assessment Response', blue, boldStyle),
          _detailsBox(
            rows: [
              (
                'Assessment/diagnosis response',
                data.assessmentStatus ?? 'Not provided',
              ),
              if (data.includeDiagnosticTechnique)
                ('Diagnostic technique', data.diagnosticTechnique!),
            ],
            baseStyle: baseStyle,
            boldStyle: boldStyle,
            borderColor: border,
            labelColor: muted,
          ),
          pw.SizedBox(height: 18),
          _sectionHeading('Final Disclaimer', blue, boldStyle),
          _noticeBox(
            text: ScreeningReportData.finalDisclaimer,
            background: paleBlue,
            borderColor: blue,
            textStyle: baseStyle,
          ),
        ],
      ),
    );

    return document.save();
  }

  static pw.Widget _sectionHeading(
    String title,
    PdfColor accent,
    pw.TextStyle boldStyle,
  ) {
    return pw.Padding(
      padding: const pw.EdgeInsets.only(bottom: 8),
      child: pw.Row(
        children: [
          pw.Container(width: 4, height: 18, color: accent),
          pw.SizedBox(width: 8),
          pw.Text(
            title,
            style: boldStyle.copyWith(fontSize: 14, color: accent),
          ),
        ],
      ),
    );
  }

  static pw.Widget _detailsBox({
    required List<(String, String)> rows,
    required pw.TextStyle baseStyle,
    required pw.TextStyle boldStyle,
    required PdfColor borderColor,
    required PdfColor labelColor,
  }) {
    return pw.Container(
      padding: const pw.EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: pw.BoxDecoration(
        border: pw.Border.all(color: borderColor),
        borderRadius: pw.BorderRadius.circular(8),
      ),
      child: pw.Column(
        children: rows
            .map(
              (row) => pw.Container(
                padding: const pw.EdgeInsets.symmetric(vertical: 6),
                decoration: pw.BoxDecoration(
                  border: pw.Border(
                    bottom: pw.BorderSide(color: borderColor, width: 0.5),
                  ),
                ),
                child: pw.Row(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.SizedBox(
                      width: 145,
                      child: pw.Text(
                        row.$1,
                        style: baseStyle.copyWith(color: labelColor),
                      ),
                    ),
                    pw.SizedBox(width: 10),
                    pw.Expanded(child: pw.Text(row.$2, style: boldStyle)),
                  ],
                ),
              ),
            )
            .toList(),
      ),
    );
  }

  static pw.Widget _noticeBox({
    required String text,
    required PdfColor background,
    required PdfColor borderColor,
    required pw.TextStyle textStyle,
  }) {
    return pw.Container(
      width: double.infinity,
      padding: const pw.EdgeInsets.all(12),
      decoration: pw.BoxDecoration(
        color: background,
        border: pw.Border.all(color: borderColor, width: 0.8),
        borderRadius: pw.BorderRadius.circular(6),
      ),
      child: pw.Text(text, style: textStyle.copyWith(fontSize: 9.5)),
    );
  }

  static String _yesNo(bool? value) => switch (value) {
    true => 'Yes',
    false => 'No',
    null => 'Not provided',
  };
}
