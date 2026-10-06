import '../data/question_banks.dart';
import '../models/screening_models.dart';

class QuestionnaireResponseEncoder {
  const QuestionnaireResponseEncoder();

  /// Converts each answer to the binary Q1-Q10 values expected by EAIP-DARV.
  /// The mapping follows the established item keys for the source question bank;
  /// no conventional total or threshold is calculated by the application.
  /// AQ keys: https://docs.autismresearchcentre.com/tests/AQ10.pdf
  /// https://docs.autismresearchcentre.com/tests/AQ10-Child.pdf
  /// https://docs.autismresearchcentre.com/tests/AQ10-Adolescent.pdf
  /// Agreement is scored only on the designated items; other items reverse it.
  Map<String, int> encodeModelItems({
    required QuestionnaireType questionnaireType,
    required Map<String, String> answers,
  }) {
    final questions = questionBanks[questionnaireType]!;
    if (answers.length != questions.length ||
        questions.any((question) => answers[question.id] == null)) {
      throw const FormatException(
        'EAIP-DARV input requires answers to all 10 questions.',
      );
    }
    return Map.unmodifiable({
      for (var index = 0; index < questions.length; index++)
        'Q${index + 1}': _itemScore(
          questionnaireType,
          index,
          questions[index],
          answers[questions[index].id]!,
        ),
    });
  }

  int _itemScore(
    QuestionnaireType questionnaireType,
    int index,
    ScreeningQuestion question,
    String answer,
  ) {
    final selectedIndex = question.options.indexOf(answer);
    if (selectedIndex == -1) {
      throw FormatException('Invalid answer for ${question.id}.');
    }
    if (questionnaireType == QuestionnaireType.qchat10) {
      return index < 9
          ? (selectedIndex >= 2 ? 1 : 0)
          : (selectedIndex <= 2 ? 1 : 0);
    }
    final itemNumber = index + 1;
    final agrees = selectedIndex <= 1;
    final agreeScoredItems = switch (questionnaireType) {
      QuestionnaireType.aq10Child => const {1, 5, 7, 10},
      QuestionnaireType.aq10Adolescent => const {1, 5, 8, 10},
      QuestionnaireType.aq10Adult => const {1, 7, 8, 10},
      QuestionnaireType.qchat10 => const <int>{},
    };
    return agreeScoredItems.contains(itemNumber)
        ? (agrees ? 1 : 0)
        : (agrees ? 0 : 1);
  }
}
