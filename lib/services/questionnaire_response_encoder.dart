import '../data/question_banks.dart';
import '../models/screening_models.dart';

class QuestionnaireResponseEncoder {
  const QuestionnaireResponseEncoder();

  /// Encodes agreement as 1 and every other valid response as 0.
  /// This is a model-input mapping, without questionnaire reverse scoring.
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
          questions[index],
          answers[questions[index].id]!,
        ),
    });
  }

  int _itemScore(ScreeningQuestion question, String answer) {
    if (!question.options.contains(answer)) {
      throw FormatException('Invalid answer for ${question.id}.');
    }
    return const {
          'Definitely Agree',
          'Strongly Agree',
          'Slightly Agree',
        }.contains(answer)
        ? 1
        : 0;
  }
}
