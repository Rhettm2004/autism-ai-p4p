import '../data/question_banks.dart';
import '../models/screening_models.dart';

class ClassicalScreeningResult {
  const ClassicalScreeningResult({
    required this.questionnaireType,
    required this.score,
    required this.referralThreshold,
  });

  final QuestionnaireType questionnaireType;
  final int score;
  final int referralThreshold;

  bool get thresholdMet => score >= referralThreshold;

  String get questionnaireName =>
      questionnaireType == QuestionnaireType.qchat10 ? 'Q-CHAT-10' : 'AQ-10';

  String get thresholdStatement => thresholdMet
      ? 'The conventional screening threshold was met.'
      : 'The conventional screening threshold was not met.';
}

class QuestionnaireScoringService {
  const QuestionnaireScoringService();

  ClassicalScreeningResult calculate({
    required QuestionnaireType questionnaireType,
    required Map<String, String> answers,
  }) {
    final questions = questionBanks[questionnaireType]!;
    if (answers.length != questions.length ||
        questions.any((question) => answers[question.id] == null)) {
      throw const FormatException(
        'A classical score requires answers to all 10 questions.',
      );
    }

    final score = questionnaireType == QuestionnaireType.qchat10
        ? _scoreQchat10(answers)
        : _scoreAq10(questionnaireType, answers);

    return ClassicalScreeningResult(
      questionnaireType: questionnaireType,
      score: score,
      referralThreshold: questionnaireType == QuestionnaireType.qchat10 ? 3 : 6,
    );
  }

  int _scoreQchat10(Map<String, String> answers) {
    var score = 0;
    for (var index = 0; index < qchat10Questions.length; index++) {
      final question = qchat10Questions[index];
      final selectedIndex = question.options.indexOf(answers[question.id]!);
      if (selectedIndex == -1) {
        throw FormatException('Invalid answer for ${question.id}.');
      }

      // Official Q-CHAT-10 key: C/D/E score on items 1-9; A/B/C score
      // on reverse-scored item 10.
      if (index < 9 ? selectedIndex >= 2 : selectedIndex <= 2) {
        score += 1;
      }
    }
    return score;
  }

  int _scoreAq10(
    QuestionnaireType questionnaireType,
    Map<String, String> answers,
  ) {
    final agreeScoredItems = switch (questionnaireType) {
      QuestionnaireType.aq10Child => const {1, 5, 7, 10},
      QuestionnaireType.aq10Adolescent => const {1, 5, 8, 10},
      QuestionnaireType.aq10Adult => const {1, 7, 8, 10},
      QuestionnaireType.qchat10 => throw StateError(
        'Q-CHAT-10 must use its own scoring key.',
      ),
    };
    final questions = questionBanks[questionnaireType]!;
    var score = 0;

    for (var index = 0; index < questions.length; index++) {
      final question = questions[index];
      final selectedIndex = question.options.indexOf(answers[question.id]!);
      if (selectedIndex == -1) {
        throw FormatException('Invalid answer for ${question.id}.');
      }

      final itemNumber = index + 1;
      final agrees = selectedIndex <= 1;
      if (agreeScoredItems.contains(itemNumber) ? agrees : !agrees) {
        score += 1;
      }
    }
    return score;
  }
}
