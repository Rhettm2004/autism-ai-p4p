import '../data/question_banks.dart';
import 'screening_models.dart';
import 'screening_session.dart';

/// A transparent preview of the fields expected by the EAIP-DARV model bundle.
///
/// The deployment bundle does not document how each questionnaire response was
/// converted to its numeric Q1-Q10 value. Until that mapping is confirmed, the
/// original answer is retained and the corresponding model value stays null.
class EaipModelInputPreview {
  const EaipModelInputPreview({
    required this.questionnaireType,
    required this.rawQuestionAnswers,
    required this.questionValues,
    required this.age,
    required this.ageUnit,
    required this.sex,
    required this.ethnicity,
    required this.jauntice,
    required this.familyAsdHistory,
    required this.autismAgeCategory,
  });

  factory EaipModelInputPreview.fromSession(ScreeningSession session) {
    final questionnaireType = session.questionnaireType;
    if (questionnaireType == null) {
      throw const FormatException(
        'A questionnaire is required to prepare the EAIP input preview.',
      );
    }

    final questions = questionBanks[questionnaireType]!;
    if (questions.any(
      (question) => session.behaviouralAnswers[question.id] == null,
    )) {
      throw const FormatException(
        'All questionnaire answers are required to prepare the EAIP input preview.',
      );
    }

    return EaipModelInputPreview(
      questionnaireType: questionnaireType,
      rawQuestionAnswers: Map.unmodifiable({
        for (var index = 0; index < questions.length; index++)
          'Q${index + 1}': session.behaviouralAnswers[questions[index].id]!,
      }),
      questionValues: Map.unmodifiable({
        for (var index = 0; index < questions.length; index++)
          'Q${index + 1}': null,
      }),
      age: session.age,
      ageUnit: session.ageUnit,
      sex: switch (session.gender?.trim().toLowerCase()) {
        'male' => 'm',
        'female' => 'f',
        _ => null,
      },
      ethnicity: session.ethnicity,
      jauntice: _yesNo(session.jaundice),
      familyAsdHistory: _yesNo(session.familyAutismHistory),
      autismAgeCategory: switch (questionnaireType) {
        QuestionnaireType.qchat10 => 'chat',
        QuestionnaireType.aq10Child => 'child',
        QuestionnaireType.aq10Adolescent => 'adolescent',
        QuestionnaireType.aq10Adult => 'adult',
      },
    );
  }

  final QuestionnaireType questionnaireType;
  final Map<String, String> rawQuestionAnswers;
  final Map<String, int?> questionValues;
  final int? age;
  final String ageUnit;
  final String? sex;
  final String? ethnicity;

  /// The misspelling is retained because it is part of the supplied schema.
  final String? jauntice;
  final String? familyAsdHistory;
  final String autismAgeCategory;

  bool get questionEncodingConfirmed =>
      questionValues.values.every((value) => value != null);

  Map<String, Object?> toModelPayload() => {
    ...questionValues,
    'Age': age,
    'Sex': sex,
    'Ethnicity': ethnicity,
    'Jauntice': jauntice,
    'FamilyASDHistory': familyAsdHistory,
    'AutismAgeCategory': autismAgeCategory,
  };

  static String? _yesNo(bool? value) => switch (value) {
    true => 'yes',
    false => 'no',
    null => null,
  };
}
