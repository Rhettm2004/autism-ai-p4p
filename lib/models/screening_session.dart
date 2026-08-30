import 'screening_models.dart';

class ScreeningSession {
  const ScreeningSession({
    required this.stage,
    required this.isToddler,
    required this.age,
    required this.ageUnit,
    required this.gender,
    required this.ethnicity,
    required this.jaundice,
    required this.familyAutismHistory,
    required this.completedBy,
    required this.questionnaireType,
    required this.currentQuestionIndex,
    required this.editingFromReview,
    required this.behaviouralAnswers,
    required this.chatMessages,
    required this.result,
    required this.assessmentStatus,
    required this.diagnosticTechnique,
  });

  static const int currentVersion = 1;

  final ScreeningStage stage;
  final bool? isToddler;
  final int? age;
  final String ageUnit;
  final String? gender;
  final String? ethnicity;
  final bool? jaundice;
  final bool? familyAutismHistory;
  final String? completedBy;
  final QuestionnaireType? questionnaireType;
  final int currentQuestionIndex;
  final bool editingFromReview;
  final Map<String, String> behaviouralAnswers;
  final List<ChatMessage> chatMessages;
  final ScreeningResult? result;
  final String? assessmentStatus;
  final String? diagnosticTechnique;

  Map<String, dynamic> toJson() {
    return {
      'version': currentVersion,
      'stage': stage.name,
      'respondent': {
        'isToddler': isToddler,
        'age': age,
        'ageUnit': ageUnit,
        'gender': gender,
        'ethnicity': ethnicity,
      },
      'background': {
        'jaundice': jaundice,
        'familyAutismHistory': familyAutismHistory,
        'completedBy': completedBy,
      },
      'questionnaireType': questionnaireType?.name,
      'currentQuestionIndex': currentQuestionIndex,
      'editingFromReview': editingFromReview,
      'behaviouralAnswers': behaviouralAnswers,
      'chatMessages': chatMessages
          .map(
            (message) => {
              'text': message.text,
              'isUser': message.isUser,
              'timestamp': message.timestamp.toIso8601String(),
            },
          )
          .toList(),
      'result': result == null
          ? null
          : {
              'traitsDetected': result!.traitsDetected,
              'similarityPercentage': result!.similarityPercentage,
              'isMock': result!.isMock,
            },
      'validation': {
        'assessmentStatus': assessmentStatus,
        'diagnosticTechnique': diagnosticTechnique,
      },
    };
  }

  factory ScreeningSession.fromJson(Map<String, dynamic> json) {
    if (json['version'] != currentVersion) {
      throw const FormatException('Unsupported screening session version.');
    }

    final respondent = _jsonMap(json['respondent']);
    final background = _jsonMap(json['background']);
    final validation = _jsonMap(json['validation']);
    final rawAnswers = _jsonMap(json['behaviouralAnswers']);
    final rawMessages = json['chatMessages'];
    final rawResult = json['result'];

    return ScreeningSession(
      stage: _enumByName(ScreeningStage.values, json['stage'], 'stage'),
      isToddler: respondent['isToddler'] as bool?,
      age: (respondent['age'] as num?)?.toInt(),
      ageUnit: respondent['ageUnit'] as String? ?? 'years',
      gender: respondent['gender'] as String?,
      ethnicity: respondent['ethnicity'] as String?,
      jaundice: background['jaundice'] as bool?,
      familyAutismHistory: background['familyAutismHistory'] as bool?,
      completedBy: background['completedBy'] as String?,
      questionnaireType: _nullableEnumByName(
        QuestionnaireType.values,
        json['questionnaireType'],
      ),
      currentQuestionIndex:
          (json['currentQuestionIndex'] as num?)?.toInt() ?? 0,
      editingFromReview: json['editingFromReview'] as bool? ?? false,
      behaviouralAnswers: rawAnswers.map(
        (key, value) => MapEntry(key, value as String),
      ),
      chatMessages: rawMessages is List
          ? rawMessages.map((entry) {
              final message = _jsonMap(entry);
              return ChatMessage(
                text: message['text'] as String,
                isUser: message['isUser'] as bool,
                timestamp: DateTime.parse(message['timestamp'] as String),
              );
            }).toList()
          : const [],
      result: rawResult == null ? null : _resultFromJson(_jsonMap(rawResult)),
      assessmentStatus: validation['assessmentStatus'] as String?,
      diagnosticTechnique: validation['diagnosticTechnique'] as String?,
    );
  }

  static ScreeningResult _resultFromJson(Map<String, dynamic> json) {
    return ScreeningResult(
      traitsDetected: json['traitsDetected'] as bool,
      similarityPercentage: (json['similarityPercentage'] as num).toDouble(),
      isMock: json['isMock'] as bool? ?? true,
    );
  }
}

Map<String, dynamic> _jsonMap(Object? value) {
  if (value is! Map) return const {};
  return value.map((key, entry) => MapEntry(key.toString(), entry));
}

T _enumByName<T extends Enum>(List<T> values, Object? name, String fieldName) {
  final value = _nullableEnumByName(values, name);
  if (value == null) {
    throw FormatException('Invalid $fieldName in screening session.');
  }
  return value;
}

T? _nullableEnumByName<T extends Enum>(List<T> values, Object? name) {
  if (name is! String) return null;
  for (final value in values) {
    if (value.name == name) return value;
  }
  return null;
}
