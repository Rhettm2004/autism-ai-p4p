import 'chat_reply.dart';

enum ScreeningStage {
  welcome,
  toddlerCheck,
  respondentDetails,
  backgroundQuestions,
  behaviouralQuestions,
  review,
  disclaimer,
  result,
  validation,
  report,
}

enum QuestionnaireType { qchat10, aq10Child, aq10Adolescent, aq10Adult }

extension QuestionnaireTypeLabel on QuestionnaireType {
  String get label => switch (this) {
    QuestionnaireType.qchat10 => 'Toddler screening — 18 to under 36 months',
    QuestionnaireType.aq10Child => 'Child screening — 3 to 11 years',
    QuestionnaireType.aq10Adolescent => 'Adolescent screening — 12 to 15 years',
    QuestionnaireType.aq10Adult => 'Adult screening — 16 years and over',
  };
}

class ScreeningQuestion {
  const ScreeningQuestion({
    required this.id,
    required this.text,
    required this.options,
  });

  final String id;
  final String text;
  final List<String> options;
}

class RespondentDetails {
  bool? isToddler;
  String? gender;
  String? ethnicity;
  int? age;

  String get ageUnit => isToddler == true ? 'months' : 'years';
}

class BackgroundDetails {
  bool? jaundice;
  bool? familyAutismHistory;
  String? completedBy;
}

class ValidationDetails {
  String? assessmentStatus;
  String? diagnosticTechnique;
}

class ScreeningResult {
  const ScreeningResult({
    required this.traitsDetected,
    required this.similarityPercentage,
    required this.isMock,
    this.disagreement,
    this.confidence,
    this.perModuleRawProbability = const {},
    this.perModuleCalibratedProbability = const {},
    this.agreementScores = const {},
    this.tunedThreshold,
  });

  final bool traitsDetected;
  final double similarityPercentage;
  final bool isMock;
  final double? disagreement;
  final double? confidence;
  final Map<String, double> perModuleRawProbability;
  final Map<String, double> perModuleCalibratedProbability;
  final Map<String, double> agreementScores;
  final double? tunedThreshold;
}

class ChatMessage {
  const ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.sources = const [],
    this.route,
    this.model,
    this.isError = false,
  });

  final String text;
  final bool isUser;
  final DateTime timestamp;
  final List<ChatSource> sources;
  final String? route;
  final String? model;
  final bool isError;
}

class ScreeningQuestionContext {
  const ScreeningQuestionContext({
    required this.id,
    required this.number,
    required this.text,
  });

  final String id;
  final int number;
  final String text;
}

class ScreeningContext {
  const ScreeningContext({
    required this.stage,
    this.sessionId = 'local',
    this.revision = 0,
    this.currentQuestionId,
    this.questionnaireType,
    this.currentQuestionIndex,
    this.currentQuestionText,
    this.questionnaireQuestions = const [],
    this.result,
  });

  final ScreeningStage stage;
  final String sessionId;
  final int revision;
  final String? currentQuestionId;
  final QuestionnaireType? questionnaireType;
  final int? currentQuestionIndex;
  final String? currentQuestionText;
  final List<ScreeningQuestionContext> questionnaireQuestions;
  final ScreeningResult? result;
}
