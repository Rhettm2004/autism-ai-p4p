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
    QuestionnaireType.qchat10 => 'Q-CHAT-10 — Toddler',
    QuestionnaireType.aq10Child => 'AQ-10 — Child',
    QuestionnaireType.aq10Adolescent => 'AQ-10 — Adolescent',
    QuestionnaireType.aq10Adult => 'AQ-10 — Adult',
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
  });

  final bool traitsDetected;
  final double similarityPercentage;
  final bool isMock;
}

class ChatClassicalResult {
  const ChatClassicalResult({
    required this.questionnaireType,
    required this.score,
    required this.referralThreshold,
    required this.thresholdMet,
  });

  final QuestionnaireType questionnaireType;
  final int score;
  final int referralThreshold;
  final bool thresholdMet;
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
    this.classicalResult,
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
  final ChatClassicalResult? classicalResult;
  final QuestionnaireType? questionnaireType;
  final int? currentQuestionIndex;
  final String? currentQuestionText;
  final List<ScreeningQuestionContext> questionnaireQuestions;
  final ScreeningResult? result;
}
