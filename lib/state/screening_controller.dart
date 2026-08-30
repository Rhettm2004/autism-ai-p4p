import 'package:flutter/foundation.dart';

import '../data/question_banks.dart';
import '../models/screening_models.dart';
import '../services/mock_services.dart';

class ScreeningController extends ChangeNotifier {
  ScreeningController({
    ChatService? chatService,
    ScreeningPredictionService? predictionService,
  }) : _chatService = chatService ?? MockChatService(),
       _predictionService =
           predictionService ?? MockScreeningPredictionService() {
    _addWelcomeMessage();
  }

  final ChatService _chatService;
  final ScreeningPredictionService _predictionService;

  ScreeningStage stage = ScreeningStage.welcome;
  RespondentDetails respondent = RespondentDetails();
  BackgroundDetails background = BackgroundDetails();
  ValidationDetails validation = ValidationDetails();
  QuestionnaireType? questionnaireType;
  ScreeningResult? result;
  int currentQuestionIndex = 0;
  bool editingFromReview = false;
  bool isSendingChat = false;
  String? errorMessage;

  final Map<String, String> behaviouralAnswers = {};
  final List<ChatMessage> chatMessages = [];

  List<ScreeningQuestion> get questions =>
      questionnaireType == null ? const [] : questionBanks[questionnaireType]!;

  ScreeningQuestion? get currentQuestion => questions.isEmpty
      ? null
      : questions[currentQuestionIndex.clamp(0, questions.length - 1)];

  ScreeningContext get context => ScreeningContext(
    stage: stage,
    questionnaireType: questionnaireType,
    currentQuestionIndex: stage == ScreeningStage.behaviouralQuestions
        ? currentQuestionIndex
        : null,
    currentQuestionText: stage == ScreeningStage.behaviouralQuestions
        ? currentQuestion?.text
        : null,
  );

  String get stageLabel => switch (stage) {
    ScreeningStage.welcome => 'Welcome',
    ScreeningStage.toddlerCheck => 'Age pathway',
    ScreeningStage.respondentDetails => 'Respondent details',
    ScreeningStage.backgroundQuestions => 'Background questions',
    ScreeningStage.behaviouralQuestions =>
      'Question ${currentQuestionIndex + 1} of ${questions.length}',
    ScreeningStage.review => 'Review answers',
    ScreeningStage.disclaimer => 'Disclaimer',
    ScreeningStage.result => 'Screening result',
    ScreeningStage.validation => 'Research validation',
    ScreeningStage.report => 'Report',
  };

  double get overallProgress => switch (stage) {
    ScreeningStage.welcome => 0,
    ScreeningStage.toddlerCheck => 0.08,
    ScreeningStage.respondentDetails => 0.16,
    ScreeningStage.backgroundQuestions => 0.24,
    ScreeningStage.behaviouralQuestions =>
      0.28 + ((currentQuestionIndex + 1) / questions.length) * 0.36,
    ScreeningStage.review => 0.68,
    ScreeningStage.disclaimer => 0.72,
    ScreeningStage.result => 0.80,
    ScreeningStage.validation => 0.90,
    ScreeningStage.report => 1,
  };

  bool get chatEnabled => stage != ScreeningStage.disclaimer;

  void startScreening() {
    _goTo(ScreeningStage.toddlerCheck);
  }

  void backToToddlerCheck() {
    _goTo(ScreeningStage.toddlerCheck);
  }

  void backToRespondentDetails() {
    _goTo(ScreeningStage.respondentDetails);
  }

  void backToLastQuestion() {
    currentQuestionIndex = questions.length - 1;
    editingFromReview = false;
    _goTo(ScreeningStage.behaviouralQuestions);
  }

  void chooseToddlerPath(bool isToddler) {
    respondent.isToddler = isToddler;
    respondent.age = null;
    questionnaireType = null;
    _goTo(ScreeningStage.respondentDetails);
  }

  bool submitRespondentDetails({
    required String? gender,
    required String? ethnicity,
    required String ageText,
  }) {
    final age = int.tryParse(ageText.trim());
    if (gender == null) {
      return _fail('Select a gender to continue.');
    }
    if (ethnicity == null) {
      return _fail('Select an ethnicity to continue.');
    }
    if (age == null || age <= 0) {
      return _fail('Enter a valid age to continue.');
    }

    if (respondent.isToddler == true) {
      if (age < 18 || age >= 36) {
        return _fail(
          'The toddler pathway currently supports ages 18 to 35 months.',
        );
      }
      questionnaireType = QuestionnaireType.qchat10;
    } else {
      if (age == 3) {
        return _fail(
          'Age routing for 3-year-old respondents is pending confirmation.',
        );
      }
      if (age < 4) {
        return _fail(
          'For respondents under 3 years old, use the toddler pathway and enter age in months.',
        );
      }
      questionnaireType = switch (age) {
        >= 4 && <= 11 => QuestionnaireType.aq10Child,
        >= 12 && <= 15 => QuestionnaireType.aq10Adolescent,
        _ => QuestionnaireType.aq10Adult,
      };
    }

    respondent.gender = gender;
    respondent.ethnicity = ethnicity;
    respondent.age = age;
    _goTo(ScreeningStage.backgroundQuestions);
    return true;
  }

  void setJaundice(bool value) {
    background.jaundice = value;
    _clearErrorAndNotify();
  }

  void setFamilyHistory(bool value) {
    background.familyAutismHistory = value;
    _clearErrorAndNotify();
  }

  void setCompletedBy(String? value) {
    background.completedBy = value;
    _clearErrorAndNotify();
  }

  bool submitBackgroundDetails() {
    if (background.jaundice == null) {
      return _fail('Select a jaundice response to continue.');
    }
    if (background.familyAutismHistory == null) {
      return _fail('Select a family history response to continue.');
    }
    if (background.completedBy == null) {
      return _fail('Select who is completing the test.');
    }
    currentQuestionIndex = 0;
    editingFromReview = false;
    _goTo(ScreeningStage.behaviouralQuestions);
    return true;
  }

  void answerCurrentQuestion(String answer) {
    final question = currentQuestion;
    if (question == null) return;
    behaviouralAnswers[question.id] = answer;
    _clearErrorAndNotify();
  }

  bool nextQuestion() {
    final question = currentQuestion;
    if (question == null || behaviouralAnswers[question.id] == null) {
      return _fail('Select an answer before continuing.');
    }
    if (editingFromReview) {
      editingFromReview = false;
      _goTo(ScreeningStage.review);
      return true;
    }
    if (currentQuestionIndex == questions.length - 1) {
      _goTo(ScreeningStage.review);
    } else {
      currentQuestionIndex += 1;
      errorMessage = null;
      notifyListeners();
    }
    return true;
  }

  void previousQuestion() {
    if (editingFromReview) {
      editingFromReview = false;
      _goTo(ScreeningStage.review);
    } else if (currentQuestionIndex > 0) {
      currentQuestionIndex -= 1;
      errorMessage = null;
      notifyListeners();
    } else {
      _goTo(ScreeningStage.backgroundQuestions);
    }
  }

  void goToQuestion(int index) {
    if (index < 0 || index >= questions.length) return;
    currentQuestionIndex = index;
    editingFromReview = true;
    _goTo(ScreeningStage.behaviouralQuestions);
  }

  void submitScreening() {
    if (behaviouralAnswers.length != questions.length) {
      _fail('Answer all 10 questions before submitting.');
      return;
    }
    _goTo(ScreeningStage.disclaimer);
  }

  Future<void> acknowledgeDisclaimer() async {
    result = await _predictionService.predict(
      respondent: respondent,
      background: background,
      answers: Map.unmodifiable(behaviouralAnswers),
    );
    _goTo(ScreeningStage.result);
  }

  void startValidation() {
    _goTo(ScreeningStage.validation);
  }

  void setAssessmentStatus(String status) {
    validation.assessmentStatus = status;
    if (status == assessmentStatuses.first) {
      validation.diagnosticTechnique = null;
    }
    _clearErrorAndNotify();
  }

  void setDiagnosticTechnique(String? technique) {
    validation.diagnosticTechnique = technique;
    _clearErrorAndNotify();
  }

  bool submitValidation() {
    final status = validation.assessmentStatus;
    if (status == null) {
      return _fail('Select an assessment status to continue.');
    }
    if (status != assessmentStatuses.first &&
        validation.diagnosticTechnique == null) {
      return _fail('Select the diagnostic technique used.');
    }
    _goTo(ScreeningStage.report);
    return true;
  }

  Future<void> sendChatMessage(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty || !chatEnabled || isSendingChat) return;

    chatMessages.add(
      ChatMessage(text: trimmed, isUser: true, timestamp: DateTime.now()),
    );
    isSendingChat = true;
    notifyListeners();

    final response = await _chatService.sendMessage(trimmed, context);
    chatMessages.add(
      ChatMessage(text: response, isUser: false, timestamp: DateTime.now()),
    );
    isSendingChat = false;
    notifyListeners();
  }

  void restart() {
    stage = ScreeningStage.welcome;
    respondent = RespondentDetails();
    background = BackgroundDetails();
    validation = ValidationDetails();
    questionnaireType = null;
    result = null;
    currentQuestionIndex = 0;
    editingFromReview = false;
    isSendingChat = false;
    errorMessage = null;
    behaviouralAnswers.clear();
    chatMessages.clear();
    _addWelcomeMessage();
    notifyListeners();
  }

  void _addWelcomeMessage() {
    chatMessages.add(
      ChatMessage(
        text: 'Hi! I’m your Autism AI assistant. Ask me about the screening process.',
        isUser: false,
        timestamp: DateTime.now(),
      ),
    );
  }

  void _goTo(ScreeningStage nextStage) {
    stage = nextStage;
    errorMessage = null;
    notifyListeners();
  }

  bool _fail(String message) {
    errorMessage = message;
    notifyListeners();
    return false;
  }

  void _clearErrorAndNotify() {
    errorMessage = null;
    notifyListeners();
  }
}

const List<String> assessmentStatuses = [
  'No, never formally assessed',
  'Assessed, but autism was not diagnosed',
  'Assessed and autism was diagnosed',
];

const List<String> diagnosticTechniques = [
  'ADI-R',
  'ADOS-G',
  'ADOS-2',
  '3DI',
  'CARS',
  "I don't know",
  'Other',
];
