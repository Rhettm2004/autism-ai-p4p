import 'dart:async';

import 'package:flutter/foundation.dart';

import '../data/question_banks.dart';
import '../models/screening_models.dart';
import '../models/screening_session.dart';
import '../services/mock_services.dart';
import '../services/screening_session_store.dart';

class ScreeningController extends ChangeNotifier {
  ScreeningController({
    ChatService? chatService,
    ScreeningPredictionService? predictionService,
    ScreeningSessionStore? sessionStore,
  }) : _chatService = chatService ?? MockChatService(),
       _predictionService =
           predictionService ?? MockScreeningPredictionService(),
       _sessionStore = sessionStore ?? const NoopScreeningSessionStore() {
    _addWelcomeMessage();
  }

  final ChatService _chatService;
  final ScreeningPredictionService _predictionService;
  final ScreeningSessionStore _sessionStore;

  Timer? _saveTimer;
  Future<void> _writeQueue = Future.value();
  ScreeningSession? _pendingSession;
  bool _persistenceReady = false;
  int _persistenceGeneration = 0;

  ScreeningStage stage = ScreeningStage.welcome;
  RespondentDetails respondent = RespondentDetails();
  BackgroundDetails background = BackgroundDetails();
  ValidationDetails validation = ValidationDetails();
  QuestionnaireType? questionnaireType;
  ScreeningResult? result;
  int currentQuestionIndex = 0;
  bool editingFromReview = false;
  bool isSendingChat = false;
  bool isSessionLoading = false;
  String? errorMessage;

  final Map<String, String> behaviouralAnswers = {};
  final List<ChatMessage> chatMessages = [];

  bool get hasRestorableSession => _pendingSession != null;

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

  Future<void> initializeSession() async {
    if (_persistenceReady || isSessionLoading) return;

    isSessionLoading = true;
    _notify(persist: false);
    try {
      _pendingSession = await _sessionStore.load();
    } catch (error) {
      debugPrint('Unable to load the local screening session: $error');
      _pendingSession = null;
    }
    _persistenceReady = true;
    isSessionLoading = false;
    _notify(persist: false);
  }

  void continueSavedSession() {
    final savedSession = _pendingSession;
    if (savedSession == null) return;

    _pendingSession = null;
    _applySession(savedSession);
    _notify(persist: false);
  }

  Future<void> restartSavedSession() => restart();

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

  void setRespondentGender(String gender) {
    respondent.gender = gender;
    _clearErrorAndNotify();
  }

  void setRespondentEthnicity(String? ethnicity) {
    respondent.ethnicity = ethnicity;
    _clearErrorAndNotify();
  }

  void setRespondentAge(int? age) {
    respondent.age = age;
    _clearErrorAndNotify();
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
          'Enter an age from 18 to under 36 months for the toddler pathway.',
        );
      }
      questionnaireType = QuestionnaireType.qchat10;
    } else {
      if (age < 3 || age > 80) {
        return _fail(
          'Enter an age from 3 to 80 years for the non-toddler pathway.',
        );
      }
      questionnaireType = switch (age) {
        >= 3 && <= 11 => QuestionnaireType.aq10Child,
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
      _notify();
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
      _notify();
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
    if (!requiresDiagnosticTechnique) {
      validation.diagnosticTechnique = null;
    }
    _clearErrorAndNotify();
  }

  bool get requiresDiagnosticTechnique =>
      validation.assessmentStatus != null &&
      validation.assessmentStatus != assessmentStatuses.first;

  void setDiagnosticTechnique(String? technique) {
    validation.diagnosticTechnique = technique;
    _clearErrorAndNotify();
  }

  bool submitValidation() {
    final status = validation.assessmentStatus;
    if (status == null) {
      return _fail('Select an assessment status to continue.');
    }
    if (requiresDiagnosticTechnique && validation.diagnosticTechnique == null) {
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
    _notify();

    final response = await _chatService.sendMessage(trimmed, context);
    chatMessages.add(
      ChatMessage(text: response, isUser: false, timestamp: DateTime.now()),
    );
    isSendingChat = false;
    _notify();
  }

  Future<void> restart() async {
    _saveTimer?.cancel();
    _saveTimer = null;
    _persistenceGeneration += 1;
    _pendingSession = null;
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
    _notify(persist: false);

    await _writeQueue;
    await _sessionStore.clear();
  }

  Future<void> flushPersistence() async {
    _saveTimer?.cancel();
    _saveTimer = null;
    if (!_canPersist) {
      await _writeQueue;
      return;
    }

    _queueSessionSave(_createSession(), _persistenceGeneration);
    await _writeQueue;
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
    _notify();
  }

  bool _fail(String message) {
    errorMessage = message;
    _notify(persist: false);
    return false;
  }

  void _clearErrorAndNotify() {
    errorMessage = null;
    _notify();
  }

  bool get _canPersist =>
      _persistenceReady &&
      !hasRestorableSession &&
      (stage != ScreeningStage.welcome ||
          chatMessages.any((message) => message.isUser));

  void _notify({bool persist = true}) {
    notifyListeners();
    if (persist) _scheduleSessionSave();
  }

  void _scheduleSessionSave() {
    if (!_canPersist) return;
    _saveTimer?.cancel();
    final generation = _persistenceGeneration;
    _saveTimer = Timer(const Duration(milliseconds: 120), () {
      if (!_canPersist || generation != _persistenceGeneration) return;
      _queueSessionSave(_createSession(), generation);
    });
  }

  void _queueSessionSave(ScreeningSession session, int generation) {
    _writeQueue = _writeQueue.then(
      (_) => _writeSessionSafely(session, generation),
    );
  }

  Future<void> _writeSessionSafely(
    ScreeningSession session,
    int generation,
  ) async {
    if (generation != _persistenceGeneration) return;
    try {
      await _sessionStore.save(session);
    } catch (error) {
      debugPrint('Unable to save the local screening session: $error');
    }
  }

  ScreeningSession _createSession() {
    return ScreeningSession(
      stage: stage,
      isToddler: respondent.isToddler,
      age: respondent.age,
      ageUnit: respondent.ageUnit,
      gender: respondent.gender,
      ethnicity: respondent.ethnicity,
      jaundice: background.jaundice,
      familyAutismHistory: background.familyAutismHistory,
      completedBy: background.completedBy,
      questionnaireType: questionnaireType,
      currentQuestionIndex: currentQuestionIndex,
      editingFromReview: editingFromReview,
      behaviouralAnswers: Map.unmodifiable(behaviouralAnswers),
      chatMessages: List.unmodifiable(chatMessages),
      result: result,
      assessmentStatus: validation.assessmentStatus,
      diagnosticTechnique: validation.diagnosticTechnique,
    );
  }

  void _applySession(ScreeningSession session) {
    respondent = RespondentDetails()
      ..isToddler = session.isToddler
      ..age = session.age
      ..gender = session.gender
      ..ethnicity = session.ethnicity;
    background = BackgroundDetails()
      ..jaundice = session.jaundice
      ..familyAutismHistory = session.familyAutismHistory
      ..completedBy = session.completedBy;
    validation = ValidationDetails()
      ..assessmentStatus = session.assessmentStatus
      ..diagnosticTechnique = session.diagnosticTechnique;
    questionnaireType = session.questionnaireType;
    behaviouralAnswers
      ..clear()
      ..addAll(session.behaviouralAnswers);
    chatMessages
      ..clear()
      ..addAll(session.chatMessages);
    if (chatMessages.isEmpty) _addWelcomeMessage();
    result = session.result;
    editingFromReview = session.editingFromReview;
    isSendingChat = false;
    errorMessage = null;

    currentQuestionIndex = questions.isEmpty
        ? 0
        : session.currentQuestionIndex.clamp(0, questions.length - 1);
    stage = session.stage;
  }

  @override
  void dispose() {
    _saveTimer?.cancel();
    super.dispose();
  }
}

const List<String> assessmentStatuses = [
  'No, the respondent has never been formally assessed',
  'Yes, the respondent has been assessed BUT autism was not diagnosed',
  'Yes, the respondent has been assessed AND autism was diagnosed',
];

const List<String> diagnosticTechniques = [
  'Autism Diagnostic Interview-Revised (ADI-R)',
  'Autism Diagnostic Observation Schedule-Generic (ADOS-G)',
  'Autism Diagnostic Observation Schedule (second edition) ADOS-2',
  'Developmental, Dimensional and Diagnostic Interview (3DI)',
  'Childhood Autism Rating Scale (CARS)',
  "I don't know",
  'Others',
];
