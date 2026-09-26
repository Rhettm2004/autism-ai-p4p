import '../models/chat_reply.dart';
import '../models/screening_models.dart';
import 'chat_service.dart';

class MockChatService extends ChatService {
  @override
  String get displayName => 'Mock';

  @override
  Future<ChatReply> sendMessage({
    required String message,
    required List<ChatMessage> history,
    required ScreeningContext context,
  }) async {
    final normalized = message.toLowerCase();

    if (context.stage == ScreeningStage.welcome &&
        _isStartRequest(normalized, history)) {
      return ChatReply(
        'Of course. Let’s begin with a few details to select the appropriate questionnaire.',
        route: 'screening_guidance',
        model: 'mock',
        action: ChatAction(
          type: ChatActionType.startScreening,
          expectedContextRevision: context.revision,
        ),
      );
    }

    if (normalized.contains('question')) {
      if (context.currentQuestionText != null) {
        return ChatReply(
          'I can help clarify the current screening question, but I cannot choose an answer for you.',
        );
      }
      return ChatReply(
        'I can help explain how the current screening step works.',
      );
    }
    if (normalized.contains('result') || normalized.contains('mean')) {
      return ChatReply(
        'This is a mock screening result, not a diagnosis. A fuller result explanation will be connected in a later phase.',
      );
    }
    if (normalized.contains('next') || normalized.contains('do now')) {
      return ChatReply(
        'If you have concerns, consider discussing the screening with a qualified health professional.',
      );
    }
    if (normalized.contains('age')) {
      return ChatReply(
        'Age is used locally to select the appropriate questionnaire for this prototype.',
      );
    }
    return ChatReply(
      'The conversational assistant will be connected in a later phase. For now, I can provide simple guidance about this screening flow.',
    );
  }

  bool _isStartRequest(String message, List<ChatMessage> history) {
    final text = message.toLowerCase().trim().replaceAll(RegExp(r'[.!?]'), '');
    if (text == '/start') {
      return true;
    }
    if (const [
      "don't start",
      'do not start',
      'not ready',
      'no screening',
      "don't want",
      'do not want',
    ].any(text.contains)) {
      return false;
    }
    if (RegExp(
          r'\b(start|begin|take|do|complete)\b.{0,24}\b(screening|questionnaire|test)\b|\b(screening|questionnaire|test)\b.{0,24}\b(start|begin|take|do|complete)\b',
        ).hasMatch(text) ||
        RegExp(
          r"\b(?:i am|i['’]?m|im|we are|we['’]?re)?\s*ready\b.{0,32}\b(?:start|begin|screening|questionnaire|test|get\s+started)\b|\blet['’]?s\s+(?:start|get\s+started)\b",
        ).hasMatch(text)) {
      return true;
    }
    if (!const {
      'yes',
      'yes please',
      'yeah',
      'yep',
      'sure',
      'okay',
      'ok',
      "let's start",
      'lets start',
      "let's get started",
      'lets get started',
      'ready',
      "i'm ready",
      'im ready',
      'i am ready',
    }.contains(text)) {
      return false;
    }
    final assistantHistory = history
        .where((entry) => !entry.isUser)
        .map((entry) => entry.text.toLowerCase())
        .join(' ');
    return const [
      'start a screening',
      'start screening',
      'begin a screening',
      'ready to start',
      'ready to get started',
      'say "yes"',
      "say 'yes'",
    ].any(assistantHistory.contains);
  }
}

abstract class ScreeningPredictionService {
  Future<ScreeningResult> predict({
    required RespondentDetails respondent,
    required BackgroundDetails background,
    required Map<String, String> answers,
  });
}

class MockScreeningPredictionService implements ScreeningPredictionService {
  @override
  Future<ScreeningResult> predict({
    required RespondentDetails respondent,
    required BackgroundDetails background,
    required Map<String, String> answers,
  }) async {
    // Deliberately deterministic mock data. This is not a questionnaire score,
    // CNN output, clinical conclusion, or production prediction.
    return const ScreeningResult(
      traitsDetected: false,
      similarityPercentage: 24,
      isMock: true,
    );
  }
}
