import '../models/screening_models.dart';

abstract class ChatService {
  Future<String> sendMessage(String message, ScreeningContext context);
}

class MockChatService implements ChatService {
  @override
  Future<String> sendMessage(String message, ScreeningContext context) async {
    final normalized = message.toLowerCase();

    if (normalized.contains('question')) {
      if (context.currentQuestionText != null) {
        return 'I can help clarify the current screening question, but I cannot choose an answer for you.';
      }
      return 'I can help explain how the current screening step works.';
    }
    if (normalized.contains('result') || normalized.contains('mean')) {
      return 'This is a mock screening result, not a diagnosis. A fuller result explanation will be connected in a later phase.';
    }
    if (normalized.contains('next') || normalized.contains('do now')) {
      return 'If you have concerns, consider discussing the screening with a qualified health professional.';
    }
    if (normalized.contains('age')) {
      return 'Age is used locally to select the appropriate questionnaire for this prototype.';
    }
    return 'The conversational assistant will be connected in a later phase. For now, I can provide simple guidance about this screening flow.';
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
