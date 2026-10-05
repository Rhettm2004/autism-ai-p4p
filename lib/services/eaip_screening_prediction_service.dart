import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/eaip_model_input.dart';
import '../models/screening_models.dart';
import '../models/screening_session.dart';
import 'mock_services.dart';

class ScreeningPredictionException implements Exception {
  const ScreeningPredictionException(this.message);
  final String message;
  @override
  String toString() => message;
}

class EaipScreeningPredictionService implements ScreeningPredictionService {
  EaipScreeningPredictionService({required String baseUrl, http.Client? client})
    : _baseUrl = baseUrl.replaceFirst(RegExp(r'/+$'), ''),
      _client = client ?? http.Client(),
      _ownsClient = client == null;

  final String _baseUrl;
  final http.Client _client;
  final bool _ownsClient;

  @override
  Future<ScreeningResult> predict({
    required String sessionId,
    required QuestionnaireType questionnaireType,
    required RespondentDetails respondent,
    required BackgroundDetails background,
    required Map<String, String> answers,
  }) async {
    final session = ScreeningSession.forPrediction(
      sessionId: sessionId,
      questionnaireType: questionnaireType,
      respondent: respondent,
      background: background,
      behaviouralAnswers: answers,
    );
    final input = EaipModelInputPreview.fromSession(session);
    http.Response response;
    try {
      response = await _client
          .post(
            Uri.parse('$_baseUrl/screening/predict'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({
              'api_version': 1,
              'request_id':
                  'prediction-${DateTime.now().microsecondsSinceEpoch}',
              'session_id': sessionId,
              'features': input.toModelPayload(),
            }),
          )
          .timeout(const Duration(seconds: 35));
    } catch (_) {
      throw const ScreeningPredictionException(
        'The EAIP-DARV screening model is unavailable. Check that the model service is running and try again.',
      );
    }

    Map<String, dynamic> body;
    try {
      body =
          jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    } catch (_) {
      throw const ScreeningPredictionException(
        'The screening service returned an unreadable response.',
      );
    }
    if (response.statusCode != 200) {
      final error = body['error'];
      final message = error is Map ? error['message'] : null;
      throw ScreeningPredictionException(
        message is String && message.isNotEmpty
            ? message
            : 'The EAIP-DARV model could not calculate this result.',
      );
    }
    try {
      final probability = (body['darv_probability'] as num).toDouble();
      return ScreeningResult(
        traitsDetected: body['classification_darv_tuned'] == 1,
        similarityPercentage: probability * 100,
        isMock: false,
        disagreement: (body['disagreement'] as num).toDouble(),
        confidence: (body['confidence_pi'] as num).toDouble(),
        perModuleRawProbability: _numberMap(body['per_module_raw_probability']),
        perModuleCalibratedProbability: _numberMap(
          body['per_module_calibrated_probability'],
        ),
        agreementScores: _numberMap(body['agreement_scores']),
        tunedThreshold: ((body['thresholds_used'] as Map)['darv_tuned'] as num)
            .toDouble(),
      );
    } catch (_) {
      throw const ScreeningPredictionException(
        'The EAIP-DARV model returned an invalid result.',
      );
    }
  }

  static Map<String, double> _numberMap(Object? value) {
    if (value is! Map) return const {};
    return Map.unmodifiable({
      for (final entry in value.entries)
        if (entry.value is num)
          entry.key.toString(): (entry.value as num).toDouble(),
    });
  }

  @override
  void dispose() {
    if (_ownsClient) _client.close();
  }
}
