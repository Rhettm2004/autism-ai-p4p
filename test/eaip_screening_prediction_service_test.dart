import 'dart:convert';

import 'package:autism_ai/data/question_banks.dart';
import 'package:autism_ai/models/screening_models.dart';
import 'package:autism_ai/models/screening_session.dart';
import 'package:autism_ai/services/eaip_screening_prediction_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test(
    'sends complete encoded fields and parses the EAIP-DARV response',
    () async {
      late Map<String, dynamic> requestBody;
      final client = MockClient((request) async {
        if (request.method == 'GET') {
          return http.Response(
            jsonEncode({
              'expected_raw_columns': [
                for (var i = 1; i <= 10; i++) 'Q$i',
                'Age',
                'Sex',
                'Ethnicity',
                'Jauntice',
                'FamilyASDHistory',
                'AutismAgeCategory',
              ],
            }),
            200,
          );
        }
        expect(
          request.url.toString(),
          'http://localhost:8000/screening/predict',
        );
        requestBody = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode({
            'api_version': 1,
            'request_id': requestBody['request_id'],
            'session_id': 'session-1',
            'model': 'eaip-darv',
            'eaip_probability': 0.71,
            'darv_probability': 0.73,
            'classification_eaip': 1,
            'classification_darv_fixed': 1,
            'classification_darv_tuned': 1,
            'disagreement': 0.08,
            'confidence_pi': 0.92,
            'per_module_raw_probability': {
              'M1_screening': 0.7,
              'M2_ASSL': 0.8,
              'M3_cluster': 0.75,
            },
            'per_module_calibrated_probability': {
              'M1_screening': 0.68,
              'M2_ASSL': 0.79,
              'M3_cluster': 0.74,
            },
            'agreement_scores': {
              'M1_screening': 0.9,
              'M2_ASSL': 0.88,
              'M3_cluster': 0.91,
            },
            'thresholds_used': {
              'eaip': 0.5,
              'darv_fixed': 0.5,
              'darv_tuned': 0.52,
            },
          }),
          200,
        );
      });
      final service = EaipScreeningPredictionService(
        baseUrl: 'http://localhost:8000/',
        client: client,
      );
      addTearDown(client.close);
      final respondent = RespondentDetails()
        ..isToddler = false
        ..age = 9
        ..gender = 'Female'
        ..ethnicity = 'Asian';
      final background = BackgroundDetails()
        ..jaundice = false
        ..familyAutismHistory = true
        ..completedBy = 'Family Member';
      final questions = questionBanks[QuestionnaireType.aq10Child]!;
      final answers = {
        for (final question in questions) question.id: question.options.first,
      };

      final result = await service.predict(
        sessionId: 'session-1',
        questionnaireType: QuestionnaireType.aq10Child,
        respondent: respondent,
        background: background,
        answers: answers,
      );

      final features = requestBody['features'] as Map<String, dynamic>;
      expect(features.keys.where((key) => key.startsWith('Q')), hasLength(10));
      expect(features['Age'], 9);
      expect(features['Sex'], 'f');
      expect(features['AutismAgeCategory'], 'child');
      expect(result.isMock, isFalse);
      expect(result.traitsDetected, isTrue);
      expect(result.similarityPercentage, 73);
      expect(result.disagreement, 0.08);
      expect(result.confidence, 0.92);
      expect(result.tunedThreshold, 0.52);
      expect(result.submission['features'], features);
      expect(result.modelResponse['eaip_probability'], 0.71);
      final restored = ScreeningSession.fromJson({
        ...result.submission['session_snapshot'] as Map<String, dynamic>,
        'result': {
          'traitsDetected': result.traitsDetected,
          'similarityPercentage': result.similarityPercentage,
          'isMock': false,
          'submission': result.submission,
          'modelResponse': result.modelResponse,
        },
      });
      expect(restored.result!.submission['features'], features);
      expect(restored.result!.modelResponse['classification_darv_fixed'], 1);
      respondent.age = 80;
      expect((result.submission['features'] as Map)['Age'], 9);
    },
  );

  test(
    'reports an unavailable model without returning a mock result',
    () async {
      final client = MockClient(
        (request) async => http.Response(
          jsonEncode({
            'error': {
              'message': 'The EAIP-DARV screening model is unavailable.',
            },
          }),
          503,
        ),
      );
      final service = EaipScreeningPredictionService(
        baseUrl: 'http://localhost:8000',
        client: client,
      );
      addTearDown(client.close);
      final respondent = RespondentDetails()
        ..isToddler = false
        ..age = 20
        ..gender = 'Male'
        ..ethnicity = 'Mixed';
      final background = BackgroundDetails()
        ..jaundice = false
        ..familyAutismHistory = false;
      final questions = questionBanks[QuestionnaireType.aq10Adult]!;

      await expectLater(
        service.predict(
          sessionId: 'session-1',
          questionnaireType: QuestionnaireType.aq10Adult,
          respondent: respondent,
          background: background,
          answers: {
            for (final question in questions)
              question.id: question.options.first,
          },
        ),
        throwsA(isA<ScreeningPredictionException>()),
      );
    },
  );
}
