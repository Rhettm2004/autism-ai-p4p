import '../config/app_config.dart';
import 'eaip_screening_prediction_service.dart';
import 'mock_services.dart';

ScreeningPredictionService createConfiguredPredictionService() {
  return switch (AppConfig.predictionProvider) {
    'backend' => EaipScreeningPredictionService(
      baseUrl: AppConfig.backendBaseUrl,
    ),
    'mock' => MockScreeningPredictionService(),
    _ => throw StateError(
      'Unsupported SCREENING_PREDICTION_PROVIDER: '
      '${AppConfig.predictionProvider}',
    ),
  };
}
