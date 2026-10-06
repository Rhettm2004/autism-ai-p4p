import httpx

from app.errors import ServiceError
from app.schemas import ScreeningPredictionRequest, ScreeningPredictionResponse


class EaipDarvClient:
    """Typed HTTP client for the separately deployed EAIP-DARV classifier."""

    def __init__(self, base_url: str, timeout_seconds: float = 30, client=None):
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def ready(self) -> bool:
        try:
            response = await self.client.get(f'{self.base_url}/health')
            if response.status_code != 200:
                return False
            body = response.json()
            return body.get('status') == 'ok' and body.get('model_loaded') is True
        except (httpx.HTTPError, ValueError):
            return False

    async def schema(self):
        try:
            response = await self.client.get(f'{self.base_url}/schema')
            response.raise_for_status()
            body = response.json()
            keys = body.get('expected_raw_columns')
            if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
                raise ValueError('Invalid schema')
            return {'expected_raw_columns': keys}
        except (httpx.HTTPError, ValueError) as exc:
            raise ServiceError('screening_schema_unavailable', 'The EAIP input schema is unavailable.', 503) from exc

    async def predict(self, request: ScreeningPredictionRequest) -> ScreeningPredictionResponse:
        schema = await self.schema()
        features = request.features.model_dump(by_alias=True)
        if set(features) != set(schema['expected_raw_columns']):
            raise ServiceError('screening_schema_mismatch', 'The model input schema has changed. Update the app before screening.', 409, False)
        try:
            response = await self.client.post(
                f'{self.base_url}/predict',
                json={'features': request.features.model_dump(by_alias=True)},
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(
                'screening_model_timeout',
                'The EAIP-DARV screening model took too long to respond.',
                504,
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                'screening_model_unavailable',
                'The EAIP-DARV screening model is unavailable.',
                503,
            ) from exc

        if response.status_code != 200:
            raise ServiceError(
                'screening_prediction_failed',
                'The EAIP-DARV model could not calculate this screening result.',
                502,
                response.status_code >= 500,
            )
        try:
            body = response.json()
            return ScreeningPredictionResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                **body,
            )
        except (ValueError, TypeError) as exc:
            raise ServiceError(
                'invalid_screening_model_response',
                'The EAIP-DARV model returned an invalid response.',
                502,
                False,
            ) from exc

    async def close(self):
        if self._owns_client:
            await self.client.aclose()
