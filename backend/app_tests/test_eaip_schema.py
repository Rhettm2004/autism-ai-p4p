import asyncio
import httpx
import pytest
from app.adapters.eaip_darv import EaipDarvClient
from app.errors import ServiceError
from app.schemas import ScreeningPredictionRequest

FEATURES = {**{f'Q{i}': 0 for i in range(1, 11)}, 'Age': 24, 'Sex': 'm', 'Ethnicity': 'Asian', 'Jauntice': 'no', 'FamilyASDHistory': 'no', 'AutismAgeCategory': 'chat'}

def test_schema_change_blocks_prediction_before_post():
    async def run():
        calls = []
        def handle(request):
            calls.append(request.method)
            return httpx.Response(200, json={'expected_raw_columns': [*FEATURES, 'new_required_field']})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = EaipDarvClient('http://eaip', client=http)
            request = ScreeningPredictionRequest(request_id='test-1', session_id='session-1', features=FEATURES)
            with pytest.raises(ServiceError) as error:
                await client.predict(request)
            assert error.value.code == 'screening_schema_mismatch'
            assert calls == ['GET']
    asyncio.run(run())

@pytest.mark.parametrize('body', [{}, {'expected_raw_columns': []}, {'expected_raw_columns': [1]}])
def test_malformed_schema_is_unavailable(body):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as http:
            with pytest.raises(ServiceError) as error:
                await EaipDarvClient('http://eaip', client=http).schema()
            assert error.value.status == 503
    asyncio.run(run())
