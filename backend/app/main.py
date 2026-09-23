from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.errors import ServiceError
from app.schemas import ChatRequest, ChatResponse
from app.settings import Settings
from app.runtime import ChatRuntime

MAX_BODY = 128 * 1024

class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > MAX_BODY:
                return await JSONResponse({'error': {'code': 'request_too_large',
                    'message': 'Request is too large.', 'retryable': False, 'request_id': None}},
                    status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(runtime=None, settings=None):
    settings = settings or Settings.load()
    runtime = runtime or ChatRuntime(settings)

    @asynccontextmanager
    async def lifespan(app):
        await runtime.initialize()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title='Autism AI', version='1', lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'Accept'])

    def error_response(request, code, message, status, retryable):
        return JSONResponse({'error': {'code': code, 'message': message,
            'retryable': retryable, 'request_id': getattr(request.state, 'request_id', None)}}, status_code=status)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(request, exc.code, exc.message, exc.status, exc.retryable)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        # Do not echo invalid health/context input in errors or logs.
        return error_response(request, 'invalid_request', 'Invalid chat request.', 422, False)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc):
        logging.getLogger(__name__).error('Unexpected backend failure: %s', type(exc).__name__)
        return error_response(request, 'internal_error', 'The assistant could not respond.', 500, True)

    @app.get('/health')
    async def health():
        ready, report = await runtime.health()
        return JSONResponse(report, status_code=200 if ready else 503)

    @app.post('/chat', response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request):
        request.state.request_id = body.request_id
        return await runtime.chat(body)

    return app

app = create_app()
