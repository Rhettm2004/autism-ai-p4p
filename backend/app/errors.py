class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 503, retryable: bool = True):
        super().__init__(message)
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
