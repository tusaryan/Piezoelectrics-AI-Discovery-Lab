from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from datetime import datetime
from uuid import uuid4
from opentelemetry import trace

class PiezoException(Exception):
    def __init__(self, code: str, message: str, field: str = None, row: int = None, suggestions: list[str] = None):
        self.code = code
        self.message = message
        self.field = field
        self.row = row
        self.suggestions = suggestions or []

def create_envelope(data=None, error=None):
    span = trace.get_current_span()
    trace_id = trace.format_trace_id(span.get_span_context().trace_id) if span.is_recording() else None
    
    return {
        "success": error is None,
        "data": data,
        "error": error,
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "version": "2.0.0"
        }
    }

def register_error_handlers(app: FastAPI):
    @app.exception_handler(PiezoException)
    async def piezo_exception_handler(request: Request, exc: PiezoException):
        error_payload = {
            "code": exc.code,
            "message": exc.message
        }
        if exc.field: error_payload["field"] = exc.field
        if exc.row is not None: error_payload["row"] = exc.row
        if exc.suggestions: error_payload["suggestions"] = exc.suggestions
        
        return JSONResponse(status_code=400, content=create_envelope(error=error_payload))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        first_error = exc.errors()[0]
        field = ".".join(str(x) for x in first_error.get("loc", []))
        
        error_payload = {
            "code": "VALIDATION_ERROR",
            "message": first_error.get("msg", "Invalid input"),
            "field": field
        }
        return JSONResponse(status_code=422, content=create_envelope(error=error_payload))

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # We don't expose stack traces to the client
        error_payload = {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred"
        }
        return JSONResponse(status_code=500, content=create_envelope(error=error_payload))
