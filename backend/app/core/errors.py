"""
Custom error classes and exception handlers for SimuStruct API.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse

class SimuStructAPIException(Exception):
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or []

class InvalidGeometryError(SimuStructAPIException):
    def __init__(self, message: str, details: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            message=message,
            code="INVALID_GEOMETRY",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

class OutOfDomainError(SimuStructAPIException):
    def __init__(self, message: str, details: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            message=message,
            code="OUT_OF_DOMAIN_PARAMETER",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )

class ModelNotFoundError(SimuStructAPIException):
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model with identifier '{model_id}' was not found in the registry.",
            code="MODEL_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=[{"model_id": model_id}]
        )

class ModelNotReadyError(SimuStructAPIException):
    def __init__(self, model_id: str, reason: str = "Artifacts not loaded or checkpoint corrupted"):
        super().__init__(
            message=f"Model '{model_id}' is not ready for inference: {reason}",
            code="MODEL_NOT_READY",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=[{"model_id": model_id, "reason": reason}]
        )

class InferenceExecutionError(SimuStructAPIException):
    def __init__(self, message: str, details: Optional[List[Dict[str, Any]]] = None):
        super().__init__(
            message=f"Inference execution failed: {message}",
            code="INFERENCE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

def format_error_response(status_code: int, code: str, message: str, details: List[Dict[str, Any]]) -> JSONResponse:
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    return JSONResponse(status_code=status_code, content=payload)

async def simustruct_exception_handler(request: Request, exc: SimuStructAPIException) -> JSONResponse:
    return format_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log internal error and return sanitized message to user without leaking trace
    return format_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="UNEXPECTED_ERROR",
        message="An unexpected internal error occurred during request processing.",
        details=[{"type": type(exc).__name__}]
    )
