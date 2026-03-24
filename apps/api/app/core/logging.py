import sys
import logging
import structlog
from opentelemetry import trace

def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Set up OTel trace injection
    def add_trace_id(logger, method_name, event_dict):
        span = trace.get_current_span()
        if span.is_recording():
            event_dict["trace_id"] = trace.format_trace_id(span.get_span_context().trace_id)
            event_dict["span_id"] = trace.format_span_id(span.get_span_context().span_id)
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_trace_id,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = None) -> structlog.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()
