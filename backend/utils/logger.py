import sys
import threading
from io import StringIO
import traceback

class ThreadLogger(StringIO):
    """
    Thread-safe logger that intercepts stdout/stderr and forwards it to a callback
    while acts as a normal file-like object.
    
    This is useful to capture 'print' statements from third-party libraries (like sklearn/xgboost)
    or legacy code executing in the training thread.
    """
    def __init__(self, callback, original_stream):
        super().__init__()
        self.callback = callback
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, message):
        # Write to original stdout so it still appears in terminal
        self.original_stream.write(message)
        
        # Buffer mechanism to handle partial writes (e.g. print("a", end=""))
        self.buffer += message
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Process all complete lines
            for line in lines[:-1]:
                if line.strip(): # Don't log empty lines
                    self.callback(line.strip())
            # Keep the incomplete part
            self.buffer = lines[-1]

    def flush(self):
        self.original_stream.flush()

class TrainingContext:
    """
    Context manager to redirect stdout/stderr for a specific thread or globally 
    during the execution of a block.
    """
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.logger = None

    def __enter__(self):
        # Create interceptors
        self.logger_out = ThreadLogger(lambda msg: self.log_callback("info", msg), self.original_stdout)
        self.logger_err = ThreadLogger(lambda msg: self.log_callback("error", msg), self.original_stderr)
        
        # Redirect
        sys.stdout = self.logger_out
        sys.stderr = self.logger_err
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Log unhandled exceptions if any
        if exc_type:
            error_msg = str(exc_val)
            # We don't suppress the exception, just log it
            self.log_callback("error", f"Uncaught Exception: {error_msg}")
