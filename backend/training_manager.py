import threading
import traceback
import logging
from datetime import datetime
import copy
import os

# Configure standard logging to file as well
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrainingManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TrainingManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.state = {
            "status": "idle", # idle, training, completed, failed
            "progress": 0,
            "message": "Ready",
            "logs": [],
            "error": None, # { message, suggestion, distinct_fix }
        }
        self.backup_state = None
        self.stop_flag = False
        self.lock = threading.RLock()

    def start_training(self):
        with self.lock:
            # Backup current state for revert
            self.backup_state = copy.deepcopy(self.state)
            
            self.state["status"] = "training"
            self.state["progress"] = 0
            self.state["message"] = "Starting training..."
            self.state["logs"] = []
            self.state["error"] = None
            self.stop_flag = False
            self.log("info", "Training session started.")

    def log(self, level, message):
        """Logs a message to the in-memory state and file."""
        # Clean the message (remove ANSI codes if any, though frontend handles text)
        msg_str = str(message)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level.upper()}] {msg_str}"
        
        with self.lock:
            self.state["logs"].append(formatted_msg)
            # Keep log size manageable (last 1000 lines)
            if len(self.state["logs"]) > 1000:
                self.state["logs"].pop(0)
        
        # Also log to system file
        if level == 'info':
            logging.info(msg_str)
        elif level == 'error':
            logging.error(msg_str)
        elif level == 'warning':
            logging.warning(msg_str)

    def update_progress(self, progress, message=None):
        with self.lock:
            self.state["progress"] = progress
            if message:
                self.state["message"] = message
                # Don't duplicate log if it's just a progress update, but user asked for "detailed logs"
                # so let's log major progress steps
                self.log("info", f"PROGRESS {progress}%: {message}")

    def set_error(self, exception, context=""):
        """
        Centralized Error Handler.
        Analyzes the exception and sets user-friendly messages and logical fixes.
        """
        if isinstance(exception, InterruptedError):
            # Already handled by cancellation logic usually, but if we get here:
            self.abort_training() 
            return

        error_msg = str(exception)
        tb = traceback.format_exc()
        
        logging.error(f"Error in {context}: {error_msg}\n{tb}")
        
        # Logic to determine suggestion based on error type
        suggestion = "Please check your dataset and try again."
        technical_details = error_msg
        
        ex_type = str(type(exception))
        
        if "KeyError" in ex_type:
            suggestion = "It seems a required column is missing."
            technical_details = "Ensure your CSV has specific columns: 'Component', 'd33 (pC/N)', and 'Tc (°C)'."
        elif "empty" in error_msg.lower() or "0 samples" in error_msg.lower():
            suggestion = "The dataset appears to be empty after filtering."
            technical_details = "Check if your CSV contains valid numerical data for d33/Tc and valid formulas."
        elif "estimat" in error_msg.lower(): # estimator errors
            suggestion = "Model initialization failed."
            technical_details = "This might be a bug in the selected algorithm parameter configuration (e.g. invalid depth). Try 'Auto Select'."
        elif "convert" in error_msg.lower() or "could not convert" in error_msg.lower():
            suggestion = "Data type mismatch."
            technical_details = "Ensure all feature columns contain only numbers or valid chemical formulas. Remove non-numeric text."
        elif "attributeerror" in error_msg.lower():
            suggestion = "Internal Logic Error."
            technical_details = "Please report this to the developer. Retry with 'Auto Mode'."
            
        with self.lock:
            self.state["status"] = "failed"
            self.state["message"] = f"Failed: {error_msg}"
            self.state["error"] = {
                "message": error_msg,
                "suggestion": suggestion,
                "details": technical_details,
                "context": context
            }
            self.log("error", f"FAILURE in {context}: {error_msg}")
            
    def complete_training(self):
        with self.lock:
            self.state["status"] = "completed"
            self.state["progress"] = 100
            self.state["message"] = "Training successfully completed!"
            self.log("info", "Training process finished successfully.")

    def get_state(self):
        with self.lock:
            return self.state.copy()

    def abort_training(self):
        """Signals the training thread to stop and reverts state."""
        with self.lock:
            if self.state["status"] == "idle":
                 return

            self.stop_flag = True
            
            # Revert to backup if available
            if self.backup_state:
                # We revert 'active model info' or 'insights' conceptually, 
                # but 'state' helps UI reset.
                # However, we should show "Cancelled" state briefly so user knows it stopped.
                self.state["status"] = "idle" 
                self.state["message"] = "Training cancelled by user."
                self.state["progress"] = 0
                self.log("warning", "Training cancelled by user.")
                
            else:
                self.state["status"] = "idle"
                self.state["message"] = "Training cancelled."
                
            # Cleanup .tmp files if they exist
            try:
                for f in ["d33_model.pkl.tmp", "Tc_model.pkl.tmp"]:
                    if os.path.exists(os.path.join("saved_models", f)):
                        os.remove(os.path.join("saved_models", f))
            except:
                pass

    def check_interruption(self):
        """Checks if stop flag is raised and raises exception if so."""
        if self.stop_flag:
            raise InterruptedError("Training stopped by user.")

# Global Instance
training_manager = TrainingManager()
