import sys
import io
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal


class OutputCapture(io.StringIO):
    """Class for capturing stdout/stderr during training."""
    def __init__(self, callback, original_stream):
        super().__init__()
        self.callback = callback
        self.original_stream = original_stream
    
    def write(self, text):
        super().write(text)
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()
        if self.callback:
            self.callback(text)
        return len(text)
    
    def flush(self):
        super().flush()
        if self.original_stream:
            self.original_stream.flush()


class TrainingThread(QThread):
    """ Thread for running training in the background."""
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, arch_instance, X_train, y_train):
        super().__init__()
        self.arch_instance = arch_instance
        self.X_train = X_train
        self.y_train = y_train
        self.should_stop = False
    
    def run(self):
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            stdout_capture = OutputCapture(lambda text: self.output_signal.emit(text), old_stdout)
            stderr_capture = OutputCapture(lambda text: self.output_signal.emit(f"[stderr]: {text}"), old_stderr)
            
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            start_time = time.time()
            self.output_signal.emit(f"Starting training at {time.strftime('%H:%M:%S')}\n")
            
            if self.should_stop:
                self.finished_signal.emit(False, "Training stopped before starting")
                return
            
            self.arch_instance.train(self.X_train, self.y_train)
            duration = time.time() - start_time
            self.finished_signal.emit(True, f"Training completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.output_signal.emit(f"\nERROR: Training failed: {str(e)}\n")
            self.finished_signal.emit(False, f"Training failed: {str(e)}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def stop(self):
        self.should_stop = True
        self.terminate()
