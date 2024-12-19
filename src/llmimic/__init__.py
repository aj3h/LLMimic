from .logger import setup_logger
from .user_data import UserData

# Set up the logger for the entire package
logger = setup_logger(name='LLMimic')

# Allow submodules to import the logger
__all__ = ['logger']

import time

class ExecutionTimer:
  def __init__(self):
    self.start_time = None

  def start(self):
    self.start_time = time.time()

  def stop(self) -> str:
    if self.start_time is None:
        raise ValueError("Timer was not started.")
    elapsed_time = time.time() - self.start_time
    self.start_time = None
    minutes, seconds = divmod(int(elapsed_time), 60)
    return f"{minutes}m{seconds}s"

from .llm_instance import LLMInstance