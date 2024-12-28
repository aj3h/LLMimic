import logging
import os
from datetime import datetime
from colorama import Fore, Style

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        log_message = f"[{color}{record.levelname}{Style.RESET_ALL}] [{record.name}] [{record.funcName}] - {record.getMessage()}"
        return color + log_message

def setup_logger(name: str = 'my_package', level: int = logging.DEBUG, logs_dir: str = './logs'):
    """
    Creates a shared logger with file and console handlers for use across the package.
    
    :param name: Name of the logger (default is 'my_package').
    :param level: Logging level (default is logging.DEBUG).
    :param logs_dir: Directory to store log files (default is './logs').
    :return: Configured logger instance.
    """
    logs_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_file = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger