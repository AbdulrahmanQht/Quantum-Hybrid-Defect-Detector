import os
import sys
import uuid
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

class Logger:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, "logs")

        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        current_date = datetime.now().strftime("%d-%m-%Y")
        log_file = os.path.join(self.log_dir, f"{current_date}.txt")

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler that rotates logs daily
        file_handler = TimedRotatingFileHandler(log_file, when = 'midnight', interval = 1, backupCount = 30)

        formatter = logging.Formatter(
            fmt='-' * 75 + '\n' + '%(asctime)s - %(levelname)s - %(message)s\n' ,
            datefmt='%d/%m/%Y - %I:%M %p'
        )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_results(self, model_name, latency, confidence, result):
        """Logs specific performance data for the Comparative Analysis"""
        rid = str(uuid.uuid4())
        data = f"ID: {rid}\nMODEL: {model_name} | Latency: {latency}ms | Confidence: {confidence}% | Result: {result}"
        self.logger.info(data)

    def info(self, msg):
        rid = str(uuid.uuid4())
        self.logger.info(f"ID: {rid}\n{msg}")

    def warn(self, msg):
        rid = str(uuid.uuid4())
        self.logger.warning(f"ID: {rid}\n{msg}")

    def error(self, msg, include_stacktrace=True):
        # Detect if there is an exception
        has_active_exception = sys.exc_info()[0] is not None

        rid = str(uuid.uuid4())
        # If there is no exception it doesn't print it to txt file
        self.logger.error(f"ID: {rid}\n{msg}", exc_info=include_stacktrace and has_active_exception)