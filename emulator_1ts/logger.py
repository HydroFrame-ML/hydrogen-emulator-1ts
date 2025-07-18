import sys
import datetime
from enum import Enum, auto


class LogLevel(Enum):
    """Enum for log levels"""
    SILENT = auto()  # No logging (default)
    INFO = auto()    # Minimal logging
    VERBOSE = auto() # Detailed logging


class Logger:
    """
    A simple logger for the emulator project with three modes:
    - Silent (default): No logging
    - Info: Minimal logging
    - Verbose: Detailed logging
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one logger instance exists"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._level = LogLevel.SILENT  # Default to silent
        return cls._instance
    
    @property
    def level(self):
        """Get the current log level"""
        return self._level
    
    @level.setter
    def level(self, level):
        """Set the log level"""
        if isinstance(level, LogLevel):
            self._level = level
        elif isinstance(level, str):
            level = level.upper()
            if level == "SILENT":
                self._level = LogLevel.SILENT
            elif level == "INFO":
                self._level = LogLevel.INFO
            elif level == "VERBOSE":
                self._level = LogLevel.VERBOSE
            else:
                raise ValueError(f"Invalid log level: {level}. Must be one of: SILENT, INFO, VERBOSE")
        else:
            raise TypeError("Log level must be a LogLevel enum or string")
    
    def _log(self, message, level, file=sys.stdout):
        """Internal logging method"""
        if self._level.value >= level.value:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level.name}] {message}", file=file)
    
    def info(self, message):
        """Log an info message (shown in INFO and VERBOSE modes)"""
        self._log(message, LogLevel.INFO)
    
    def verbose(self, message):
        """Log a verbose message (shown only in VERBOSE mode)"""
        self._log(message, LogLevel.VERBOSE)
    
    def error(self, message):
        """Log an error message (shown in all modes except SILENT)"""
        self._log(message, LogLevel.INFO, file=sys.stderr)


# Create a singleton instance
logger = Logger()


def set_log_level(level):
    """Set the log level for the logger"""
    logger.level = level


def get_log_level():
    """Get the current log level"""
    return logger.level


def info(message):
    """Log an info message"""
    logger.info(message)


def verbose(message):
    """Log a verbose message"""
    logger.verbose(message)


def error(message):
    """Log an error message"""
    logger.error(message)
