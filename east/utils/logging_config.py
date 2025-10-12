"""
Logging Configuration for EAST-Implement

Provides structured logging with multiple handlers and formatters.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: str = "east-experiment"
) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        experiment_name: Name for experiment-specific logging
    """
    
    # Create logs directory
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'east': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'root': {
                'level': log_level,
                'handlers': ['console']
            }
        }
    }
    
    # Add file handler if log_file provided
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        
        # Add file handler to loggers
        config['loggers']['east']['handlers'].append('file')
        config['loggers']['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger('east.logging')
    logger.info(f"Logging setup complete for experiment: {experiment_name}")
    logger.info(f"Log level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(f"east.{name}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file="logs/east.log",
        experiment_name="test-run"
    )
    
    # Test logging
    logger = get_logger("test")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")