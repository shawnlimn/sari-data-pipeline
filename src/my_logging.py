import logging
import os


def setup_logging(
    log_level: str = "INFO", log_file: str = "logs/sari_data_house.log"
) -> None:
    """Set up logging configuration with error handling and level validation"""

    try:
        # Validate log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Clear any existing handlers if logging is already configured
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up logging
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),  # This keeps console output
            ],
        )
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)
