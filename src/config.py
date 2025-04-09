from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration class for the data preprocessing pipeline."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        # Default configuration values
        self.default_config = {
            # Data paths
            "data_path": "example_data/input.jsonl",
            "output_path": "example_data/processed.jsonl",
            "log_file": "logs/preprocessing.log",
            # Deduplication parameters
            "deduplicate_threshold": 0.8,
            "deduplicate_ngram_n": 3,
            "num_perm": 128,
            # Text normalization parameters
            "remove_stopwords": True,
            "remove_extra_whitespace": True,
            "remove_special_chars": False,
            "normalize_unicode": True,
            "lowercase": True,
            # Other preprocessing options
            "remove_coding_questions": False,
        }

        # Update with provided config if any
        self.config = self.default_config.copy()
        if config_dict:
            self.config.update(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to config values."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting of config values."""
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.config.copy()

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from a file."""
        import json

        import yaml

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r") as f:
            if file_path.suffix == ".json":
                config_dict = json.load(f)
            elif file_path.suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")

        return cls(config_dict)

