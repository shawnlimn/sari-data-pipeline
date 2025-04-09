import json
from typing import List, Type, TypeVar, Union

import pandas as pd

from src.data_schema import DataEntry

T = TypeVar("T")


class DataHandler:
    """A utility class for loading and saving DataEntry objects in various formats (parquet, json, jsonl)."""

    @staticmethod
    def save_to_parquet(data: List[DataEntry], path: str):
        """
        Save a list of DataEntry objects to a parquet file.

        Args:
            data: List of DataEntry objects to save
            path: The file path where the parquet file will be saved
        """
        # Convert DataEntry objects to dictionaries and create DataFrame
        df = pd.DataFrame([entry.model_dump() for entry in data])
        df.to_parquet(path, index=False)

    @staticmethod
    def load_parquet(path: str) -> List[DataEntry]:
        """
        Load a parquet file into a list of DataEntry objects.

        Args:
            path: The file path of the parquet file to load

        Returns:
            List of DataEntry objects containing the data from the parquet file
        """
        df = pd.read_parquet(path)
        return [DataEntry(**row) for row in df.to_dict("records")]

    @staticmethod
    def save_to_json(data: Union[DataEntry, List[DataEntry]], path: str):
        """
        Save DataEntry object(s) to a JSON file with proper formatting.

        Args:
            data: Either a single DataEntry object or a list of DataEntry objects
            path: The file path where the JSON file will be saved
        """
        if isinstance(data, list):
            data = [entry.model_dump() for entry in data]
        else:
            data = data.model_dump()

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(path: str) -> Union[DataEntry, List[DataEntry]]:
        """
        Load a JSON file into DataEntry object(s).

        Args:
            path: The file path of the JSON file to load

        Returns:
            Either a single DataEntry object or a list of DataEntry objects
        """
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [DataEntry(**item) for item in data]
        return DataEntry(**data)

    @staticmethod
    def save_to_jsonl(data: List[DataEntry], path: str, append: bool = False):
        """
        Save a list of DataEntry objects to a JSONL file.

        Args:
            data: List of DataEntry objects to save
            path: The file path where the JSONL file will be saved
            append: If True, appends to the file instead of overwriting
        """
        mode = "a" if append else "w"
        with open(path, mode) as f:
            for entry in data:
                f.write(json.dumps(entry.model_dump()) + "\n")

    @staticmethod
    def load_jsonl(path: str, model_class: Type[T] = DataEntry) -> List[T]:
        """
        Load a JSONL file and parse each line into the specified Pydantic model.

        Args:
            path: Path to the JSONL file
            model_class: Pydantic model class to parse each line into (defaults to DataEntry)

        Returns:
            List of parsed Pydantic model instances

        Raises:
            ValidationError: If any line in the JSONL file doesn't match the specified Pydantic model schema
        """
        items = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    items.append(model_class(**data))
        return items
