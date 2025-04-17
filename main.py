import argparse
import os

from src.config import Config
from src.my_logging import get_logger, setup_logging
from src.preprocessing import preprocess_data
from src.data_handler import DataHandler
from typing import List


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SARI Data Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (YAML or JSON)",
    )
    parser.add_argument("--data-path", type=str, help="Override data path from config")
    parser.add_argument(
        "--output-path", type=str, help="Override output path from config"
    )
    parser.add_argument(
        "--deduplicate-threshold",
        type=float,
        help="Override deduplication threshold from config",
    )
    return parser.parse_args()


def main():
    """Main entry point for the SARI Data Pipeline."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    try:
        config = Config.from_file(args.config)
        logger = get_logger(__name__)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger = get_logger(__name__)
        logger.warning(
            f"Config file {args.config} not found, using default configuration"
        )
        config = Config()

    # Override config with command line arguments if provided
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.deduplicate_threshold:
        config["deduplicate_threshold"] = args.deduplicate_threshold

    # Setup logging
    setup_logging(log_file=config["log_file"])
    logger = get_logger(__name__)
    logger.info("=" * 100)
    logger.info("Starting SARI Data Pipeline")

    # Ensure output directory exists
    output_dir = os.path.dirname(config["output_path"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Run preprocessing
    logger.info("Running preprocessing pipeline")
    preprocess_data(
        data_path=config["data_path"],
        output_path=config["output_path"],
        deduplicate_threshold=config["deduplicate_threshold"],
        deduplicate_ngram_n=config["deduplicate_ngram_n"],
        num_perm=config["num_perm"],
    )

    logger.info("SARI Data Pipeline completed successfully")
    logger.info("=" * 100)


def combine_data(data_paths: List[str], output_path: str):
    data_handler = DataHandler()
    for data_path in data_paths:
        data = data_handler.load_jsonl(data_path)
        data_handler.save_to_jsonl(data, output_path, append=True)


# combine_data(["../example_data/example_data.jsonl", "../example_data/example_data_2.jsonl"], "../example_data/combined_data.jsonl")


if __name__ == "__main__":
    # main()
    combine_data(
        [
            "./example_data/example_math_problem_processed_lsh_duplicated.jsonl",
            "./example_data/example_math_problem_processed_non_english.jsonl",
        ],
        "./output_combined.jsonl",
    )
