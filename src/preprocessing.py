from typing import List, Tuple

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from src.data_handler import DataHandler
from src.data_schema import DataEntry
from src.data_utils import create_minhash, get_word_ngrams, normalize_text
from src.my_logging import get_logger


def deduplicate_data_with_lsh(
    data: List[DataEntry], threshold: float = 0.8, num_perm: int = 128, ngram_n: int = 2
) -> Tuple[List[DataEntry], List[DataEntry]]:
    logger = get_logger(__name__)
    logger.info(
        f"Deduplication using LSH with threshold: {threshold}, num_perm: {num_perm}, n-gram size: {ngram_n}"
    )

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    result = []
    duplicated_entries = []
    # Deduplicate the data with tqdm.
    for idx, entry in tqdm(enumerate(data), total=len(data), desc="Deduplicating data"):
        try:
            raw_text = entry.question + " " + entry.sft_data[0][1]
        except (IndexError, TypeError) as e:
            logger.warning(f"Skipping entry {idx} due to malformed sft_data: {e}")
            logger.debug(f"Entry: {entry}")
            continue

        ngrams = get_word_ngrams(raw_text, n=ngram_n)
        if not ngrams:
            logger.warning(
                f"Skipping entry {idx} due to insufficient text length for {ngram_n}-grams."
            )
            continue

        m = MinHash(num_perm=num_perm)
        for ng in ngrams:
            token = " ".join(ng)
            m.update(token.encode("utf8"))

        if any(lsh.query(m)):
            duplicated_entries.append(entry)
        else:
            lsh.insert(f"entry_{idx}", m)
            result.append(entry)

    logger.info(f"Original size: {len(data)}, deduplicated size: {len(result)}")

    # duplicate data
    return result, duplicated_entries


def normalize_data(
    texts: List[str],
    remove_stopwords: bool = True,
    remove_extra_whitespace: bool = True,
    remove_special_chars: bool = False,
    normalize_unicode: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """
    Normalize the data by removing extra whitespace and converting to lowercase.
    """
    logger = get_logger(__name__)
    logger.info(
        f"Normalizing the data with remove_stopwords: {remove_stopwords}, remove_extra_whitespace: {remove_extra_whitespace}, remove_special_chars: {remove_special_chars}, normalize_unicode: {normalize_unicode}, lowercase: {lowercase}"
    )
    return [
        normalize_text(
            t,
            remove_stopwords,
            remove_extra_whitespace,
            remove_special_chars,
            normalize_unicode,
            lowercase,
        )
        for t in texts
    ]


def deduplicate_data(data: List[DataEntry], threshold: float = 0.8) -> List[DataEntry]:
    """
    Deduplicate the data by question + CoT.
    """
    # setup logging
    logger = get_logger(__name__)
    logger.info(f"Deduplication with threshold: {threshold}")

    # Step 1: Normalize the data.
    # create minhashes
    texts = [entry.question + " " + entry.sft_data[0][1] for entry in data]
    normalized_texts = normalize_data(texts)
    minhashes = [create_minhash(t) for t in normalized_texts]

    # Check if the lengths of the data and minhashes and texts are the same.
    if len(data) != len(minhashes) or len(data) != len(texts):
        logger.error(
            "The lengths of the data and minhashes and texts are not the same."
        )
        raise ValueError(
            "The lengths of the data and minhashes and texts are not the same."
        )

    seen = set()
    result = []
    for i in range(len(texts)):
        if i in seen:
            continue
        result.append(data[i])
        for j in range(i + 1, len(texts)):
            if minhashes[i].jaccard(minhashes[j]) >= threshold:
                seen.add(j)
    return result


def remove_coding_questions(data: List[DataEntry]) -> List[DataEntry]:
    """
    Remove all the coding questions.
    """
    return data


def preprocess_data(
    data_path: str,
    output_path: str,
    deduplicate_threshold: float = 0.8,
    deduplicate_ngram_n: int = 3,
    log_file: str = "logs/preprocessing.log",
):
    """
    data_path: path to the data file, jsonl format.

    Preprocess the data by removing duplicates and empty questions.
    1. Deduplication (question + CoT → Hashing).
    2. Remove all the coding questions first.
    """
    # setup logging
    # setup_logging(log_file=log_file)
    logger = get_logger(__name__)
    logger.info("Preprocessing data...")

    # Step 0: Initialize the data handler.
    data_helper = DataHandler()
    data = data_helper.load_jsonl(data_path)
    logger.info(f"Number of data entries: {len(data)}")

    # Step 1: Deduplication (question + CoT).
    logger.info("Deduplication...")
    data, duplicated_entries = deduplicate_data_with_lsh(
        data, threshold=deduplicate_threshold, num_perm=128, ngram_n=deduplicate_ngram_n
    )
    logger.info(f"Number of data entries after deduplication: {len(data)}")
    logger.info(f"Number of duplicated entries: {len(duplicated_entries)}")

    # # Step 2: Remove all the coding questions first.
    # logger.info("Removing all the coding questions...")
    # data = remove_coding_questions(data)
    # logger.info(f"Number of data entries after removing coding questions: {len(data)}")

    # Step 3: Save the processed data.
    logger.info("Saving the processed data...")
    data_helper.save_to_jsonl(data, output_path)
    logger.info("Data saved successfully.")

    # Step 4: Save the duplicated entries.
    logger.info("Saving the duplicated entries...")
    data_helper.save_to_jsonl(
        duplicated_entries, output_path.replace(".jsonl", "_duplicated.jsonl")
    )
    logger.info("Duplicated entries saved successfully.")
