from typing import List, Tuple
import json
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from src.data_handler import DataHandler
from src.data_schema import DataEntry
from src.data_utils import create_minhash, get_word_ngrams, normalize_text
from src.my_logging import get_logger
from lingua import LanguageDetectorBuilder


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
            result.append(entry)
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
    logger = get_logger(__name__)
    logger.info("Removing all the coding questions...")

    # Step 1: Normalize the data.
    results = []
    for entry in data:
        if entry.metadata.main_category != "coding":
            results.append(entry)

    return results


def deduplicate_data_with_exact_match(data: List[DataEntry]) -> List[DataEntry]:
    """
    Deduplicate the data by exact match.
    """
    logger = get_logger(__name__)
    logger.info("Deduplication with exact match...")
    # Step 1: Normalize the data.
    texts = [entry.question + " " + entry.sft_data[0][1] for entry in data]
    normalized_texts = normalize_data(texts)

    # Step 2: Deduplicate the data.
    seen = set()
    result = []
    duplicated_entries = []
    for i, norm_text in enumerate(normalized_texts):
        if norm_text not in seen:
            seen.add(norm_text)
            result.append(data[i])
        else:
            duplicated_entries.append(data[i])
    return result, duplicated_entries


def remove_non_english_questions(
    data: List[DataEntry],
) -> Tuple[List[DataEntry], List[DataEntry], float, float]:
    detector = LanguageDetectorBuilder.from_all_languages().build()

    total = len(data)
    chinese_count = 0
    non_english_count = 0
    result = []
    non_english_data = []
    for entry in data:
        text = (
            entry.question
            + " "
            + (
                entry.sft_data[0][1]
                if entry.sft_data and len(entry.sft_data[0]) > 1
                else ""
            )
        )
        lang = detector.detect_language_of(text)

        if lang.name == "CHINESE":
            chinese_count += 1
            non_english_data.append(entry)
        elif lang.name != "ENGLISH":
            non_english_count += 1
            non_english_data.append(entry)
        else:
            print(entry.question)
            result.append(entry)

    chinese_percent = (chinese_count / total) * 100 if total else 0
    non_english_percent = (non_english_count / total) * 100 if total else 0

    return result, non_english_data, chinese_percent, non_english_percent


def benchmark_decontamination(
    data: List[DataEntry],
    threshold: float = 0.8,
    num_perm: int = 256,
    ngram_n: int = 32,
) -> List[DataEntry]:
    """
    Decontaminate the data by removing the questions that are in the benchmark data.
    """
    logger = get_logger(__name__)
    logger.info("Decontamination...")

    # load benchmark data
    with open("./benchmarks/benchmark_data.json", "r") as f:
        benchmark_data = json.load(f)
    benchmark_data = [item for sublist in benchmark_data.values() for item in sublist]

    normalized_benchmark_data = normalize_data(benchmark_data)

    # decontaminate the data
    from src.decontamination import benchmark_decontamination_with_lsh

    result, contaminated = benchmark_decontamination_with_lsh(
        data,
        normalized_benchmark_data,
        threshold=threshold,
        num_perm=num_perm,
        ngram_n=ngram_n,
    )

    return result, contaminated


def preprocess_data(
    data_path: str,
    output_path: str,
    deduplicate_threshold: float = 0.8,
    deduplicate_ngram_n: int = 3,
    num_perm: int = 256,
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

    logger.info("Deduplication...")
    # Step 1: Deduplication with exact match (question + CoT).
    logger.info(f"Deduplication with exact match for {len(data)} entries...")
    data, duplicated_entries = deduplicate_data_with_exact_match(data)
    logger.info(f"Number of data entries after exact match deduplication: {len(data)}")
    logger.info(f"Number of duplicated entries: {len(duplicated_entries)}")

    # Step 1.1: Save the duplicated entries.
    logger.info("Saving the duplicated entries...")
    data_helper.save_to_jsonl(
        duplicated_entries,
        output_path.replace(".jsonl", "_exact_match_duplicated.jsonl"),
    )
    logger.info("Duplicated entries saved successfully.")

    # Step 2: Deduplication with LSH (question + CoT).
    logger.info(f"Deduplication with LSH for {len(data)} entries...")
    data, duplicated_entries = deduplicate_data_with_lsh(
        data,
        threshold=deduplicate_threshold,
        num_perm=num_perm,
        ngram_n=deduplicate_ngram_n,
    )
    logger.info(f"Number of data entries after LSH deduplication: {len(data)}")
    logger.info(f"Number of duplicated entries: {len(duplicated_entries)}")

    # Step 2.1: Save the duplicated entries.
    logger.info("Saving the duplicated entries...")
    data_helper.save_to_jsonl(
        duplicated_entries, output_path.replace(".jsonl", "_lsh_duplicated.jsonl")
    )
    logger.info("Duplicated entries saved successfully.")

    # Step 2.2: Save the deduplicated data.
    logger.info("Saving the deduplicated data...")
    data_helper.save_to_jsonl(
        data, output_path.replace(".jsonl", "_deduplicated.jsonl")
    )
    logger.info("Deduplicated data saved successfully.")

    # Step 3: Remove non-English questions.
    logger.info("Removing non-English questions...")
    data, non_english_data, chinese_percent, non_english_percent = (
        remove_non_english_questions(data)
    )
    logger.info(
        f"Number of data entries after removing non-English questions: {len(data)}"
    )
    logger.info(f"Number of non-English questions: {len(non_english_data)}")
    logger.info(f"Chinese percentage: {chinese_percent}%")
    logger.info(f"Non-English percentage: {non_english_percent}%")

    # Step 3.1: Save the non-English questions.
    logger.info("Saving the non-English questions...")
    data_helper.save_to_jsonl(
        non_english_data, output_path.replace(".jsonl", "_non_english.jsonl")
    )
    logger.info("Non-English questions saved successfully.")

    # Step 3.2: Save the processed data.
    logger.info("Saving the data after removing non-English questions...")
    data_helper.save_to_jsonl(
        data, output_path.replace(".jsonl", "_non_english_removed.jsonl")
    )
    logger.info("Data after removing non-English questions saved successfully.")

    # # Step 3: Remove all the coding questions first.
    # logger.info("Removing all the coding questions...")
    # data = remove_coding_questions(data)
    # logger.info(f"Number of data entries after removing coding questions: {len(data)}")

    # Step 4: Decontamination.
    logger.info("Decontamination...")
    data, contaminated = benchmark_decontamination(
        data,
        threshold=deduplicate_threshold,
        num_perm=num_perm,
        ngram_n=deduplicate_ngram_n,
    )
    logger.info(f"Number of data entries after decontamination: {len(data)}")
    logger.info(f"Number of contaminated entries: {len(contaminated)}")

    # Step 4.1: Save the contaminated entries.
    logger.info("Saving the contaminated entries...")
    data_helper.save_to_jsonl(
        contaminated, output_path.replace(".jsonl", "_contaminated.jsonl")
    )
    logger.info("Contaminated entries saved successfully.")

    # Step 4.2: Save the decontaminated data.
    logger.info("Saving the decontaminated data...")
    data_helper.save_to_jsonl(
        data, output_path.replace(".jsonl", "_decontaminated.jsonl")
    )
    logger.info("Decontaminated data saved successfully.")

    # Step N: Save the processed data.
    logger.info("Saving the processed data...")
    data_helper.save_to_jsonl(data, output_path)
    logger.info("Data saved successfully.")
