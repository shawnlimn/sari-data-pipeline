from src.data_schema import DataEntry
from src.data_utils import get_word_ngrams
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from typing import List, Tuple
from src.my_logging import get_logger
from src.data_utils import normalize_text

# def normalize_data(
#     texts: List[str],
#     remove_stopwords: bool = True,
#     remove_extra_whitespace: bool = True,
#     remove_special_chars: bool = False,
#     normalize_unicode: bool = True,
#     lowercase: bool = True,
# ) -> List[str]:
#     """
#     Normalize the data by removing extra whitespace and converting to lowercase.
#     """
#     logger = get_logger(__name__)
#     logger.info(
#         f"Normalizing the data with remove_stopwords: {remove_stopwords}, remove_extra_whitespace: {remove_extra_whitespace}, remove_special_chars: {remove_special_chars}, normalize_unicode: {normalize_unicode}, lowercase: {lowercase}"
#     )
#     return [
#         normalize_text(
#             t,
#             remove_stopwords,
#             remove_extra_whitespace,
#             remove_special_chars,
#             normalize_unicode,
#             lowercase,
#         )
#         for t in texts
#     ]


def benchmark_decontamination_with_lsh(
    data: List[DataEntry],
    benchmark_texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 256,
    ngram_n: int = 32,
) -> Tuple[List[DataEntry], List[DataEntry]]:
    """
    Decontaminate the dataset using LSH by removing entries that are near-duplicates of benchmark samples.
    """
    logger = get_logger(__name__)
    logger.info(
        f"Starting decontamination against benchmark using LSH (threshold={threshold})"
    )

    # Step 1: Initialize LSH index with benchmark data
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for idx, text in enumerate(tqdm(benchmark_texts, desc="Indexing benchmark data")):
        normalized_text = normalize_text(text)
        ngrams = get_word_ngrams(normalized_text, n=ngram_n)
        if not ngrams:
            continue
        m = MinHash(num_perm=num_perm)
        for ng in ngrams:
            token = " ".join(ng)
            m.update(token.encode("utf8"))
        lsh.insert(f"benchmark_{idx}", m)

    # Step 2: Check and filter your data
    clean_data = []
    contaminated = []

    for idx, entry in tqdm(
        enumerate(data), total=len(data), desc="Checking for contamination"
    ):
        try:
            raw_text = entry.question
            normalized_text = normalize_text(raw_text)
        except (IndexError, TypeError) as e:
            logger.warning(f"Skipping entry {idx} due to malformed sft_data: {e}")
            continue

        ngrams = get_word_ngrams(normalized_text, n=ngram_n)
        if not ngrams:
            continue

        m = MinHash(num_perm=num_perm)
        for ng in ngrams:
            token = " ".join(ng)
            m.update(token.encode("utf8"))

        if any(lsh.query(m)):
            contaminated.append(entry)
        else:
            clean_data.append(entry)

    logger.info(
        f"Total: {len(data)}, Clean: {len(clean_data)}, Contaminated: {len(contaminated)}"
    )
    return clean_data, contaminated
