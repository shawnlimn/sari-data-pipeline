import re
import unicodedata
from typing import Set, Tuple

from datasketch import MinHash
from nltk.corpus import stopwords


def create_minhash(text, num_perm=128):
    ngrams = get_word_ngrams(text)
    m = MinHash(num_perm=num_perm)
    for gram in ngrams:
        m.update(gram.encode("utf8"))
    return m


def get_word_ngrams(text: str, n: int = 2) -> Set[Tuple[str, ...]]:
    """
    Get word n-grams from a text.
    """
    if not isinstance(text, str):
        print("Input must be a string")  # or logging
    if n <= 0:
        print("n must be a positive integer")  # or logging

    words = text.split()
    if len(words) < n:
        return set()

    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_extra_whitespace: bool = True,
    remove_special_chars: bool = False,
    normalize_unicode: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """Normalize text by applying various text cleaning operations.

    Args:
        text: Input text to normalize
        lowercase: Whether to convert text to lowercase
        remove_extra_whitespace: Whether to remove extra whitespace
        remove_special_chars: Whether to remove special characters
        normalize_unicode: Whether to normalize unicode characters
        remove_stopwords: Whether to remove stopwords
    Returns:
        Normalized text string
    """
    if not text:
        return text

    # Convert to lowercase if configured
    if lowercase:
        text = text.lower()

    # Remove extra whitespace if configured
    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    # Remove special characters if configured
    if remove_special_chars:
        text = re.sub(r"[^\w\s]", "", text)

    # Normalize unicode if configured
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    # Remove stopwords using nltk if configured
    if remove_stopwords:
        try:
            stopwords_list = set(stopwords.words("english"))
        except LookupError:
            import nltk

            nltk.download("stopwords")
            stopwords_list = set(stopwords.words("english"))

        text = " ".join([word for word in text.split() if word not in stopwords_list])

    return text
