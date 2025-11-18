#!/usr/bin/env python3

"""
This script extracts training text from supervision manifests and creates:
1. transcript_tokens.txt - phoneme sequences (for P.arpa)
2. transcript_words.txt - word sequences (for G.arpa)
"""

import argparse
import json
import gzip
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests-dir",
        type=str,
        help="Input manifests directory, e.g., data/manifests",
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="Output lang directory, e.g., data/lang_phone",
    )
    parser.add_argument(
        "--word-delimiter",
        type=str,
        default=" | ",
        help="Delimiter used to separate words in phoneme sequences",
    )
    return parser.parse_args()


def prepare_training_text(
    manifests_dir: str, lang_dir: str, word_delimiter: str = " | "
):
    """
    Extract training text from supervision manifests.

    Args:
        manifests_dir: Directory containing supervision files
        lang_dir: Output directory for training text files
        word_delimiter: Delimiter separating words in phoneme sequences
    """
    manifests_dir = Path(manifests_dir)
    lang_dir = Path(lang_dir)

    # Find all training supervision files
    supervision_files = list(manifests_dir.glob("*_supervisions_train*.jsonl.gz"))

    if not supervision_files:
        raise ValueError(f"No training supervision files found in {manifests_dir}")

    phoneme_sequences = []
    word_sequences = []

    for supervision_file in supervision_files:
        logging.info(f"Processing {supervision_file}")

        with gzip.open(supervision_file, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data["text"].strip()

                if not text:
                    continue

                # Add phoneme sequence (tokens)
                phoneme_sequences.append(text)

                # Convert to word sequence
                # Split by word delimiter, then join with spaces
                if word_delimiter in text:
                    words = text.split(word_delimiter)
                    # Clean up each word (remove extra spaces)
                    words = [word.strip() for word in words if word.strip()]
                    if words:
                        word_sequences.append(" ".join(words))
                else:
                    # If no delimiter, treat as single word
                    word_sequences.append(text)

    # Write transcript_tokens.txt (phoneme sequences)
    tokens_file = lang_dir / "transcript_tokens.txt"
    with open(tokens_file, "w", encoding="utf-8") as f:
        for seq in phoneme_sequences:
            f.write(seq + "\n")

    logging.info(f"Wrote {len(phoneme_sequences)} phoneme sequences to {tokens_file}")

    # Write transcript_words.txt (word sequences)
    words_file = lang_dir / "transcript_words.txt"
    with open(words_file, "w", encoding="utf-8") as f:
        for seq in word_sequences:
            f.write(seq + "\n")

    logging.info(f"Wrote {len(word_sequences)} word sequences to {words_file}")


def main():
    args = get_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    prepare_training_text(args.manifests_dir, args.lang_dir, args.word_delimiter)


if __name__ == "__main__":
    main()
