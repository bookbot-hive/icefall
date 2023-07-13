#!/usr/bin/env python3
# Copyright    2022  Behavox LLC.        (authors: Daniil Kulko)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script takes as input supervisions json dir "data/manifests"
consisting of tedlium_supervisions_train.json and does the following:

1. Generate words.txt.

"""
import argparse
import json
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests-dir",
        type=str,
        help="Input directory.",
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="Output directory.",
    )

    return parser.parse_args()


def prepare_words(manifests_dir: Path, lang_dir: Path) -> None:
    """
    Args:
      manifests_dir:
        The manifests directory, e.g., data/manifests.
      lang_dir::
        The language directory, e.g., data/lang.

    Return:
      The words.txt file.
    """
    import gzip

    words = set()
    words_path = Path(lang_dir) / "words.txt"

    supervisions_train = Path(manifests_dir).rglob("*_supervisions_train*.jsonl.gz")

    for supervision in supervisions_train:
        logging.info(f"Loading {supervision}!")
        with gzip.open(supervision, "r") as load_f:
            for line in load_f.readlines():
                load_dict = json.loads(line)
                text = load_dict["text"]
                for word in text.split():
                    words.add(word)

    words.add("<unk>")
    words = ["<eps>", "!SIL"] + sorted(words) + ["#0", "<s>", "</s>"]

    with open(words_path, "w+", encoding="utf8") as f:
        for idx, word in enumerate(words):
            f.write(f"{word} {idx}\n")


def main() -> None:
    args = get_args()
    manifests_dir = Path(args.manifests_dir)
    lang_dir = Path(args.lang_dir)

    logging.info("Generating words.txt")
    prepare_words(manifests_dir, lang_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
