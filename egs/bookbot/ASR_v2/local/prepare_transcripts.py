#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (author: Mingshuang Luo)
# Copyright    2022  Behavox LLC.        (author: Daniil Kulko)
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
consisting of supervisions_train.json and writes all transcripts into one file.

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
        "--output-text-path",
        type=str,
        help="Output text file path.",
    )

    return parser.parse_args()


def prepare_transcripts(manifests_dir: Path, output_text_path: Path) -> None:
    """
    Args:
      manifests_dir:
        The manifests directory, e.g., data/manifests.
      output_text_path:
        The output data text file path, e.g., data/lang/train.txt.

    Return:
      Saved text file in output_text_path.
    """
    import gzip

    supervisions_train = Path(manifests_dir) / "bookbot_supervisions_train.jsonl.gz"

    logging.info(f"Loading {supervisions_train}!")
    with gzip.open(supervisions_train, "r") as load_f:
        with open(output_text_path, "w+", encoding="utf8") as f:
            for line in load_f.readlines():
                load_dict = json.loads(line)
                text = load_dict["text"]
                f.write(f"{text}\n")

def main() -> None:
    args = get_args()
    manifests_dir = Path(args.manifests_dir)
    output_text_path = Path(args.output_text_path)

    logging.info(f"Generating {output_text_path.name}")
    prepare_transcripts(manifests_dir, output_text_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
