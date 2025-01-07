#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
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

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from tqdm.contrib.concurrent import process_map
from g2p_id import G2p

g2p = G2p()
WORD_TOKEN_DELIMITER = " "


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("download/gigaspeech2"),
    )
    return parser


def _phonemize(audio_info: str) -> str:
    segment_id, text = audio_info.split("\t")
    try:
        phonemes = WORD_TOKEN_DELIMITER.join(phoneme for word in g2p(text) for phoneme in word)
        return f"{segment_id}\t{phonemes}"
    except Exception as e:
        logging.warning(f"Failed to phonemize {segment_id}: {e}, {text}")
        return None


def phonemize_gigaspeech(args):
    data_dir = args.download_dir / "data" / "id"
    tsv_files = sorted(data_dir.glob("*.tsv"))

    for tsv_file in tsv_files:
        if ".phoneme" in tsv_file.name:
            continue

        with open(tsv_file) as f:
            audio_infos = f.read().splitlines()
        audio_infos = process_map(_phonemize, audio_infos, desc=tsv_file.name, chunksize=1000)
        audio_infos = [info for info in audio_infos if info is not None]

        with open(tsv_file.with_suffix(".phoneme.tsv"), "w") as f:
            f.write("\n".join(audio_infos))


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    phonemize_gigaspeech(args)


if __name__ == "__main__":
    main()
