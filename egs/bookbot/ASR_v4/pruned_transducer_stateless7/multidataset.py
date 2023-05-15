# Copyright      2023  Xiaomi Corp.        (authors: Yifan Yang)
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


import logging
from functools import lru_cache
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:

            - timit_cuts_train.jsonl.gz
            - libriphone_cuts_train.clean.jsonl.gz
            - libriphone_cuts_dev.jsonl.gz
            - common-voice-13_0-en-v1_cuts_train.jsonl.gz
            - bookbot_en_phonemes_cuts_train.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # TIMIT
        logging.info("Loading TIMIT in lazy mode")
        timit_cuts = load_manifest_lazy(self.manifest_dir / "timit_cuts_train.jsonl.gz")

        # LibriPhone
        logging.info("Loading LibriPhone in lazy mode")
        libriphone_cuts = load_manifest_lazy(
            self.manifest_dir / "libriphone_cuts_train.clean.jsonl.gz"
        )

        # Common Voice
        logging.info("Loading Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-13_0-en-v1_cuts_train.jsonl.gz"
        )

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(
            self.manifest_dir / "bookbot_en_phonemes_cuts_train.jsonl.gz"
        )

        return CutSet.mux(
            timit_cuts,
            libriphone_cuts,
            commonvoice_cuts,
            bookbot_cuts,
            weights=[0.03, 0.2, 0.57, 0.2],
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get LibriPhone valid cuts")

        logging.info("Loading LibriPhone in lazy mode")
        libriphone_cuts = load_manifest_lazy(
            self.manifest_dir / "libriphone_cuts_dev.jsonl.gz"
        )

        return libriphone_cuts
