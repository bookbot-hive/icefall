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

            - fleurs_sw_cuts_train.jsonl.gz
            - fleurs_sw_cuts_validation.jsonl.gz
            - fleurs_sw_cuts_test.jsonl.gz
            - common_voice_16_1_sw_cuts_train.jsonl.gz
            - common_voice_16_1_sw_cuts_validation.jsonl.gz
            - common_voice_16_1_sw_cuts_test.jsonl.gz
            - ALFFA_swahili_cuts_train.jsonl.gz
            - ALFFA_swahili_cuts_validation.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # FLEURS
        logging.info("Loading FLEURS in lazy mode")
        fleurs_cuts = load_manifest_lazy(self.manifest_dir / "fleurs_sw_cuts_train.jsonl.gz")

        # Common Voice
        logging.info("Loading Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(self.manifest_dir / "common_voice_16_1_sw_cuts_train.jsonl.gz")

        # ALFFA
        logging.info("Loading ALFFA in lazy mode")
        alffa_cuts = load_manifest_lazy(self.manifest_dir / "ALFFA_swahili_cuts_train.jsonl.gz")

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_sw_v1_cuts_train.jsonl.gz")

        # Bookbot noisy
        logging.info("Loading Bookbot noisy in lazy mode")
        bookbot_noisy_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_sw_noisy_v1_cuts_train.jsonl.gz")

        return CutSet.mux(
            commonvoice_cuts,
            fleurs_cuts,
            alffa_cuts,
            bookbot_cuts,
            bookbot_noisy_cuts,
            weights=[0.62, 0.13, 0.13, 0.1, 0.02],
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # FLEURS
        logging.info("Loading FLEURS in lazy mode")
        fleurs_cuts = load_manifest_lazy(self.manifest_dir / "fleurs_sw_cuts_validation.jsonl.gz")

        # Common Voice
        logging.info("Loading Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(self.manifest_dir / "common_voice_16_1_sw_cuts_validation.jsonl.gz")

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_sw_v1_cuts_validation.jsonl.gz")

        return CutSet.mux(fleurs_cuts, commonvoice_cuts, bookbot_cuts)

    @lru_cache()
    def test_cuts_fleurs(self) -> CutSet:
        logging.info("About to get FLEURS test cuts")

        logging.info("Loading FLEURS in lazy mode")
        fleurs_cuts = load_manifest_lazy(self.manifest_dir / "fleurs_sw_cuts_test.jsonl.gz")

        return fleurs_cuts

    @lru_cache()
    def test_cuts_commonvoice(self) -> CutSet:
        logging.info("About to get Common Voice test cuts")

        logging.info("Loading Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(self.manifest_dir / "common_voice_16_1_sw_cuts_test.jsonl.gz")

        return commonvoice_cuts

    @lru_cache()
    def test_cuts_bookbot(self) -> CutSet:
        logging.info("About to get Bookbot test cuts")

        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_sw_v1_cuts_test.jsonl.gz")

        return bookbot_cuts
