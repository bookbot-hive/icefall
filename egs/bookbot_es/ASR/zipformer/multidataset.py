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

            - common_voice_16_1_es_cuts_train.jsonl.gz
            - common_voice_16_1_es_cuts_test.jsonl.gz
            - slr72_dataset_cuts_train.jsonl.gz
            - slr72_dataset_cuts_test.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts from all datasets")

        # Spanish Common Voice
        logging.info("Loading Spanish Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(
            self.manifest_dir / "common_voice_16_1_es_cuts_train.jsonl.gz"
        )

        # SLR72 Dataset
        logging.info("Loading SLR72 dataset in lazy mode")
        slr72_cuts = load_manifest_lazy(
            self.manifest_dir / "slr72_dataset_cuts_train.jsonl.gz"
        )

        # Combine all training datasets
        logging.info("Combining all training datasets")
        combined_cuts = commonvoice_cuts + slr72_cuts

        return combined_cuts

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get validation cuts from all datasets")

        # Use test sets as validation since we only have train/test splits
        logging.info("Loading Spanish Common Voice test set as validation")
        commonvoice_cuts = load_manifest_lazy(
            self.manifest_dir / "common_voice_16_1_es_cuts_test.jsonl.gz"
        )

        logging.info("Loading SLR72 dataset test set as validation")
        slr72_cuts = load_manifest_lazy(
            self.manifest_dir / "slr72_dataset_cuts_test.jsonl.gz"
        )

        # Combine all validation datasets
        logging.info("Combining all validation datasets")
        combined_cuts = commonvoice_cuts + slr72_cuts

        return combined_cuts

    @lru_cache()
    def test_cuts_commonvoice(self) -> CutSet:
        logging.info("About to get Spanish Common Voice test cuts")

        logging.info("Loading Spanish Common Voice in lazy mode")
        commonvoice_cuts = load_manifest_lazy(
            self.manifest_dir / "common_voice_16_1_es_cuts_test.jsonl.gz"
        )

        return commonvoice_cuts

    @lru_cache()
    def test_cuts_slr72(self) -> CutSet:
        logging.info("About to get SLR72 dataset test cuts")

        logging.info("Loading SLR72 dataset in lazy mode")
        slr72_cuts = load_manifest_lazy(
            self.manifest_dir / "slr72_dataset_cuts_test.jsonl.gz"
        )

        return slr72_cuts

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts from all datasets")

        # Spanish Common Voice
        logging.info("Loading Spanish Common Voice test cuts")
        commonvoice_cuts = load_manifest_lazy(
            self.manifest_dir / "common_voice_16_1_es_cuts_test.jsonl.gz"
        )

        # SLR72 Dataset
        logging.info("Loading SLR72 dataset test cuts")
        slr72_cuts = load_manifest_lazy(
            self.manifest_dir / "slr72_dataset_cuts_test.jsonl.gz"
        )

        # Combine all test datasets
        logging.info("Combining all test datasets")
        combined_cuts = commonvoice_cuts + slr72_cuts

        return combined_cuts
