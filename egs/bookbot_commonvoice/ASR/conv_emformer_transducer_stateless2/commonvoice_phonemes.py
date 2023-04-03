# Copyright      2021  Piotr Żelasko
#                2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy


class CommonVoicePhonemes:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - commonvoice_cuts_train.jsonl.gz
                - commonvoice_cuts_validation.jsonl.gz
                - commonvoice_cuts_test.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)
    
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        cuts_train = load_manifest_lazy(
            self.args.feature_dir / "commonvoice_cuts_train.jsonl.gz"
        )

        return cuts_train

    def valid_cuts(self) -> CutSet:
        logging.info("About to get validation cuts")
        cuts_valid = load_manifest_lazy(
            self.args.feature_dir / "commonvoice_cuts_validation.jsonl.gz"
        )

        return cuts_valid

    def test_cuts(self) -> CutSet:
        logging.debug("About to get test cuts")
        cuts_test = load_manifest_lazy(
            self.args.feature_dir / "commonvoice_cuts_test.jsonl.gz"
        )

        return cuts_test