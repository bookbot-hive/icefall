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

            - bookbot_en_v1-v2_cuts_test.jsonl.gz
            - bookbot_en_v1-v2_cuts_train.jsonl.gz
            - bookbot_en_v1-v2_cuts_validation.jsonl.gz
            - GigaSpeech_cuts_train.jsonl.gz
            - GigaSpeech_cuts_test.jsonl.gz
            - GigaSpeech_cuts_validation.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # GigaSpeech
        logging.info("Loading GigaSpeech in lazy mode")
        gigaspeech_cuts = load_manifest_lazy(self.manifest_dir / "GigaSpeech_cuts_train.jsonl.gz")

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_en_v1-v2_cuts_train.jsonl.gz")

        return CutSet.mux(gigaspeech_cuts, bookbot_cuts)

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # GigaSpeech
        logging.info("Loading GigaSpeech in lazy mode")
        gigaspeech_cuts = load_manifest_lazy(self.manifest_dir / "GigaSpeech_cuts_validation.jsonl.gz")

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_en_v1-v2_cuts_validation.jsonl.gz")

        return CutSet.mux(gigaspeech_cuts, bookbot_cuts)

    @lru_cache()
    def test_cuts_bookbot(self) -> CutSet:
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(self.manifest_dir / "bookbot_en_v1-v2_cuts_test.jsonl.gz")

        return bookbot_cuts

    @lru_cache()
    def test_cuts_gigaspeech(self) -> CutSet:
        logging.info("Loading GigaSpeech in lazy mode")
        gigaspeech_cuts = load_manifest_lazy(self.manifest_dir / "GigaSpeech_cuts_test.jsonl.gz")

        return gigaspeech_cuts
