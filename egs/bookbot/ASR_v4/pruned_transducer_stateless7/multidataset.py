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
            - timit_cuts_test.jsonl.gz
            - libriphone_cuts_train.clean.jsonl.gz
            - libriphone_cuts_dev.jsonl.gz
            - libriphone_cuts_test.jsonl.gz
            - common_voice_13_0_en_zipformer_cuts_train.jsonl.gz
            - gigaspeech_zipformer_cuts_train.jsonl.gz
            - bookbot_en_phonemes_cuts_train.jsonl.gz
            - austalk_words_mq_cuts_test.jsonl.gz
            - sc_cw_children_cuts_test.jsonl.gz
            - l2-arctic_cuts_test.jsonl.gz
            - speechocean762_cuts_test.jsonl.gz
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
            self.manifest_dir / "common_voice_13_0_en_zipformer_cuts_train.jsonl.gz"
        )

        # GigaSpeech
        logging.info("Loading GigaSpeech in lazy mode")
        gigaspeech_cuts = load_manifest_lazy(
            self.manifest_dir / "gigaspeech_zipformer_cuts_train.jsonl.gz"
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
            gigaspeech_cuts,
            bookbot_cuts,
            weights=[0.002, 0.105, 0.213, 0.469, 0.211],
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get LibriPhone dev cuts")

        logging.info("Loading LibriPhone in lazy mode")
        libriphone_cuts = load_manifest_lazy(
            self.manifest_dir / "libriphone_cuts_dev.jsonl.gz"
        )

        return libriphone_cuts

    @lru_cache()
    def test_cuts_libri(self) -> CutSet:
        logging.info("About to get LibriPhone test cuts")

        logging.info("Loading LibriPhone in lazy mode")
        libriphone_cuts = load_manifest_lazy(
            self.manifest_dir / "libriphone_cuts_test.jsonl.gz"
        )

        return libriphone_cuts

    @lru_cache()
    def test_cuts_timit(self) -> CutSet:
        logging.info("About to get TIMIT test cuts")

        logging.info("Loading TIMIT in lazy mode")
        timit_cuts = load_manifest_lazy(self.manifest_dir / "timit_cuts_test.jsonl.gz")

        return timit_cuts

    @lru_cache()
    def test_cuts_austalk(self) -> CutSet:
        logging.info("About to get AusTalk test cuts")

        logging.info("Loading AusTalk in lazy mode")
        austalk_cuts = load_manifest_lazy(
            self.manifest_dir / "austalk_words_mq_cuts_test.jsonl.gz"
        )

        return austalk_cuts

    @lru_cache()
    def test_cuts_sccw(self) -> CutSet:
        logging.info("About to get SC-CW test cuts")

        logging.info("Loading SC-CW in lazy mode")
        sccw_cuts = load_manifest_lazy(
            self.manifest_dir / "sc_cw_children_cuts_test.jsonl.gz"
        )

        return sccw_cuts

    @lru_cache()
    def test_cuts_l2a(self) -> CutSet:
        logging.info("About to get L2 Arctic test cuts")

        logging.info("Loading L2 Arctic in lazy mode")
        l2a_cuts = load_manifest_lazy(
            self.manifest_dir / "l2-arctic_cuts_test.jsonl.gz"
        )

        return l2a_cuts

    @lru_cache()
    def test_cuts_speechocean(self) -> CutSet:
        logging.info("About to get SpeechOcean test cuts")

        logging.info("Loading SpeechOcean in lazy mode")
        speechocean_cuts = load_manifest_lazy(
            self.manifest_dir / "speechocean762_cuts_test.jsonl.gz"
        )

        return speechocean_cuts
