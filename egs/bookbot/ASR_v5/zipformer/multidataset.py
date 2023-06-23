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
            - common-voice-accent-au_cuts_test.jsonl.gz
            - common-voice-accent-au_cuts_train.jsonl.gz
            - common-voice-accent-au_cuts_validation.jsonl.gz
            - common-voice-accent-ca_cuts_test.jsonl.gz
            - common-voice-accent-ca_cuts_train.jsonl.gz
            - common-voice-accent-ca_cuts_validation.jsonl.gz
            - common-voice-accent-gb_cuts_test.jsonl.gz
            - common-voice-accent-gb_cuts_train.jsonl.gz
            - common-voice-accent-gb_cuts_validation.jsonl.gz
            - common-voice-accent-in_cuts_test.jsonl.gz
            - common-voice-accent-in_cuts_train.jsonl.gz
            - common-voice-accent-in_cuts_validation.jsonl.gz
            - common-voice-accent-nz_cuts_test.jsonl.gz
            - common-voice-accent-nz_cuts_train.jsonl.gz
            - common-voice-accent-nz_cuts_validation.jsonl.gz
            - common-voice-accent-us_cuts_test.jsonl.gz
            - common-voice-accent-us_cuts_train.jsonl.gz
            - common-voice-accent-us_cuts_validation.jsonl.gz
            - libriphone_cuts_dev.jsonl.gz
            - libriphone_cuts_test.jsonl.gz
            - libriphone_cuts_train.clean.jsonl.gz
            - timit_cuts_test.jsonl.gz
            - timit_cuts_train.jsonl.gz
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
        logging.info("Loading Common Voice US in lazy mode")
        commonvoice_us_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-us_cuts_train.jsonl.gz"
        )

        logging.info("Loading Common Voice GB in lazy mode")
        commonvoice_gb_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-gb_cuts_train.jsonl.gz"
        )

        logging.info("Loading Common Voice AU in lazy mode")
        commonvoice_au_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-au_cuts_train.jsonl.gz"
        )

        logging.info("Loading Common Voice NZ in lazy mode")
        commonvoice_nz_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-nz_cuts_train.jsonl.gz"
        )

        logging.info("Loading Common Voice CA in lazy mode")
        commonvoice_ca_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-ca_cuts_train.jsonl.gz"
        )

        logging.info("Loading Common Voice IN in lazy mode")
        commonvoice_in_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-in_cuts_train.jsonl.gz"
        )

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(
            self.manifest_dir / "bookbot_en_v1-v2_cuts_train.jsonl.gz"
        )

        return CutSet.mux(
            timit_cuts,
            libriphone_cuts,
            commonvoice_us_cuts,
            commonvoice_gb_cuts,
            commonvoice_au_cuts,
            commonvoice_nz_cuts,
            commonvoice_ca_cuts,
            commonvoice_in_cuts,
            bookbot_cuts,
            weights=[0.007, 0.042, 0.367, 0.152, 0.049, 0.01, 0.067, 0.115, 0.191],
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # LibriPhone
        logging.info("Loading LibriPhone in lazy mode")
        libriphone_cuts = load_manifest_lazy(
            self.manifest_dir / "libriphone_cuts_dev.jsonl.gz"
        )

        # Common Voice
        logging.info("Loading Common Voice US in lazy mode")
        commonvoice_us_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-us_cuts_validation.jsonl.gz"
        )

        logging.info("Loading Common Voice GB in lazy mode")
        commonvoice_gb_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-gb_cuts_validation.jsonl.gz"
        )

        logging.info("Loading Common Voice AU in lazy mode")
        commonvoice_au_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-au_cuts_validation.jsonl.gz"
        )

        logging.info("Loading Common Voice NZ in lazy mode")
        commonvoice_nz_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-nz_cuts_validation.jsonl.gz"
        )

        logging.info("Loading Common Voice CA in lazy mode")
        commonvoice_ca_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-ca_cuts_validation.jsonl.gz"
        )

        logging.info("Loading Common Voice IN in lazy mode")
        commonvoice_in_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-in_cuts_validation.jsonl.gz"
        )

        # Bookbot
        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(
            self.manifest_dir / "bookbot_en_v1-v2_cuts_validation.jsonl.gz"
        )

        return CutSet.mux(
            libriphone_cuts,
            commonvoice_us_cuts,
            commonvoice_gb_cuts,
            commonvoice_au_cuts,
            commonvoice_nz_cuts,
            commonvoice_ca_cuts,
            commonvoice_in_cuts,
            bookbot_cuts,
        )

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
    def test_cuts_cva_us(self) -> CutSet:
        logging.info("About to get Common Voice US test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-us_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_cva_gb(self) -> CutSet:
        logging.info("About to get Common Voice GB test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-gb_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_cva_au(self) -> CutSet:
        logging.info("About to get Common Voice AU test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-au_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_cva_nz(self) -> CutSet:
        logging.info("About to get Common Voice NZ test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-nz_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_cva_ca(self) -> CutSet:
        logging.info("About to get Common Voice CA test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-ca_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_cva_in(self) -> CutSet:
        logging.info("About to get Common Voice IN test cuts")

        logging.info("Loading Common Voice US in lazy mode")
        cva_cuts = load_manifest_lazy(
            self.manifest_dir / "common-voice-accent-in_cuts_test.jsonl.gz"
        )

        return cva_cuts

    @lru_cache()
    def test_cuts_bookbot(self) -> CutSet:
        logging.info("About to get Bookbot test cuts")

        logging.info("Loading Bookbot in lazy mode")
        bookbot_cuts = load_manifest_lazy(
            self.manifest_dir / "bookbot_en_v1-v2_cuts_test.jsonl.gz"
        )

        return bookbot_cuts

    # @lru_cache()
    # def test_cuts_austalk(self) -> CutSet:
    #     logging.info("About to get AusTalk test cuts")

    #     logging.info("Loading AusTalk in lazy mode")
    #     austalk_cuts = load_manifest_lazy(
    #         self.manifest_dir / "austalk_words_mq_cuts_test.jsonl.gz"
    #     )

    #     return austalk_cuts

    # @lru_cache()
    # def test_cuts_sccw(self) -> CutSet:
    #     logging.info("About to get SC-CW test cuts")

    #     logging.info("Loading SC-CW in lazy mode")
    #     sccw_cuts = load_manifest_lazy(
    #         self.manifest_dir / "sc_cw_children_cuts_test.jsonl.gz"
    #     )

    #     return sccw_cuts

    # @lru_cache()
    # def test_cuts_l2a(self) -> CutSet:
    #     logging.info("About to get L2 Arctic test cuts")

    #     logging.info("Loading L2 Arctic in lazy mode")
    #     l2a_cuts = load_manifest_lazy(
    #         self.manifest_dir / "l2-arctic_cuts_test.jsonl.gz"
    #     )

    #     return l2a_cuts

    # @lru_cache()
    # def test_cuts_speechocean(self) -> CutSet:
    #     logging.info("About to get SpeechOcean test cuts")

    #     logging.info("Loading SpeechOcean in lazy mode")
    #     speechocean_cuts = load_manifest_lazy(
    #         self.manifest_dir / "speechocean762_cuts_test.jsonl.gz"
    #     )

    #     return speechocean_cuts
