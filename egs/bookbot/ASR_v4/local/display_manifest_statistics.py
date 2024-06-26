#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in transducer/train.py
for usage.
"""

from pathlib import Path
from lhotse import combine, load_manifest_lazy


def main():
    filenames = Path("./data/fbank/").rglob("*_cuts_train*.jsonl.gz")

    cuts = combine(load_manifest_lazy(p) for p in filenames)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
## train
Cuts count: 144863
Total duration (hh:mm:ss): 219:55:44
Speech duration (hours): 219:55:44 (100.0%)
***
Duration statistics (seconds):
mean    5.5
std     4.3
min     0.8
25%     2.6
50%     3.9
75%     6.1
99%     18.6
99.5%   19.5
99.9%   20.5
max     83.3
"""
