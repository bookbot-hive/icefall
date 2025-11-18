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
Cuts count: 362921
Total duration (hh:mm:ss): 194:47:34
Speech duration (hours): 194:47:34 (100.0%)
***
Duration statistics (seconds):
mean    1.9
std     2.2
min     0.0
25%     0.6
50%     1.1
75%     2.4
99%     10.4
99.5%   13.1
99.9%   19.4
max     96.0
"""
