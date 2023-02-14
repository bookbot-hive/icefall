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


from lhotse import load_manifest_lazy


def main():
    path = "./data/fbank/bookbot_cuts_train.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe()


if __name__ == "__main__":
    main()

"""
## train
Cuts count: 154152
Total duration (hh:mm:ss): 136:06:05
Speech duration (hours): 136:06:05 (100.0%)
***
Duration statistics (seconds):
mean    3.2
std     4.5
min     0.2
25%     0.5
50%     1.8
75%     3.9
99%     22.1
99.5%   28.2
99.9%   42.9
max     91.5
"""