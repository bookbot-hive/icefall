#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/bookbot
#      You can find `training` folder inside it.
#
#  - $dl_dir/lm
#      This directory contains the language model(LM) downloaded from
#      https://huggingface.co/bookbot/bookbot_en_kaldilm, and the LM is based
#	     on 40 phones. About how to get these LM files, you can know it
#      from https://github.com/luomingshuang/Train_LM_with_kaldilm.
#
#	  - lm_3_gram.arpa
#     - lm_4_gram.arpa
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
dl_dir=$PWD/download
splits_dir=$PWD/splits_dir

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download LM"
  # We assume that you have installed the git-lfs, if not, you could install it
  # using: `sudo apt-get install git-lfs && git-lfs install`
  [ ! -e $dl_dir/lm ] && mkdir -p $dl_dir/lm
  git clone https://huggingface.co/bookbot/bookbot_en_kaldilm $dl_dir/lm
  cd $dl_dir/lm && git lfs pull
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/bookbot,
  # you can create a symlink
  #
  #   ln -sfv /path/to/bookbot $dl_dir/bookbot
  #
  if [ ! -d $dl_dir/bookbot ]; then
    lhotse download bookbot bookbot/bookbot_en_v1-v2 $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare Bookbot manifest"

  # Create a symlink from local path
  #
  #   ln -sfv /path/to/training $dl_dir/bookbot/
  #
  mkdir -p data/manifests
  lhotse prepare bookbot $dl_dir/bookbot/training data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for Bookbot"
  mkdir -p data/fbank
  ./local/compute_fbank_bookbot.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  ./local/prepare_lexicon.py \
   --manifests-dir data/manifests \
   --lang-dir $lang_dir

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare G"
  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/lm_3_gram.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/lm_4_gram.arpa > data/lm/G_4_gram.fst.txt
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone
fi
