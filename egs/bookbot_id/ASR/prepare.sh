#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

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


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  if [ ! -d $dl_dir/librivox-indonesia ]; then
    lhotse download bookbot-huggingface bookbot/librivox-indonesia $dl_dir
  fi

  if [ ! -d $dl_dir/common-voice-13_0-id ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-13_0-id $dl_dir
  fi

  if [ ! -d $dl_dir/fleurs-id ]; then
    lhotse download bookbot-huggingface bookbot/fleurs-id $dl_dir
  fi

  if [ ! -d $dl_dir/bookbot_id_phonemes ]; then
    lhotse download bookbot-huggingface bookbot/bookbot_id_phonemes $dl_dir
  fi

  if [ ! -d $dl_dir/magichub-indocsc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-indocsc $dl_dir
  fi

  if [ ! -d $dl_dir/magichub-sindodusc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-sindodusc $dl_dir
  fi

  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi

  if [ ! -d $dl_dir/audio_splits ]; then
    lhotse download hallway $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare manifests"
  mkdir -p data/manifests

  lhotse prepare bookbot-huggingface $dl_dir/librivox-indonesia data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-13_0-id data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/fleurs-id data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/bookbot_id_phonemes data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/magichub-indocsc data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/magichub-sindodusc data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
  lhotse prepare hallway $dl_dir/audio_splits data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbanks"
  mkdir -p data/fbank

  ./local/compute_fbank_librivox.py
  ./local/compute_fbank_commonvoice.py
  ./local/compute_fbank_fleurs.py
  ./local/compute_fbank_indocsc.py
  ./local/compute_fbank_sindodusc.py
  ./local/compute_fbank_bookbot.py
  ./local/compute_fbank_musan.py
  ./local/compute_fbank_hallway.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  ./local/prepare_lexicon.py \
   --manifests-dir data/manifests \
   --lang-dir $lang_dir

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi
fi