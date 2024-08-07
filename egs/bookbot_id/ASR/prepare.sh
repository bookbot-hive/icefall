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
    lhotse download bookbot-huggingface bookbot/librivox-indonesia $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/common-voice-13_0-id ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-13_0-id $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/fleurs-id ]; then
    lhotse download bookbot-huggingface bookbot/fleurs-id $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/bookbot_id_v4 ]; then
    lhotse download bookbot-huggingface bookbot/bookbot_id_v4 $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-US-MultiNeural ]; then
    lhotse download bookbot-huggingface bookbot/id-US-MultiNeural $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/magichub-indocsc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-indocsc $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/magichub-sindodusc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-sindodusc $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-ID-AlthafNeural-Matcha ]; then
    lhotse download bookbot-huggingface bookbot/id-ID-AlthafNeural-Matcha $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-ID-AlthafNeural ]; then
    lhotse download bookbot-huggingface bookbot/id-ID-AlthafNeural $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-ID-Althaf-Neural-words ]; then
    lhotse download bookbot-huggingface bookbot/id-ID-Althaf-Neural-words $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-ZA-Neural-words ]; then
    lhotse download bookbot-huggingface bookbot/id-ZA-Neural-words $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-IN-1-Neural-words ]; then
    lhotse download bookbot-huggingface bookbot/id-IN-1-Neural-words $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-IN-2-Neural-words ]; then
    lhotse download bookbot-huggingface bookbot/id-IN-2-Neural-words $dl_dir phonemes_ipa " "
  fi

  if [ ! -d $dl_dir/id-US-Neural-words ]; then
    lhotse download bookbot-huggingface bookbot/id-US-Neural-words $dl_dir phonemes_ipa " "
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
  lhotse prepare bookbot-huggingface $dl_dir/bookbot_id_v4 data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-US-MultiNeural data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/magichub-indocsc data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/magichub-sindodusc data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-ID-AlthafNeural-Matcha data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-ID-AlthafNeural data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-ID-Althaf-Neural-words data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-ZA-Neural-words data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-IN-1-Neural-words data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-IN-2-Neural-words data/manifests
  lhotse prepare bookbot-huggingface $dl_dir/id-US-Neural-words data/manifests
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
  ./local/compute_fbank_id_us.py
  ./local/compute_fbank_althaf.py
  ./local/compute_fbank_althaf_matcha.py
  ./local/compute_fbank_althaf_neural_words.py
  ./local/compute_fbank_id_za_words.py
  ./local/compute_fbank_id_in_1_words.py
  ./local/compute_fbank_id_in_2_words.py
  ./local/compute_fbank_id_us_words.py
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