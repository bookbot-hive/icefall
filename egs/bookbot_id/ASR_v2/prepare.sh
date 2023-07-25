#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

vocab_sizes=(
  # 5000
  # 2000
  # 1000
  500
)

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
    lhotse download bookbot-huggingface bookbot/librivox-indonesia $dl_dir text ""
  fi

  if [ ! -d $dl_dir/common-voice-13_0-id ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-13_0-id $dl_dir text ""
  fi

  if [ ! -d $dl_dir/fleurs-id ]; then
    lhotse download bookbot-huggingface bookbot/fleurs-id $dl_dir text ""
  fi

  if [ ! -d $dl_dir/bookbot_id_phonemes ]; then
    lhotse download bookbot-huggingface bookbot/bookbot_id_phonemes $dl_dir text ""
  fi

  if [ ! -d $dl_dir/magichub-indocsc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-indocsc $dl_dir text ""
  fi

  if [ ! -d $dl_dir/magichub-sindodusc ]; then
    lhotse download bookbot-huggingface bookbot/magichub-sindodusc $dl_dir text ""
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

  lhotse prepare bookbot-huggingface $dl_dir/librivox-indonesia data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-13_0-id data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/fleurs-id data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/bookbot_id_phonemes data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/magichub-indocsc data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/magichub-sindodusc data/manifests --normalize-words=true
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
  log "Stage 3: Prepare BPE train data and set of words"
  lang_dir=data/lang
  mkdir -p $lang_dir

  if [ ! -f $lang_dir/train.txt ]; then
    ./local/prepare_transcripts.py \
      --manifests-dir data/manifests \
      --output-text-path $lang_dir/train.txt
  fi

  if [ ! -f $lang_dir/words.txt ]; then
    ./local/prepare_words.py \
      --manifests-dir data/manifests \
      --lang-dir $lang_dir
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    cp data/lang/words.txt $lang_dir

    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript data/lang/train.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir --oov "<unk>"
    fi
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Generate LM training data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    lang_dir=data/lang_bpe_${vocab_size}
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --manifests-dir data/manifests \
      --split train \
      --lm-archive $out_dir/lm_data.pt
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Generate LM validation data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    lang_dir=data/lang_bpe_${vocab_size}
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --manifests-dir data/manifests \
      --split '[dev]' \
      --lm-archive $out_dir/lm_data-valid.pt
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Generate LM test data"

  for vocab_size in ${vocab_sizes[@]}; do
    log "Processing vocab_size == ${vocab_size}"
    lang_dir=data/lang_bpe_${vocab_size}
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir

    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --manifests-dir data/manifests \
      --split test \
      --lm-archive $out_dir/lm_data-test.pt
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Sort LM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of BPE tokens
  # in a sentence.

  for vocab_size in ${vocab_sizes[@]}; do
    out_dir=data/lm_training_bpe_${vocab_size}
    mkdir -p $out_dir
    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data.pt \
      --out-lm-data $out_dir/sorted_lm_data.pt \
      --out-statistics $out_dir/statistics.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-valid.pt \
      --out-lm-data $out_dir/sorted_lm_data-valid.pt \
      --out-statistics $out_dir/statistics-valid.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-test.pt \
      --out-lm-data $out_dir/sorted_lm_data-test.pt \
      --out-statistics $out_dir/statistics-test.txt
  done
fi
