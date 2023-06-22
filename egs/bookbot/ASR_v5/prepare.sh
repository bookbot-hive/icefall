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

  if [ ! -d $dl_dir/timit ]; then
    lhotse download bookbot-huggingface bookbot/timit $dl_dir text ""
  fi

  if [ ! -d $dl_dir/libriphone ]; then
    lhotse download bookbot-huggingface bookbot/libriphone $dl_dir text ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-us ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-us $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-gb ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-gb $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-au ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-au $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-nz ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-nz $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-in ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-in $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/common-voice-accent-ca ]; then
    lhotse download bookbot-huggingface bookbot/common-voice-accent-ca $dl_dir sentence ""
  fi

  if [ ! -d $dl_dir/bookbot_en_v1-v2 ]; then
    lhotse download bookbot-huggingface bookbot/bookbot_en_v1-v2 $dl_dir text ""
  fi

  # if [ ! -d $dl_dir/austalk_words_mq ]; then
  #   lhotse download bookbot-huggingface bookbot/austalk_words_mq $dl_dir
  # fi

  # if [ ! -d $dl_dir/sc_cw_children ]; then
  #   lhotse download bookbot-huggingface bookbot/sc_cw_children $dl_dir
  # fi

  # if [ ! -d $dl_dir/l2-arctic ]; then
  #   lhotse download bookbot-huggingface bookbot/l2-arctic $dl_dir
  # fi

  # if [ ! -d $dl_dir/speechocean762 ]; then
  #   lhotse download bookbot-huggingface bookbot/speechocean762 $dl_dir
  # fi

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

  lhotse prepare bookbot-huggingface $dl_dir/timit data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/libriphone data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-us data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-gb data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-au data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-nz data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-in data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/common-voice-accent-ca data/manifests --normalize-words=true
  lhotse prepare bookbot-huggingface $dl_dir/bookbot_en_v1-v2 data/manifests --normalize-words=true
  # lhotse prepare bookbot-huggingface $dl_dir/austalk_words_mq data/manifests
  # lhotse prepare bookbot-huggingface $dl_dir/sc_cw_children data/manifests
  # lhotse prepare bookbot-huggingface $dl_dir/l2-arctic data/manifests
  # lhotse prepare bookbot-huggingface $dl_dir/speechocean762 data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
  lhotse prepare hallway $dl_dir/audio_splits data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbanks"
  mkdir -p data/fbank

  ./local/compute_fbank_timit.py
  ./local/compute_fbank_libriphone.py
  ./local/compute_fbank_commonvoice_accented.py
  ./local/compute_fbank_bookbot.py
  # ./local/compute_fbank_austalk.py
  # ./local/compute_fbank_sccw.py
  # ./local/compute_fbank_l2.py
  # ./local/compute_fbank_speechocean.py
  ./local/compute_fbank_musan.py
  ./local/compute_fbank_hallway.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  (echo '!SIL	SIL'; echo '<SPOKEN_NOISE>	SPN'; echo '<UNK>	SPN'; ) |
    cat - $dl_dir/lm/lexicon.tsv |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi

  if [ ! -f $lang_dir/L.fst ]; then
    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L.pt \
      $lang_dir/L.fst
  fi

  if [ ! -f $lang_dir/L_disambig.fst ]; then
    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L_disambig.pt \
      $lang_dir/L_disambig.fst
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/4-gram.arpa > data/lm/G_4_gram.fst.txt
  fi
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compile LG"
  ./local/compile_lg.py --lang-dir data/lang_phone
fi