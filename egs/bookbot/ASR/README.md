<!-- Copied from: https://icefall.readthedocs.io/en/latest/recipes/Non-streaming-ASR/timit/tdnn_ligru_ctc.html -->

# TDNN-LiGRU-CTC

<!-- permissions and paths
chmod 755 *.sh
chmod 755 */*.py
export PYTHONPATH=/Users/mac/Desktop/notebooks/bookbot/ml/icefall:$PYTHONPATH 
-->

## Data preparation

```sh
cd egs/bookbot/ASR
./prepare.sh
```

## Training

```sh
cd egs/bookbot/ASR
export CUDA_VISIBLE_DEVICES="0"
./tdnn_ligru_ctc/train.py
```

## Decoding

```sh
export CUDA_VISIBLE_DEVICES="0"
./tdnn_ligru_ctc/decode.py
```