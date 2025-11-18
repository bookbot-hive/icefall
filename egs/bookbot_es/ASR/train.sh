export CUDA_VISIBLE_DEVICES="0,1"
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 80 \
  --exp-dir tmp/exp-causal-80-epoch \
  --causal 1 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --max-duration 1000 \
  --base-lr 0.04 \
  --use-transducer True \
  --use-fp16 1