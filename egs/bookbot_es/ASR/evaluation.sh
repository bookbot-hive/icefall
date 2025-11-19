for m in greedy_search fast_beam_search modified_beam_search; do
  ./zipformer/streaming_decode.py \
    --epoch 80 \
    --avg 5 \
    --causal 1 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --exp-dir tmp/zipformer-streaming-robust-es-v0/ \
    --use-transducer True \
    --decoding-method $m \
    --num-decode-streams 1000
done