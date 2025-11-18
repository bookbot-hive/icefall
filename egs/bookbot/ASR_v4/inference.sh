# ./zipformer/jit_pretrained_streaming.py \
#     --nn-model-filename ./tmp/zipformer-streaming-robust-en-v10/exp-causal/jit_script_chunk_32_left_128.pt \
#     --tokens ./tmp/zipformer-streaming-robust-en-v10/data/lang_phone/tokens.txt \
#     /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/native-test-suite/debug_audios/end_of_sentence_cut_off/en-AU-DeanNeural-1104.wav  


./zipformer/onnx_pretrained-streaming.py \
  --encoder-model-filename ~/sherpa-onnx-zipformer-streaming-robust-en-v10/encoder-epoch-40-avg-16-chunk-16-left-128.int8.onnx \
  --decoder-model-filename ~/sherpa-onnx-zipformer-streaming-robust-en-v10/decoder-epoch-40-avg-16-chunk-16-left-128.int8.onnx\
  --joiner-model-filename ~/sherpa-onnx-zipformer-streaming-robust-en-v10/joiner-epoch-40-avg-16-chunk-16-left-128.int8.onnx \
  --tokens ~/sherpa-onnx-zipformer-streaming-robust-en-v10/tokens.txt \
    /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/lexicon-extraction-pipeline/neural-books/test_waves/sample1.wav