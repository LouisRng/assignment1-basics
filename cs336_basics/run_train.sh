#!/usr/bin/env bash

# python train.py \
#     --train_data ../result/TinyStoriesV2-GPT4-train_tokens.npy \
#     --val_data ../result/TinyStoriesV2-GPT4-valid_tokens.npy \
#     --vocab_size 10000 \
#     --d_model 256 \
#     --num_layer 4 \
#     --num_head 8 \
#     --d_ff 768 \
#     --context_length 256 \
#     --batch_size 1 \
#     --num_steps 10000 \
#     --vocab_path "../result/tinystories_bpe_vocab.json" \
#     --merges "../result/tinystories_bpe_merge.json" \
#     --generate "what are you doing?"

# python train.py \
#     --vocab_size 10000 \
#     --d_model 512 \
#     --d_ff 1344 \
#     --num_layer 4 \
#     --num_head 16 \
#     --context_length 256 \
#     --batch_size 32 \
#     --num_steps 10000 \
#     --vocab_path "../result/tinystories_bpe_vocab.json" \
#     --merges "../result/tinystories_bpe_merge.json" \
#     --generate "what are you doing?"

# benchmark
 
# python train.py \
#     --vocab_size 10000 \
#     --d_model 256 \
#     --num_layer 4 \
#     --num_head 8 \
#     --d_ff 768 \
#     --context_length 256 \
#     --batch_size 1 \
#     --num_steps 2000 \
#     --vocab_path "../result/tinystories_bpe_vocab.json" \
#     --merges "../result/tinystories_bpe_merge.json" \
#     --bm_mode True \
#     --forward True

# generate
python train.py \
    --train_data ../result/TinyStoriesV2-GPT4-train_tokens.bin \
    --val_data ../result/TinyStoriesV2-GPT4-valid_tokens.bin \
    --vocab_size 10000 \
    --d_model 256 \
    --num_layer 4 \
    --num_head 8 \
    --d_ff 768 \
    --context_length 256 \
    --batch_size 1 \
    --num_steps 500 \
    --from_pretrained "checkpoints/" \
    --generate "hello there" \
    --vocab_path "../result/tinystories_bpe_vocab.json" \
    --merges "../result/tinystories_bpe_merge.json" \
