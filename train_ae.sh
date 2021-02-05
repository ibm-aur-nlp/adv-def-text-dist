#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate
DATA_DIR=data
MODEL_DIR=saved_models

python train.py --do_train \
    --vocab_file=$DATA_DIR/vocab.in \
    --emb_file=$DATA_DIR/emb.json \
    --input_file=$DATA_DIR/train.in \
    --output_file=$DATA_DIR/train.out \
    --dev_file=$DATA_DIR/dev.in \
    --dev_output=$DATA_DIR/dev.out \
    --enc_type=bi --attention --enc_num_units=512 --dec_num_units=512 \
    --learning_rate=0.001 --batch_size=32 --max_len=50 \
    --num_epochs=10 --print_every_steps=100 --stop_steps=20000 \
    --output_dir=$MODEL_DIR/ae_output \
    --save_checkpoints \
    --num_gpus=0
