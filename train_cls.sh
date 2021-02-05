#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate
DATA_DIR=data
MODEL_DIR=saved_models

# RNN train
python train.py --do_train \
    --vocab_file=$DATA_DIR/vocab.in \
    --emb_file=$DATA_DIR/emb.json \
    --input_file=$DATA_DIR/train.in \
    --output_file=$DATA_DIR/train.out \
    --dev_file=$DATA_DIR/dev.in \
    --dev_output=$DATA_DIR/dev.out \
    --classification --classification_model=RNN --output_classes=2 \
    --enc_type=bi --enc_num_units=256 --cls_attention --cls_attention_size=50 \
    --learning_rate=0.001 --batch_size=32 --max_len=50 \
    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
    --output_dir=$MODEL_DIR/cls_output_rnn \
    --save_checkpoints \
    --num_gpus=0

## CNN train
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --classification --classification_model=CNN --output_classes=2 \
#    --enc_type=bi --enc_num_units=256 --cls_attention_size=50 \
#    --learning_rate=0.001 --batch_size=32 --max_len=50 --dropout_keep_prob=0.8 \
#    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
#    --output_dir=$MODEL_DIR/cls_output_cnn \
#    --save_checkpoints \
#    --num_gpus=0

## BERT train
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/data/bert/uncased_L-12_H-768_A-12/vocab.txt \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --classification --classification_model=BERT --output_classes=2 \
#    --bert_config_file=/Users/yxu132/data/bert/uncased_L-12_H-768_A-12/bert_config.json \
#    --bert_init_chk=/Users/yxu132/data/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
#    --learning_rate=1e-5 --batch_size=32 --max_len=50 \
#    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
#    --output_dir=$MODEL_DIR/cls_output_bert \
#    --save_checkpoints \
#    --num_gpus=0