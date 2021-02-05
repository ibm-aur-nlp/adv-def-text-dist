#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate
DATA_DIR=data
MODEL_DIR=saved_models

python train.py --do_test \
    --vocab_file=$DATA_DIR/vocab.in \
    --emb_file=$DATA_DIR/emb.json \
    --test_file=$DATA_DIR/test.in \
    --test_output=$DATA_DIR/test.out \
    --load_model=$MODEL_DIR/cls_output/bi_att  \
    --classification --classification_model=RNN --output_classes=2 \
    --enc_type=bi --enc_num_units=256 --cls_attention --cls_attention_size=50 \
    --learning_rate=0.001 --batch_size=32 --max_len=50 \
    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
    --output_dir=$MODEL_DIR/cls_output_test \
    --save_checkpoints \
    --num_gpus=0


## Test against augmented classifier from the AE+LS+CF
#python train.py --do_test \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --test_file=$DATA_DIR/test.in \
#    --test_output=$DATA_DIR/test.out \
#    --load_model=$MODEL_DIR/adv_output_lscf/nmt-T2.ckpt  \
#    --classification --classification_model=RNN --output_classes=2 \
#    --enc_type=bi --enc_num_units=256 --cls_attention --cls_attention_size=50 \
#    --learning_rate=0.001 --batch_size=32 --max_len=50 \
#    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
#    --output_dir=$MODEL_DIR/cls_output_test \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --use_defending_as_target

