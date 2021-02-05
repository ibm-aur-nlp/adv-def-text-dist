#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate
DATA_DIR=data
MODEL_DIR=saved_models

# AE+LS+CF
python train.py --do_test \
    --vocab_file=$DATA_DIR/vocab.in \
    --emb_file=$DATA_DIR/emb.json \
    --test_file=$DATA_DIR/test.in \
    --test_output=$DATA_DIR/test.out \
    --load_model=$MODEL_DIR/adv_output_lscf/nmt-T2.ckpt  \
    --adv --classification_model=RNN  --output_classes=2  \
    --gumbel_softmax_temporature=0.1  \
    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
    --cls_attention --cls_attention_size=50 --attention \
    --batch_size=16 --max_len=50 \
    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
    --output_dir=$MODEL_DIR/adv_test_output \
    --save_checkpoints \
    --num_gpus=0 \
    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
	--ae_emb_file=$DATA_DIR/cf_emb.json \
	--use_cache_dir=/dccstor/ddig/ying/use_cache  \
	--accept_name=xlnet


## AE+LS+CF+CPY
#python train.py --do_test \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --test_file=$DATA_DIR/test.in \
#    --test_output=$DATA_DIR/test.out \
#    --load_model=$MODEL_DIR/adv_output_lscf/nmt-T2.ckpt  \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --output_dir=adv_test \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json \
#	--use_cache_dir=/dccstor/ddig/ying/use_cache  \
#	--accept_name=xlnet  \
#	--copy --attention_copy_mask --use_stop_words --top_k_attack=9
#
# Test Conditional Generation: AE+LS+CF
#python train.py --do_test \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --test_file=$DATA_DIR/test.in \
#    --test_output=$DATA_DIR/test.out \
#    --load_model_pos=$MODEL_DIR/adv_output_lscfcp_ptn  \
#    --load_model_neg=$MODEL_DIR/adv_output_lscfcp_ntp \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --output_dir=$MODEL_DIR/adv_test_lscfcp_ptn \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	 --ae_emb_file=$DATA_DIR/cf_emb.json
#
#
### AE+LS+CF+DEFENCE
#python train.py --do_test \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --test_file=$DATA_DIR/test.in \
#    --test_output=$DATA_DIR/test.out \
#    --load_model=$MODEL_DIR/adv_output_lscf/nmt-T2.ckpt  \
#    --adv --classification_model=RNN  --output_classes=2 --defending  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --output_dir=$MODEL_DIR/adv_def_test \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json \
#	--use_cache_dir=/dccstor/ddig/ying/use_cache  \
#	--accept_name=xlnet


