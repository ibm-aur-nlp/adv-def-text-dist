#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate
DATA_DIR=data
MODEL_DIR=saved_models

# AE+bal
python train.py --do_train \
    --vocab_file=$DATA_DIR/vocab.in \
    --emb_file=$DATA_DIR/emb.json \
    --input_file=$DATA_DIR/train.in \
    --output_file=$DATA_DIR/train.out \
    --dev_file=$DATA_DIR/dev.in \
    --dev_output=$DATA_DIR/dev.out \
    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att \
    --adv --classification_model=RNN  --output_classes=2 --balance \
    --gumbel_softmax_temporature=0.1  \
    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
    --cls_attention --cls_attention_size=50 --attention \
    --batch_size=16 --max_len=50 \
    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
    --learning_rate=0.0001 --ae_lambda=0.2 --seq_lambda=0.7  \
    --output_dir=$MODEL_DIR/adv_train_bal \
    --save_checkpoints \
    --num_gpus=0


## AE+LS
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_ls \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95
#
#
#
## AE+LS+GAN
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_lsgan \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95 \
#    --gan --at_steps=2
#
#
## AE+LS+CF
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_lscf \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json
#
## AE+LS+CF+CPY
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_lscfcp \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json \
#	--copy --attention_copy_mask --use_stop_words --top_k_attack=9
#
## Conditional PTN: AE+LS+CF
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.pos.in \
#    --output_file=$DATA_DIR/train.pos.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_lscfcp_ptn \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	 --ae_emb_file=$DATA_DIR/cf_emb.json \
#    --target_label=0
#
## Conditional NTP: AE+LS+CF
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.neg.in \
#    --output_file=$DATA_DIR/train.neg.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_train_lscfcp_ntp \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	 --ae_emb_file=$DATA_DIR/cf_emb.json \
#    --target_label=1
#
# AE+LS+CF+DEFENCE
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_cls=$MODEL_DIR/yelp50_x3-cls/bi_att  \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0 \
#    --output_dir=$MODEL_DIR/def_train \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json \
#    --defending --at_steps=2
#
#
## Attacking an augmented AE+LS+CF model: AE+LS+CF
#python train.py --do_train \
#    --vocab_file=$DATA_DIR/vocab.in \
#    --emb_file=$DATA_DIR/emb.json \
#    --input_file=$DATA_DIR/train.in \
#    --output_file=$DATA_DIR/train.out \
#    --dev_file=$DATA_DIR/dev.in \
#    --dev_output=$DATA_DIR/dev.out \
#    --load_model_ae=$MODEL_DIR/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=$MODEL_DIR/adv_aeaug_lscf \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=$DATA_DIR/cf_vocab.in  \
#	--ae_emb_file=$DATA_DIR/cf_emb.json \
#    --load_model_cls=$MODEL_DIR/def_output/nmt-T2.ckpt  \
#    --use_defending_as_target
