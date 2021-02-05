"""
Parameter specification for train.py
Author:         Ying Xu
Date:           Jul 8, 2020
"""

from misc import input_data
from argparse import ArgumentParser
import os
from misc.use import USE
from transformers import XLNetTokenizer, XLNetLMHeadModel, BertTokenizer, BertForMaskedLM
import torch


def add_arguments():
    parser = ArgumentParser()

    # basic
    parser.add_argument('--do_train', action='store_true', help="do training")
    parser.add_argument('--do_test', action='store_true', help="do independent test")
    parser.add_argument('--do_cond_test', action='store_true', help="do test for conditional generation")

    parser.add_argument('--input_file', type=str, default=None, help="")
    parser.add_argument('--dev_file', type=str, default=None, help="")
    parser.add_argument('--test_file', type=str, default=None, help="")
    parser.add_argument('--vocab_file', type=str, default=None, help="")
    parser.add_argument('--emb_file', type=str, default=None, help="")
    parser.add_argument('--output_dir', type=str, default=None, help="")
    parser.add_argument('--attention', action='store_true', help='whether use attention in seq2seq')
    parser.add_argument('--cls_attention', action='store_true', help="")
    parser.add_argument('--cls_attention_size', type=int, default=300, help="")

    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help="")
    parser.add_argument('--num_epochs', type=int, default=5, help="")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="")
    parser.add_argument('--enc_type', type=str, default='bi', help="")
    parser.add_argument('--enc_num_units', type=int, default=512, help="")
    parser.add_argument('--enc_layers', type=int, default=2, help="")
    parser.add_argument('--dec_num_units', type=int, default=512, help="")
    parser.add_argument('--dec_layers', type=int, default=2, help="")
    parser.add_argument('--epochs', type=int, default=2, help="")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument('--max_to_keep', type=int, default=5, help="")
    parser.add_argument('--lowest_bound_score', type=float, default=10.0, help="Stop the training once achieving the lowest_bound_score")

    parser.add_argument('--beam_width', type=int, default=0, help="")
    parser.add_argument("--num_buckets", type=int, default=5, help="Put data into similar-length buckets.")
    parser.add_argument("--max_len", type=int, default=50, help="Lenth max of input sentences")
    parser.add_argument('--tgt_min_len', type=int, default=0, help='Length min of target sentences')

    # training control
    parser.add_argument('--print_every_steps', type=int, default=1, help="")
    parser.add_argument('--save_every_epoch', type=int, default=1, help="")
    parser.add_argument('--stop_steps', type=int, default=20000, help="number of steps of non-improve to terminate training")
    parser.add_argument('--total_steps', type=int, default=None, help="total number of steps for training")
    parser.add_argument('--random_seed', type=int, default=1, help="")
    parser.add_argument('--num_gpus', type=int, default=0, help="")
    parser.add_argument('--save_checkpoints', action='store_true', help='Whether save models while training')

    # classification
    parser.add_argument('--classification', action='store_true', help="Perform classification")
    parser.add_argument('--classification_model', type=str, default='RNN', help='')
    parser.add_argument('--output_classes', type=int, default=2, help="number of classes for classification")
    parser.add_argument('--output_file', type=str, default=None, help="Classification output for train set")
    parser.add_argument('--dev_output', type=str, default=None, help="Classification output for dev set")
    parser.add_argument('--test_output', type=str, default=None, help="Classification output for test set")
    parser.add_argument('--filter_sizes', nargs='+', default=[5, 3], type=int, help='filter sizes, only for CNN')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help='dropout, only for CNN')
    parser.add_argument('--bert_config_file', type=str, default=None, help='pretrained bert config file')
    parser.add_argument('--bert_init_chk', type=str, default=None, help='checkpoint for pretrained Bert')

    # adversarial attack and defence
    parser.add_argument('--adv', action='store_true', help="Perform adversarial attack training/testing")
    parser.add_argument('--cls_enc_type', type=str, default='bi', help="")
    parser.add_argument('--cls_enc_num_units', type=int, default=256, help="")
    parser.add_argument('--cls_enc_layers', type=int, default=2, help="")
    parser.add_argument('--gumbel_softmax_temporature', type=float, default=0.1, help="")
    parser.add_argument('--load_model_cls', type=str, default=None, help="Path to target classification model")
    parser.add_argument('--load_model_ae', type=str, default=None, help="Path to pretrained AE")
    parser.add_argument('--load_model', type=str, default=None, help="Trained model for testing")
    parser.add_argument('--load_model_pos', type=str, default=None, help="PTN attack model for testing")
    parser.add_argument('--load_model_neg', type=str, default=None, help="NTP attack model for testing")


    # balanced attack
    parser.add_argument('--balance', action='store_true', help="Whether balance between pos/neg attack")
    # label smoothing
    parser.add_argument('--label_beta', type=float, default=None, help='label smoother param, must be > 0.5')
    # use counter-fitted embedding for AE (AE embedding different from CLS embeddings)
    parser.add_argument('--ae_vocab_file', type=str, default=None, help='Path to counter-fitted vocabulary')
    parser.add_argument('--ae_emb_file', type=str, default=None, help='Path to counter-fitted embeddings')
    # gan auxiliary loss
    parser.add_argument('--gan', action='store_true', help='Whether use GAN as regularization')
    # conditional generation (1 or 0)
    parser.add_argument('--target_label', type=int, default=None, help="Target label for conditional generation, 0 (PTN) or 1 (NTP)")
    # include defending
    parser.add_argument('--defending', action='store_true', help="whether train C* for more robust classification models")
    # train defending classifier with augmented dataset
    parser.add_argument('--def_train_set', nargs='+', default=[], type=str, help='Set of adversarial examples to include in adv training')
    # attack an AE model using the augmented classifier as the target classifier
    parser.add_argument('--use_defending_as_target', action='store_true', help='Use the defending component as the target classifier')

    # loss control
    parser.add_argument('--at_steps', type=int, default=1, help='Alternative steps for GAN/Defending')
    parser.add_argument('--ae_lambda', type=float, default=0.8, help='weighting ae_loss+sent_loss v.s. adv_loss')
    parser.add_argument('--seq_lambda', type=float, default=1.0, help='weighting ae_loss v.s. sent_loss')
    parser.add_argument('--aux_lambda', type=float, default=1.0, help='weighting ae_loss v.s. auxiliary losses')
    parser.add_argument('--sentiment_emb_dist', type=str, default='avgcos', help="whether involve embedding distance as aux loss")
    parser.add_argument('--loss_attention', action='store_true', help="whether weight emb dist")
    parser.add_argument('--loss_attention_norm', action='store_true', help="whether apply minimax norm to ae_loss_attention")

    # copy mechanism
    parser.add_argument('--copy', action='store_true', help="Whether use copy mechanism")
    parser.add_argument('--attention_copy_mask', action='store_true', help="Whether use attention to calculate copy mask")
    parser.add_argument('--use_stop_words', action='store_true', help="whether mask stop words")
    parser.add_argument('--top_k_attack', type=int, default=None, help="number of words to attack in copy mechanism, only set when args.copy is set to true.")
    parser.add_argument('--load_copy_model', type=str, default=None, help="Pretrained attention layer from the bi_att model")

    # evaluation options
    parser.add_argument('--use_cache_dir', type=str, default=None, help='cache dir for use (sem) eval')
    parser.add_argument('--accept_name', type=str, default=None, help="model name for acceptibility scores (xlnet), only used when set")


    args=parser.parse_args()
    if args.save_checkpoints and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    vocab_size, vocab_file = input_data.check_vocab(args.vocab_file, args.output_dir, check_special_token=False if (args.classification_model == 'BERT') else True,
                                                    vocab_base_name='vocab.txt')
    args.vocab_file = vocab_file
    args.vocab_size = vocab_size

    if args.ae_vocab_file is not None:
        ae_vocab_size, ae_vocab_file = input_data.check_vocab(args.ae_vocab_file, args.output_dir, check_special_token=False if (args.classification_model == 'BERT') else True,
                                                    vocab_base_name='ae_vocab.txt')
        args.ae_vocab_size = ae_vocab_size
        args.ae_vocab_file = ae_vocab_file

    args.use_model = None
    if args.use_cache_dir is not None:
        args.use_model = USE(args.use_cache_dir)

    if args.accept_name is not None:
        if args.accept_name == 'bert':
            args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            args.acpt_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        elif args.accept_name == 'xlnet':
            args.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            args.acpt_model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

        args.device = torch.device('cpu') if args.num_gpus == 0 else torch.device('cuda:0')
        args.acpt_model.to(args.device)
        args.acpt_model.eval()

    return args


