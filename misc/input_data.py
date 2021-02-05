"""
Input data stream-reading
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
import json
import codecs
import numpy as np
import os
from misc import iterator
import misc.utils as utils

VOCAB_SIZE_THRESHOLD_CPU = 50000
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2
SEP_ID = 3

def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None, vocab_base_name=None):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    print("# Vocab file "+vocab_file+" exists")
    vocab, vocab_size = load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      if not unk: unk = UNK
      if not sos: sos = SOS
      if not eos: eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3
        new_vocab_file = os.path.join(out_dir, vocab_base_name)
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size, vocab_file

def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size

def load_embed_json(embed_file, vocab_size=None, dtype=tf.float32):
  with codecs.open(embed_file, "r", "utf-8") as fh:
    emb_dict = json.load(fh)
  emb_size = len(emb_dict[0])
  emb_mat = np.array(emb_dict, dtype=dtype.as_numpy_dtype())
  if vocab_size > len(emb_mat):
      np.random.seed(0)
      emb_mat_var = np.random.rand(vocab_size-len(emb_mat), emb_size)
      emb_mat_var = np.array(emb_mat_var, dtype=dtype.as_numpy_dtype())
      emb_mat = np.concatenate([emb_mat_var, emb_mat], axis=0)
  return emb_mat, emb_size

def _load_simlex(lex_file, vocab_size):
    lexis = utils.readlines(lex_file)
    sim_num = int(lex_file.split('-')[-2])+1
    base = 0
    if vocab_size > len(lexis):
        base = 3
        lexis = np.concatenate([['']*base, lexis], axis=0)
    lex = []
    for ind, line in enumerate(lexis):
        if line == '':
            lex.append(np.array([ind]+[-1]*(sim_num-1)))
        else:
            comps = line.split(' ')
            lex.append(np.array([int(a)+base for a in comps]+[ind]+[-1]*(sim_num-(len(comps)+1))))
    return np.array(lex)

def _get_embed_device(vocab_size, num_gpus):
  """Decide on which device to place an embed matrix given its vocab size."""
  if num_gpus == 0 or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"

def _create_pretrained_embeddings_from_jsons(
        vocab_file, embed_file, cls_vocab_file, cls_embed_file,
        dtype=tf.float32, cls_model='RNN'):

    vocab, _ = load_vocab(vocab_file)
    emb_mat, emb_size = load_embed_json(embed_file, vocab_size=len(vocab), dtype=dtype)

    cls_vocab, _ = load_vocab(cls_vocab_file)
    cls_emb_mat, cls_emb_size = load_embed_json(cls_embed_file, vocab_size=len(cls_vocab), dtype=dtype)

    transfer_emb_mat = []
    unk_id = 100 if cls_model == 'BERT' else UNK_ID
    if cls_model == 'BERT':
        for word in vocab:
            if word == '<sos>':
                transfer_emb_mat.append(cls_emb_mat[cls_vocab.index('[CLS]')])
            elif word == '<eos>':
                transfer_emb_mat.append(cls_emb_mat[cls_vocab.index('[SEP]')])
            elif word == '<unk>':
                transfer_emb_mat.append(cls_emb_mat[cls_vocab.index('[UNK]')])
            elif word in cls_vocab:
                transfer_emb_mat.append(cls_emb_mat[cls_vocab.index(word)])
            else:
                transfer_emb_mat.append(cls_emb_mat[unk_id])
    else:
        for word in vocab:
            if word in cls_vocab:
                transfer_emb_mat.append(cls_emb_mat[cls_vocab.index(word)])
            else:
                transfer_emb_mat.append(cls_emb_mat[unk_id])

    # with tf.device("/cpu:0"):
    cls_emb_mat = tf.constant(cls_emb_mat)
    emb_mat = tf.constant(emb_mat)
    transfer_emb_mat = tf.constant(np.array(transfer_emb_mat))
    return cls_emb_mat, emb_mat, transfer_emb_mat

def _create_pretrained_emb_from_txt(
        vocab_file, embed_file, trainable_tokens=3, dtype=tf.float32):

    vocab, _ = load_vocab(vocab_file)
    emb_mat, emb_size = load_embed_json(embed_file, vocab_size=len(vocab), dtype=dtype)

    emb_mat = tf.constant(emb_mat)
    return emb_mat


def get_labels(args):
    output_labels = []
    for line in open(args.output_file, 'r'):
        output_labels.append(int(line.strip()))
    return output_labels

def get_dataset_iter(args, input_file, output_file, task, is_training=True, is_test=False, is_bert=False):
    unk = UNK_ID
    if is_bert:
        unk = 100
    table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=args.vocab_file, default_value=unk)
    src_dataset = tf.data.TextLineDataset(tf.gfile.Glob(input_file))

    if len(args.def_train_set) > 0:
        src_datasets = [src_dataset]
        base_name = os.path.basename(args.input_file) if is_training else os.path.basename(args.dev_file)
        path_to_base_name = args.input_file[:args.input_file.rfind('/')] if is_training else args.dev_file[:args.dev_file.rfind('/')]

        for set_name in args.def_train_set:
            base_names = base_name.split('.')
            base_names.insert(-1, set_name)
            base_name_1 = '.'.join(base_names)
            full_name = path_to_base_name+'/'+base_name_1
            src_datasets.append(tf.data.TextLineDataset(tf.gfile.Glob(full_name)))

        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(output_file))
        if len(src_datasets) > 2:
            iter = iterator.get_cls_multi_def_iterator(src_datasets, tgt_dataset, table, args.batch_size, args.num_epochs,
                                                 SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
                                                 args.random_seed, args.num_buckets,
                                                 src_max_len=args.max_len, is_training=is_training)
        else:
            iter = iterator.get_cls_def_iterator(src_datasets, tgt_dataset, table, args.batch_size, args.num_epochs,
                                         SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
                                         args.random_seed, args.num_buckets,
                                         src_max_len=args.max_len, is_training=is_training)
        return iter

    if task=='adv' or task=='ae':
        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(output_file))
        iter = iterator.get_adv_iterator(src_dataset, tgt_dataset, table, args.batch_size, args.num_epochs,
                                         SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
                                         SEP if (not is_bert) else '[SEP]',
                                         args.random_seed, args.num_buckets,
                                         src_max_len=args.max_len, is_training=is_training)
        return iter

    if task=='clss':
        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(output_file))
        iter = iterator.get_cls_iterator(src_dataset, tgt_dataset, table, args.batch_size, args.num_epochs,
                                         SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
                                         SEP if (not is_bert) else '[SEP]',
                                         args.random_seed, args.num_buckets,
                                         src_max_len=args.max_len, is_training=is_training)
        return iter

    if task == 'adv_counter_fitting':
        ae_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=args.ae_vocab_file, default_value=unk)
        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(output_file))
        iter = iterator.get_adv_cf_iterator(src_dataset, tgt_dataset, table, args.batch_size, args.num_epochs,
                                         SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
                                         SEP if (not is_bert) else '[SEP]',
                                         args.random_seed, args.num_buckets,
                                         src_max_len=args.max_len, is_training=is_training,
                                         ae_vocab_table=ae_table)
        return iter

    # if task=='ae':
    #     if output_file is not None:
    #         tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(output_file))
    #     else:
    #         tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(input_file))
    #     iter = iterator.get_iterator(src_dataset, tgt_dataset, table, args.batch_size, args.num_epochs,
    #                                  SOS if (not is_bert) else '[CLS]', EOS if (not is_bert) else '[SEP]',
    #                                  args.random_seed, args.num_buckets, src_max_len=args.max_len,
    #                                  tgt_max_len=args.max_len, is_training=is_training, min_len=args.tgt_min_len)
    #     return iter


def parse_generated(input_file):
    sampled, generated = [], []
    for line in open('data_gen/'+input_file+'.txt', 'r'):
        if line.startswith('Example '):
            if ' spl:' in line:
                comps = line.strip().split('\t')
                sampled.append(comps[1])
            elif ' nmt:' in line:
                comps = line.strip().split('\t')
                generated.append(comps[1])
    with open('data_gen/'+input_file+'_spl.in', 'w') as output_file:
        for sample in sampled:
            output_file.write(sample+'\n')
    with open('data_gen/'+input_file+'_dec.in', 'w') as output_file:
        for generate in generated:
            output_file.write(generate+'\n')

    with open('data_gen/'+input_file+'_acc.txt', 'w') as output_file:
        for generate in generated:
            output_file.write('yelp\t0\t\t'+generate+'\n')


if __name__ == '__main__':
    parse_generated('cls_bi_att-hinge10-lr0001-wl5-emb3-beam')
