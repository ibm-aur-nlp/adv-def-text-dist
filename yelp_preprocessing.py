"""
Dataset proprocessing.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

from __future__ import unicode_literals
from argparse import ArgumentParser

import json
from tqdm import tqdm
from collections import Counter
import spacy
nlp = spacy.blank("en")
GLOVE_WORD_SIZE = int(2.2e6)
CF_WORD_SIZE = 65713

parser = ArgumentParser()
parser.add_argument('--data_dir', default='/Users/yxu132/Downloads/yelp_dataset', type=str, help='path to DATA_DIR')
parser.add_argument('--embed_file', default='/Users/yxu132/pub-repos/decaNLP/embeddings/glove.840B.300d.txt', type=str, help='path to glove embeding file')
parser.add_argument('--para_limit', default=50, type=int, help='maximum number of words for each paragraph')
args = parser.parse_args()

def parse_json():
    texts = []
    ratings = []
    for line in open(args.data_dir+'/yelp_academic_dataset_review.json', 'r'):
    # for line in open(args.data_dir + '/sample.json', 'r'):
        example = json.loads(line)
        texts.append(example['text'].replace('\n', ' ').replace('\r', ''))
        ratings.append(example['stars'])
    with open(args.data_dir+'/yelp_review.full', 'w') as output_file:
        output_file.write('\n'.join(texts))
    with open(args.data_dir+'/yelp_review.ratings', 'w') as output_file:
        output_file.write('\n'.join([str(rating) for rating in ratings]))

def readLinesList(filename):
    ret = []
    for line in open(filename, 'r'):
        ret.append(line.strip())
    return ret

def read_lines():
    ret = []
    labels = readLinesList(args.data_dir+'/yelp_review.ratings')
    for ind, line in tqdm(enumerate(open(args.data_dir+'/yelp_review.full', 'r'))):
        line = line.strip().lower()
        line = line.replace('\\n', ' ').replace('\\', '')
        line = line.replace('(', ' (').replace(')', ') ')
        line = line.replace('!', '! ')
        line = ' '.join(line.split())
        example = {}
        example['text'] = line
        example['label'] = labels[ind]
        ret.append(example)
    return ret

def get_tokenize(sent):
    sent = sent.replace(
        "''", '" ').replace("``", '" ')
    doc = nlp(sent)
    context_tokens = [token.text for token in doc]
    new_sent = ' '.join(context_tokens)
    return new_sent, context_tokens

def tokenize_sentences(sentences, para_limit=None):
    print('Tokenize input sentences...')
    word_counter = Counter()
    context_list, context_tokens_list = [], []
    labels = []
    for sentence in tqdm(sentences):
        context, context_tokens = get_tokenize(sentence['text'])
        if len(context_tokens) > para_limit:
            continue
        for token in context_tokens:
            word_counter[token] += 1
        context_list.append(context)
        context_tokens_list.append(context_tokens)
        labels.append(sentence['label'])
    return context_list, context_tokens_list, labels, word_counter

def filter_against_embedding(sentences, counter, emb_file, limit=-1,
                             size=GLOVE_WORD_SIZE, vec_size=300):

    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    assert size is not None
    assert vec_size is not None
    with codecs.open(emb_file, "r", "utf-8") as fh:
        for line in tqdm(fh, total=size):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in counter and counter[word] > limit:
                embedding_dict[word] = vector
    print("{} / {} tokens have corresponding embedding vector".format(
        len(embedding_dict), len(filtered_elements)))

    embedding_tokens = set(embedding_dict.keys())
    filtered_sentences = []
    for sentence in sentences:
        tokens = sentence['text'].split()
        if len(set(tokens) - embedding_tokens) > 0:
            continue
        filtered_sentences.append(sentence)

    return filtered_sentences, embedding_dict

def writeLines(llist, output_file):
    with codecs.open(output_file, "w", "utf-8") as output:
        output.write('\n'.join(llist))

def get_embedding(counter, data_type, emb_file, limit=-1, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}

    filtered_elements = [k for k, v in counter.items() if v > limit]
    assert size is not None
    assert vec_size is not None
    with codecs.open(emb_file, "r", "utf-8") as fh:
        for line in tqdm(fh, total=size):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in counter and counter[word] > limit:
                embedding_dict[word] = vector
    missing_words = set(filtered_elements) - set(embedding_dict.keys())
    print('\n'.join(missing_words))

    print("{} / {} tokens have corresponding {} embedding vector".format(
        len(embedding_dict), len(filtered_elements), data_type))

    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 0)}

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def embed_sentences(word_counter, word_emb_file):
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=GLOVE_WORD_SIZE, vec_size=300)
    return word_emb_mat, word2idx_dict

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

import numpy as np
def process():

    print("Step 2.1: Tokenize sentences...")
    sentences = read_lines()
    context_list, context_tokens_list, labels, word_counter = \
        tokenize_sentences(sentences, para_limit=args.para_limit)
    writeLines(context_list, args.data_dir+'/yelp.in')
    writeLines(labels, args.data_dir+'/yelp.out')

    print("\nStep 2.2: Filter dataset against glove embedding...")
    texts = readLinesList(args.data_dir+'/yelp.in')
    labels = readLinesList(args.data_dir+'/yelp.out')
    sentences = []
    for ind, text in enumerate(texts):
        sentence = {}
        sentence['text'] = text
        sentence['label'] = labels[ind]
        sentences.append(sentence)
    print('\nbefore filtering: '+str(len(sentences)))

    filtered_sentences, embed_dict = filter_against_embedding(sentences, word_counter, emb_file=args.embed_file)
    print('\nafter filtering: '+str(len(filtered_sentences)))

    texts = [sentence['text'] for sentence in filtered_sentences]
    labels = [sentence['label'] for sentence in filtered_sentences]
    writeLines(texts, args.data_dir + '/yelp_filtered.in')
    writeLines(labels,args.data_dir + '/yelp_filtered.out')


    print("\nStep 2.3: Split into train, dev and test datasets...")
    dev_test_percentage = 0.05
    sentences = []
    texts = readLinesList(args.data_dir+'/yelp_filtered.in')
    labels = readLinesList(args.data_dir+'/yelp_filtered.out')
    for ind, text in enumerate(texts):
        sentence={}
        sentence['text'] = text
        sentence['label'] = labels[ind]
        sentences.append(sentence)
    sentences = np.array(sentences)

    total = len(sentences)
    dev_test_num = int(total * dev_test_percentage)
    dev = sentences[:dev_test_num]
    test = sentences[dev_test_num: dev_test_num*2]
    train = sentences[dev_test_num*2: ]

    writeLines([sent['text'] for sent in train], args.data_dir + '/yelp_train.in')
    writeLines([sent['text'] for sent in dev], args.data_dir + '/yelp_dev.in')
    writeLines([sent['text'] for sent in test], args.data_dir + '/yelp_test.in')
    writeLines([sent['label'] for sent in train], args.data_dir + '/yelp_train.out')
    writeLines([sent['label'] for sent in dev], args.data_dir + '/yelp_dev.out')
    writeLines([sent['label'] for sent in test], args.data_dir + '/yelp_test.out')

    print("Step 2.4: Extract embeddings for filtered sentence vocabs...")

    sentences_tokens = [sent['text'].split() for sent in sentences]
    word_counter = dict()
    for sentence in sentences_tokens:
        for token in sentence:
            if token in word_counter:
                word_counter[token] = word_counter[token] + 1
            else:
                word_counter[token] = 1

    word_counter_new = sorted(word_counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    vocab_output_file = codecs.open(args.data_dir + '/vocab_count.txt', "w", "utf-8")
    for word in word_counter_new:
        vocab_output_file.write(word[0]+' '+str(word[1])+'\n')

    word_emb_mat, word2idx_dict = embed_sentences(word_counter, word_emb_file=args.embed_file)
    writeLines(word2idx_dict.keys(), args.data_dir + '/vocab.in')
    save(args.data_dir + '/emb.json', word_emb_mat, message="word embedding")

def binarise_and_balance():
    partitions = ['train', 'dev', 'test']

    for partition in partitions:
        sentences = readLinesList(args.data_dir+'/yelp_'+partition+'.in')
        pos_sents, neg_sents = [], []
        for ind, line in enumerate(open(args.data_dir+'/yelp_'+partition+'.out', 'r')):
            if line.strip() == '1.0' or line.strip() == '2.0':
                neg_sents.append(sentences[ind])
            elif line.strip() == '4.0' or line.strip() == '5.0':
                pos_sents.append(sentences[ind])

        np.random.seed(0)
        shuffled_ids = np.arange(len(pos_sents))
        np.random.shuffle(shuffled_ids)
        pos_sents = np.array(pos_sents)[shuffled_ids]

        sents = neg_sents + pos_sents.tolist()[:len(neg_sents)]
        labels = ['1.0 0.0'] * len(neg_sents) + ['0.0 1.0'] * len(neg_sents)

        shuffled_ids = np.arange(len(sents))
        np.random.shuffle(shuffled_ids)
        sents = np.array(sents)[shuffled_ids]
        labels = np.array(labels)[shuffled_ids]

        with open(args.data_dir+'/'+partition+'.in', 'w') as output_file:
            for line in sents:
                output_file.write(line+'\n')
        with open(args.data_dir+'/'+partition+'.out', 'w') as output_file:
            for line in labels:
                output_file.write(line+'\n')


###################### CF embedding ###################

import codecs
import os

def parse_cf_emb(cf_file_path):

    vocab = []
    matrix = []
    for line in tqdm(open(cf_file_path, 'r'), total=CF_WORD_SIZE):
        comps = line.strip().split()
        word = ''.join(comps[0:-300])
        vec = comps[-300:]
        vocab.append(word)
        matrix.append(vec)
    writeLines(vocab, 'embeddings/counter-fitted-vectors-vocab.txt')
    json.dump(matrix, open('embeddings/counter-fitted-vectors-emb.json', 'w'))

def transform_cf_emb():

    if not os.path.exists('embeddings/counter-fitted-vectors-vocab.txt') or \
        not os.path.exists('embeddings/counter-fitted-vectors-emb.json'):
        parse_cf_emb('embeddings/counter-fitted-vectors.txt')


    vocab = readLinesList(args.data_dir + '/vocab.txt')
    cf_vocab = readLinesList('embeddings/counter-fitted-vectors-vocab.txt')

    print('glove_vocab_size: '+str(len(vocab)))
    print('cf_vocab_size: ' + str(len(cf_vocab)))

    with codecs.open(args.data_dir + '/emb.json', "r", "utf-8") as fh:
        emb = json.load(fh)

    with codecs.open('embeddings/counter-fitted-vectors-emb.json', "r", "utf-8") as fh:
        cf_emb = json.load(fh)

    vocab_diff = []
    vocab_diff_ind = []
    for ind, word in enumerate(vocab):
        if word not in cf_vocab:
            vocab_diff.append(word)
            vocab_diff_ind.append(ind)

    print('extend_vocab_size: ' + str(len(vocab_diff_ind)))


    new_cf_vocab = cf_vocab + vocab_diff
    new_emb = cf_emb
    for ind, word in enumerate(vocab_diff):
        new_emb.append(emb[vocab_diff_ind[ind]])

    print('combined_cf_vocab_size: ' + str(len(new_emb)))

    writeLines(new_cf_vocab, args.data_dir + '/cf_vocab.in')
    json.dump(new_emb, open(args.data_dir + '/cf_emb.json', 'w'))

def split_pos_neg():

    input_sents = readLinesList(args.data_dir+'/train.in')
    labels = readLinesList(args.data_dir+'/train.out')

    pos_out_file = open(args.data_dir+'/train.pos.in', 'w')
    neg_out_file = open(args.data_dir+'/train.neg.in', 'w')
    pos_lab_file = open(args.data_dir+'/train.pos.out', 'w')
    neg_lab_file = open(args.data_dir+'/train.neg.out', 'w')

    for ind, sent in enumerate(input_sents):
        label = labels[ind]
        if label == '1.0 0.0':
            neg_out_file.write(sent+'\n')
            neg_lab_file.write(label+'\n')
        elif label == '0.0 1.0':
            pos_out_file.write(sent + '\n')
            pos_lab_file.write(label + '\n')


if __name__ == '__main__':
    print("Step 1: Parse json file...")
    parse_json()
    print("\nStep 2: Data partition/GloVe embedding extraction...")
    process()
    print("\nStep 3: Binarise and downsampling...")
    binarise_and_balance()
    print("\nStep 4: Counter-fitted embedding extraction...")
    transform_cf_emb()
    print("\nStep 5: Split train set into pos/neg examples (for conditional generation only)...")
    split_pos_neg()


