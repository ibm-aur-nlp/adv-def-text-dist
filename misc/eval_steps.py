"""
Evaluation for adversarial attacks
Author:         Ying Xu
Date:           Jul 8, 2020
"""

from misc import evaluate, utils, input_data
from misc.evaluate_attacks import evaluate_attack
import numpy as np
import random
import tensorflow as tf
from timeit import default_timer

def get_copy_mask(sess, model, input_batch, batch_max_length, n_top_k):
    copy_masks = []
    input_batch = input_batch  + (np.ones(np.shape(input_batch[2])), )
    importance_scores = sess.run(model.importance_score, feed_dict=model.make_train_inputs(input_batch))
    if n_top_k is not None:
        for important_score_index in np.argsort(-importance_scores, axis=-1, ):
            top_k_index = important_score_index[:n_top_k]
            mask = np.ones(batch_max_length, dtype=np.float)
            mask[top_k_index] = 0.0
            copy_masks.append(mask)
        copy_masks = np.array(copy_masks)
    else:
        copy_masks = (importance_scores == 0.0).astype(np.float)
    return copy_masks


def printSentence(tokenized_sentences, vocab):
    train_sentence = ''
    for word in tokenized_sentences:
        train_sentence += vocab[word] + ' '
    utils.print_out(train_sentence)

def getSentencesFromIDs(tokenized_sentences_list, vocab, eos_id=input_data.EOS_ID):
    train_sentences = []
    for tokenized_sentences in tokenized_sentences_list:
        train_sentence = []
        for word in tokenized_sentences:
            if word == eos_id:
                break
            train_sentence.append(vocab[word])
        train_sentences.append(train_sentence)
    return train_sentences

def padding(decoder_outputs, eos_id=input_data.EOS_ID):
    ret = []
    for example in decoder_outputs:
        example_list = example.tolist()
        if eos_id in example_list:
            eos_index = example_list.index(eos_id)
            example[eos_index:] = eos_id
        ret.append(example)
    return np.array(ret)

def get_lengths(outputs, eos_id=input_data.EOS_ID):
    lengths = []
    for example in outputs:
        example_list = example.tolist()
        if eos_id in example_list:
            lengths.append(example_list.index(eos_id) + 1)
        else:
            lengths.append(len(example_list))
    return np.array(lengths, dtype=np.int32)

def lookup(input, vocab_map):
    train_sentences = []
    for ind, tokenized_sentences in enumerate(input):
        train_sentence = [vocab_map[word] for word in tokenized_sentences]
        train_sentences.append(np.array(train_sentence))
    train_sentences = np.array(train_sentences)
    return train_sentences

def run_adv(args, model_dev, sess, dev_iter, dev_next):
    sess.run(dev_iter.initializer)
    cls_labels = []
    decoder_reference_list = []
    decoder_prediction_list = []
    copy_masks = []
    i = 0
    while True:
        try:
            dev_batch = sess.run(dev_next)
            if args.copy:
                copy_mask = get_copy_mask(sess, model_dev, dev_batch, np.max(dev_batch[5]), args.top_k_attack)
                dev_batch = dev_batch + (copy_mask,)

            results = sess.run(model_dev.make_infer_outputs(),
                               feed_dict=model_dev.make_train_inputs(dev_batch))
            cls_labels.extend(dev_batch[3])

            if args.beam_width > 0:
                decoder_outputs = results[0][0]
            else:
                decoder_outputs = results[0]

            if args.ae_vocab_file is not None:
                decoder_outputs = lookup(decoder_outputs, args.vocab_map)

            decoder_reference_list.extend(dev_batch[2])

            if args.copy:
                copy_masks.extend(results[1])

            decoder_prediction_list.extend(decoder_outputs)
            # if i >= 1:
            #     break
            i += 1
        except tf.errors.OutOfRangeError:
            break
    return decoder_reference_list, decoder_prediction_list, cls_labels, copy_masks


def run_classifications(args, sess, dev_iter, decoder_prediction_list, model_classifier, dev_next):
    sess.run(dev_iter.initializer)
    cls_logits_def = []
    cls_origs_def = []
    cls_logits = []
    cls_orig_logits = []
    sent_embs = []
    adv_sent_embs = []
    orig_alphas, trans_alphas = [], []
    trans_alphas_def = []
    i = 0
    while True:
        try:
            dev_batch = sess.run(dev_next)

            decoder_outputs = np.array(decoder_prediction_list[i*args.batch_size: (i+1)*args.batch_size])

            decoder_preds_lengths = get_lengths(decoder_outputs, eos_id=102 if args.classification_model == 'BERT' else input_data.EOS_ID)

            # classification based on decoder_predictions_inference
            cls_logit = sess.run(model_classifier.make_classifier_outputs(),
                                 feed_dict=model_classifier.make_classifier_input(dev_batch,
                                                                                  decoder_outputs,
                                                                                  decoder_preds_lengths))
            cls_logits.extend(cls_logit[0])
            adv_sent_embs.extend(cls_logit[1])
            if args.cls_attention:
                trans_alphas.extend(cls_logit[2])

            # classification based on orginal input
            cls_orig_logit = sess.run(model_classifier.make_classifier_outputs(),
                                      feed_dict=model_classifier.make_classifier_input(dev_batch,
                                                                                       dev_batch[2], dev_batch[5]))
            cls_orig_logits.extend(cls_orig_logit[0])
            sent_embs.extend(cls_orig_logit[1])
            if args.cls_attention:
                orig_alphas.extend(cls_orig_logit[2])

            # defending classification based on decoder_predictions_inference
            if args.defending:
                cls_logit_def = sess.run(model_classifier.make_def_classifier_outputs(),
                                         feed_dict=model_classifier.make_classifier_input(dev_batch,
                                                                                          decoder_outputs,
                                                                                          decoder_preds_lengths))
                cls_logits_def.extend(cls_logit_def[0])
                if args.cls_attention:
                    trans_alphas_def.extend(cls_logit_def[2])

                cls_orig_def = sess.run(model_classifier.make_def_classifier_outputs(),
                                        feed_dict=model_classifier.make_classifier_input(dev_batch,
                                                                                         dev_batch[2],
                                                                                         dev_batch[5]))
                cls_origs_def.extend(cls_orig_def[0])

            # if i >=1:
            #     break
            i += 1
        except tf.errors.OutOfRangeError:
            break

    return cls_logits_def, cls_origs_def, cls_logits, cls_orig_logits, sent_embs, adv_sent_embs, \
           orig_alphas, trans_alphas, trans_alphas_def


def eval_adv(args, sess, dev_iter, model_dev, model_classifier, dev_next, vocab,
             step, demo_per_step, is_train=True):
    sess.run(dev_iter.initializer)
    # sentiment_distances = []
    cls_labels = []
    cls_logits_def = []
    cls_origs_def = []
    cls_logits = []
    cls_orig_logits = []
    decoder_reference_list = []
    decoder_prediction_list = []
    sent_embs = []
    adv_sent_embs = []
    orig_alphas, trans_alphas = [], []
    trans_alphas_def = []
    copy_masks = []
    i = 0

    start = default_timer()
    while True:
        try:
            dev_batch = sess.run(dev_next)
            if args.copy:
                copy_mask = get_copy_mask(sess, model_dev, dev_batch, np.max(dev_batch[5]), args.top_k_attack)
                dev_batch = dev_batch +  (copy_mask,)
            results = sess.run(model_dev.make_infer_outputs(),
                                feed_dict=model_dev.make_train_inputs(dev_batch))
            cls_labels.extend(dev_batch[3])


            if args.beam_width > 0:
                decoder_outputs = results[0][0]
            else:
                decoder_outputs = results[0]

            if args.ae_vocab_file is not None:
                decoder_outputs = lookup(decoder_outputs, args.vocab_map)

            decoder_reference_list.extend(dev_batch[2])

            if args.copy:
                copy_masks.extend(results[1])

            decoder_prediction_list.extend(decoder_outputs)

            decoder_preds_lengths = get_lengths(decoder_outputs, eos_id=102 if args.classification_model == 'BERT' else input_data.EOS_ID)
            decoder_outputs = padding(decoder_outputs, eos_id=102 if args.classification_model == 'BERT' else input_data.EOS_ID)

            if args.classification_model == 'BERT':
                decoder_outputs = np.concatenate([np.array([[101]] * len(decoder_outputs)), decoder_outputs], axis=1)

            # classification based on decoder_predictions_inference
            cls_logit = sess.run(model_classifier.make_classifier_outputs(),
                    feed_dict=model_classifier.make_classifier_input(dev_batch,
                    decoder_outputs, decoder_preds_lengths))
            cls_logits.extend(cls_logit[0])
            adv_sent_embs.extend(cls_logit[1])
            if args.cls_attention:
                trans_alphas.extend(cls_logit[2])

            # classification based on orginal input
            cls_orig_logit = sess.run(model_classifier.make_classifier_outputs(),
                    feed_dict=model_classifier.make_classifier_input(dev_batch,
                    dev_batch[2], dev_batch[5]))
            cls_orig_logits.extend(cls_orig_logit[0])
            sent_embs.extend(cls_orig_logit[1])
            if args.cls_attention:
                orig_alphas.extend(cls_orig_logit[2])

            # defending classification based on decoder_predictions_inference
            if args.defending:
                cls_logit_def = sess.run(model_classifier.make_def_classifier_outputs(),
                        feed_dict=model_classifier.make_classifier_input(dev_batch,
                        decoder_outputs, decoder_preds_lengths))
                cls_logits_def.extend(cls_logit_def[0])
                if args.cls_attention:
                    trans_alphas_def.extend(cls_logit_def[2])

                cls_orig_def = sess.run(model_classifier.make_def_classifier_outputs(),
                                         feed_dict=model_classifier.make_classifier_input(dev_batch,
                                                                                          dev_batch[2],
                                                                                          dev_batch[5]))
                cls_origs_def.extend(cls_orig_def[0])

            if is_train and i >= 30:
            # if i >= 0:
                break
            i += 1
        except tf.errors.OutOfRangeError:
            break
    end = default_timer()
    if not is_train:
        utils.print_out('Adversarial attack elapsed:' + '{0:.4f}'.format(end - start) + 's')

    cls_acc, cls_acc_pos, cls_acc_neg, changed_bleu = evaluate_attack(args, step, decoder_reference_list, decoder_prediction_list,
                    cls_logits, cls_orig_logits, cls_labels, vocab,
                    sent_embs, adv_sent_embs,
                    is_test=(not is_train), orig_alphas=orig_alphas, trans_alphas=trans_alphas,
                    cls_logits_def=cls_logits_def, cls_origs_def=cls_origs_def,
                    copy_masks=copy_masks)

    return cls_acc, cls_acc_pos, cls_acc_neg, changed_bleu


def run_classification(args, model, vocab, sess, dev_iter, dev_next):
    sess.run(dev_iter.initializer)
    dev_srcs = []
    dev_labels = []
    dev_logits = []
    alphas = []
    count = 0
    while True:
        try:
            dev_batch = sess.run(dev_next)

            # classification based on orginal input
            cls_orig_logit = sess.run(model.make_classifier_outputs(),
                                      feed_dict=model.make_classifier_input(dev_batch, dev_batch[2], dev_batch[5]))
            dev_srcs.append(dev_batch[2])
            dev_labels.append(dev_batch[3])
            dev_logits.append(cls_orig_logit[0])
            if args.cls_attention:
                alphas.append(cls_orig_logit[-1])
            # if count >= 1:
            #     break
            count += 1
        except tf.errors.OutOfRangeError:
            break
    reference_list = []
    alphas_list = []
    for ind, decoder_references in enumerate(dev_srcs):
        references = getSentencesFromIDs(decoder_references, vocab)
        reference_list.extend(references)
        if len(alphas) > 0:
            alphas_list.extend(alphas[ind])

    dev_logits = np.concatenate(dev_logits, axis=0)
    dev_labels = np.concatenate(dev_labels, axis=0)
    return reference_list, alphas_list, dev_logits, dev_labels

def eval_classification(args, sess, dev_iter, dev_next, model, vocab):
    sess.run(dev_iter.initializer)
    dev_srcs = []
    dev_labels = []
    dev_logits = []
    alphas = []
    count = 0
    while True:
        try:
            dev_batch = sess.run(dev_next)
            results = sess.run(model.make_test_outputs(), feed_dict=model.make_train_inputs(dev_batch))
            dev_srcs.append(results[-2])
            dev_labels.append(results[-1])
            dev_logits.append(results[1])
            if args.cls_attention:
                alphas.append(results[3])
            count += 1
        except tf.errors.OutOfRangeError:
            break
    reference_list = []
    alphas_list = []
    for ind, decoder_references in enumerate(dev_srcs):
        references = getSentencesFromIDs(decoder_references, vocab)
        reference_list.extend(references)
        if len(alphas) > 0:
            alphas_list.extend(alphas[ind])

    dev_logits = np.concatenate(dev_logits, axis=0)
    dev_labels = np.concatenate(dev_labels, axis=0)

    rand_ind = random.randint(1, len(reference_list) - 1)
    utils.print_out('src  : ' + str(reference_list[rand_ind]) + ', label: ' + str(dev_labels[rand_ind]))
    if len(alphas_list) > 0:
        utils.print_out('alpha: ' + ', '.join(["{:.3f}".format(a[0]) for a in alphas_list[rand_ind]]))

    # Print: False positives, false negatives
    fp_spl = open(args.output_dir + '/false_positive.txt', 'w')
    fn_spl = open(args.output_dir + '/false_negative.txt', 'w')
    for i in range(len(reference_list)):
        # utils.print_out('Example ' + str(i) + ': src:\t' + ' '.join(reference_list[i]) + '\t' + str(dev_labels[i]))
        # utils.print_out(' ')
        spl_predict = evaluate.max_index(dev_logits[i])
        if dev_labels[i][1] == 1:
            if spl_predict == 0:
                fn_spl.write(
                    'Example ' + str(i) + ': src:\t' + ' '.join(reference_list[i]) + '\t: ' + str(dev_labels[i]) + '\n')
                fn_spl.write(
                    'Example ' + str(i) + ': spl:\t' + ' '.join(reference_list[i]) + '\t: ' + str(dev_logits[i]) + '\n')
                fn_spl.write('\n')
        elif dev_labels[i][1] == 0:
            if spl_predict == 1:
                fp_spl.write(
                    'Example ' + str(i) + ': src:\t' + ' '.join(reference_list[i]) + '\t: ' + str(dev_labels[i]) + '\n')
                fp_spl.write(
                    'Example ' + str(i) + ': spl:\t' + ' '.join(reference_list[i]) + '\t: ' + str(dev_logits[i]) + '\n')
                fp_spl.write('\n')
    fp_spl.close()
    fn_spl.close()

    acc = evaluate._clss_accuracy(dev_labels, dev_logits)
    auc = evaluate._clss_auc(dev_labels, dev_logits)
    return auc, acc

def eval_ae(sess, dev_iter, model_dev, dev_next, vocab, step, demo_per_step):
    sess.run(dev_iter.initializer)
    decoder_reference_list = []
    decoder_prediction_list = []
    i = 0
    while True:
        try:
            dev_batch = sess.run(dev_next)
            decoder_predictions_inference = sess.run(model_dev.make_infer_outputs(),
                                                        feed_dict=model_dev.make_train_inputs(dev_batch))
            decoder_reference_list.append(dev_batch[2])
            decoder_prediction_list.append(decoder_predictions_inference)
            if i >= 30:
                break
            i+=1
        except tf.errors.OutOfRangeError:
            break
    reference_list, translation_list = [], []
    for decoder_references in decoder_reference_list:
        references = getSentencesFromIDs(decoder_references, vocab)
        reference_list.extend(references)
    for decoder_predictions in decoder_prediction_list:
        translations = getSentencesFromIDs(decoder_predictions, vocab)
        translation_list.extend(translations)
    if demo_per_step == 1:
        for i in range(len(reference_list)):
            utils.print_out('Example ' + str(i) + ': src: ' + ' '.join(reference_list[i]))
            utils.print_out('Example ' + str(i) + ': nmt: ' + ' '.join(translation_list[i]))
    elif step % demo_per_step == 0:
        rand_ind = random.randint(1, len(reference_list)-1)
        utils.print_out('Step: ' + str(step) + ', src: ' + ' '.join(reference_list[rand_ind]))
        utils.print_out('Step: ' + str(step) + ', nmt: ' + ' '.join(translation_list[rand_ind]))
        # utils.print_out('Step' + str(step) + ', copy: ' + ' '.join([str(a) for a in copy_mask_list[rand_ind]]))
        # utils.print_out('Step' + str(step) + ', logits: ' + ' '.join([str(a) for a in decoder_logits_list[rand_ind]]))

    acc = evaluate._accuracy(reference_list, translation_list)
    word_acc = evaluate._word_accuracy(reference_list, translation_list)
    rouge = evaluate._rouge(reference_list, translation_list)
    bleu = evaluate._bleu(reference_list, translation_list)
    return acc, word_acc, rouge, bleu