"""
Adversarial attack evaluations.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import numpy as np
from misc import evaluate as general_evaluate
from misc import utils
from misc import input_data

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


def read_false_record(file_name):
    ret = []
    for line in open(file_name, 'r'):
        if line.startswith('Example '):
            comps = line.strip().split(': ')
            example_id = comps[0].split(' ')[1]
            ret.append(int(example_id))
    return ret

from sklearn.metrics.pairwise import cosine_similarity
def avgcos(emb1, emb2):
    avg_emb1 = [np.mean(a, axis=0) for a in emb1]
    avg_emb2 = [np.mean(a, axis=0) for a in emb2]
    cos_sim = [cosine_similarity([a], [b])[0][0] for a, b in zip(avg_emb1, avg_emb2)]
    avg_cos = sum(cos_sim) / len(cos_sim)
    return avg_cos

def evaluate_attack(args, step, decoder_reference_list, decoder_prediction_list,
                    cls_logits, cls_orig_logits, cls_labels, vocab,
                    sent_embs, adv_sent_embs,
                    is_test=False, X_adv_flip_num=None,
                    orig_alphas=None, trans_alphas=None,
                    cls_logits_def=None, cls_origs_def=None,
                    copy_masks=None):

    cls_orig_acc = general_evaluate._clss_accuracy(cls_labels, cls_orig_logits)
    cls_orig_auc = general_evaluate._clss_auc(cls_labels, cls_orig_logits)

    cls_acc = general_evaluate._clss_accuracy(cls_labels, cls_logits)
    cls_auc = general_evaluate._clss_auc(cls_labels, cls_logits)
    cls_acc_pos = general_evaluate._clss_accuracy_micro(cls_labels, cls_logits, orig_label=1)
    cls_acc_neg = general_evaluate._clss_accuracy_micro(cls_labels, cls_logits, orig_label=0)

    if cls_logits_def is not None and len(cls_logits_def) > 0:
        cls_def_acc = general_evaluate._clss_accuracy(cls_labels, cls_logits_def)
        cls_def_auc = general_evaluate._clss_auc(cls_labels, cls_logits_def)
        org_def_acc = general_evaluate._clss_accuracy(cls_labels, cls_origs_def)
        org_def_auc = general_evaluate._clss_auc(cls_labels, cls_origs_def)

    reference_list = getSentencesFromIDs(decoder_reference_list, vocab)
    translation_list = getSentencesFromIDs(decoder_prediction_list, vocab)

    ref_pos, ref_neg, trans_pos, trans_neg, ref_changed, trans_changed = [], [], [], [], [], []
    label_changed, logits_changed, flip_num_changed, ids_changed = [], [], [], []
    ref_emb_pos, trans_emb_pos, ref_emb_neg, trans_emb_neg, ref_emb_cha, trans_emb_cha = [], [], [], [], [], []

    for ind, references in enumerate(reference_list):
        ref_pos.append(references) if cls_labels[ind][1] > 0 else ref_neg.append(references)
        trans_pos.append(translation_list[ind]) if cls_labels[ind][1] > 0 else trans_neg.append(translation_list[ind])
        ref_emb_pos.append(sent_embs[ind]) if cls_labels[ind][1] > 0 else ref_emb_neg.append(sent_embs[ind])
        trans_emb_pos.append(adv_sent_embs[ind]) if cls_labels[ind][1] > 0 else trans_emb_neg.append(adv_sent_embs[ind])
        if np.argmax(cls_logits[ind]) != np.argmax(cls_orig_logits[ind]):
            ids_changed.append(ind)
            ref_changed.append(references)
            trans_changed.append(translation_list[ind])
            label_changed.append(cls_labels[ind])
            logits_changed.append(cls_logits[ind])
            ref_emb_cha.append(sent_embs[ind])
            trans_emb_cha.append(adv_sent_embs[ind])
            if X_adv_flip_num is not None:
                flip_num_changed.append(X_adv_flip_num[ind])

    ae_acc = general_evaluate._accuracy(reference_list, translation_list)
    word_acc = general_evaluate._word_accuracy(reference_list, translation_list)
    rouge = general_evaluate._rouge(reference_list, translation_list)
    bleu = general_evaluate._bleu(reference_list, translation_list)
    use = general_evaluate._use_scores(reference_list, translation_list, args.use_model)
    accept = general_evaluate._accept_score(reference_list, translation_list, args)

    # positive examples
    pos_rouge = general_evaluate._rouge(ref_pos, trans_pos)
    pos_bleu = general_evaluate._bleu(ref_pos, trans_pos)
    pos_accept = general_evaluate._accept_score(ref_pos, trans_pos, args)
    pos_semsim = avgcos(ref_emb_pos, trans_emb_pos)
    pos_use = general_evaluate._use_scores(ref_pos, trans_pos, args.use_model)

    # negative examples
    neg_rouge = general_evaluate._rouge(ref_neg, trans_neg)
    neg_bleu = general_evaluate._bleu(ref_neg, trans_neg)
    neg_accept = general_evaluate._accept_score(ref_neg, trans_neg, args)
    neg_semsim = avgcos(ref_emb_neg, trans_emb_neg)
    neg_use = general_evaluate._use_scores(ref_neg, trans_neg, args.use_model)


    # changed examples
    if len(ref_changed) == 0:
        changed_rouge = -1.0
        changed_bleu = -1.0
        changed_accept = -1.0
        changed_semsim = -1.0
        changed_use = -1.0
    else:
        changed_rouge = general_evaluate._rouge(ref_changed, trans_changed)
        changed_bleu = general_evaluate._bleu(ref_changed, trans_changed)
        changed_accept = general_evaluate._accept_score(ref_changed, trans_changed, args)
        changed_semsim = avgcos(ref_emb_cha, trans_emb_cha)
        changed_use = general_evaluate._use_scores(ref_changed, trans_changed, args.use_model)
        # changed_use = 0.0

    # print out src, spl, and nmt
    for i in range(len(ref_changed)):
        reference_changed = ref_changed[i]
        translation_changed = trans_changed[i]
        if orig_alphas is not None and len(orig_alphas) > 0:
            orig_alpha = orig_alphas[ids_changed[i]]
            reference_changed = [s + '('+'{:.3f}'.format(orig_alpha[ind][0])+')' for ind, s in enumerate(ref_changed[i])]
            trans_alpha = trans_alphas[ids_changed[i]]
            translation_changed = [s + '('+'{:.3f}'.format(trans_alpha[ind][0])+')' for ind, s in enumerate(trans_changed[i])]
        utils.print_out('Example ' + str(ids_changed[i]) + ': src:\t' + ' '.join(reference_changed) + '\t' + str(label_changed[i]))
        utils.print_out('Example ' + str(ids_changed[i]) + ': nmt:\t' + ' '.join(translation_changed) + '\t' + str(logits_changed[i]))
        if copy_masks is not None and len(copy_masks)>0:
            copy_mask = copy_masks[ids_changed[i]]
            copy_mask_str = [str(mask) for mask in copy_mask]
            utils.print_out('Example ' + str(ids_changed[i]) + ': msk:\t' + ' '.join(copy_mask_str))
        if X_adv_flip_num is not None:
            utils.print_out('Example ' + str(ids_changed[i]) + ' flipped tokens: ' + str(flip_num_changed[i]))
        utils.print_out(' ')

    if X_adv_flip_num is not None:
        lenght = 0
        for num in X_adv_flip_num:
            if num > 0:
                lenght += 1
        utils.print_out('Average flipped tokens: ' + str(sum(X_adv_flip_num) / lenght))

    utils.print_out('Step: ' + str(step) + ', cls_acc_pos=' + str(cls_acc_pos) + ', cls_acc_neg=' + str(cls_acc_neg))
    utils.print_out('Step: ' + str(step) + ', rouge_pos=' + str(pos_rouge) + ', rouge_neg=' + str(neg_rouge) + ', rouge_changed=' + str(changed_rouge))
    utils.print_out('Step: ' + str(step) + ', bleu_pos=' + str(pos_bleu) + ', bleu_neg=' + str(neg_bleu) + ', bleu_changed=' + str(changed_bleu))
    utils.print_out('Step: ' + str(step) + ', accept_pos=' + str(pos_accept) + ', accept_neg=' + str(neg_accept) + ', accept_changed=' + str(changed_accept))
    utils.print_out('Step: ' + str(step) + ', semsim_pos=' + str(pos_semsim) + ', semsim_neg=' + str(neg_semsim) + ', semsim_changed=' + str(changed_semsim))
    utils.print_out('Step: ' + str(step) + ', use_pos=' + str(pos_use) + ', use_neg=' + str(neg_use) + ', use_changed=' + str(changed_use))
    utils.print_out('Step: ' + str(step) + ', ae_acc=' + str(ae_acc) + ', word_acc=' + str(word_acc) + ', rouge=' + str(rouge) + ', bleu=' + str(bleu) +
                    ', accept=' + str(accept) + ', use=' + str(use) + ', semsim=' + str(avgcos(sent_embs, adv_sent_embs)))
    utils.print_out('Step: ' + str(step) + ', cls_orig_acc=' + str(cls_orig_acc) + ', cls_orig_auc=' + str(cls_orig_auc))
    utils.print_out('Step: ' + str(step) + ', cls_acc=' + str(cls_acc) + ', cls_auc=' + str(cls_auc))
    if cls_logits_def is not None and len(cls_logits_def) > 0:
        utils.print_out('Step: ' + str(step) + ', org_def_acc=' + str(org_def_acc) + ', org_def_auc=' + str(org_def_auc))
        utils.print_out('Step: ' + str(step) + ', cls_def_acc=' + str(cls_def_acc) + ', cls_def_auc=' + str(cls_def_auc))

    if is_test:
        with open(args.output_dir+'/src_changed.txt', 'w') as output_file:
            output_file.write('\n'.join([' '.join(a) for a in ref_changed]))
        with open(args.output_dir+'/adv_changed.txt', 'w') as output_file:
            output_file.write('\n'.join([' '.join(a) for a in trans_changed]))

        with open(args.output_dir+'/adv.txt', 'w') as output_file:
            output_file.write('\n'.join([' '.join(a) for a in translation_list]))
        with open(args.output_dir+'/adv_score.txt', 'w') as output_file:
            for score in cls_logits:
                output_file.write(' '.join([str(a) for a in score])+'\n')

    return cls_acc, cls_acc_pos, cls_acc_neg, changed_bleu


