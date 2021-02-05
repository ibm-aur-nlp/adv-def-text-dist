"""
Evaluation metrics for adversarial attacks
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import re
from misc.scripts import bleu, rouge
from misc import acc_transformer


##################### Sequence Reconstruction evaluation scores ####################

# from acceptability import test
def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(references, translations, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False
  references_bleu = [[reference] for reference in references]
  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      references_bleu, translations, max_order, smooth)
  return 100 * bleu_score


def _rouge(references, translations, subword_option=None):
  """Compute ROUGE scores and handling BPE."""
  translations_sent = [' '.join(translation) for translation in translations]
  references_sent = [' '.join(reference) for reference in references]
  rouge_score_map = rouge.rouge(translations_sent, references_sent)
  return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(references, translations):
  """Compute accuracy, each line contains a label."""

  count = 0.0
  match = 0.0
  for ind, label in enumerate(references):
    label_sentence = ' '.join(label)
    pred_sentence = ' '.join(translations[ind])
    if label_sentence == pred_sentence:
      match += 1
    count += 1
  return 100 * match / count


def _word_accuracy(references, translations):
  """Compute accuracy on per word basis."""

  total_acc, total_count = 0., 0.
  for ind, reference in enumerate(references):
    translation = translations[ind]
    match = 0.0
    for pos in range(min(len(reference), len(translation))):
      label = reference[pos]
      pred = translation[pos]
      if label == pred:
        match += 1
    total_acc += 100 * match / max(len(reference), len(translation))
    total_count += 1
  return total_acc / total_count


##################### Classification evaluation scores ####################
def max_index(arr):
    max_v, max_p = -1, -1
    for ind, a in enumerate(arr):
      a = float(a)
      if a > max_v:
        max_v = a
        max_p = ind
    return max_p

def _clss_accuracy(labels, predicts):
    """Compute accuracy for classification"""
    total_count = 0.
    match = 0.0
    for ind, label in enumerate(labels):
      max_lab_index = max_index(label)
      max_pred_index = max_index(predicts[ind])
      if max_pred_index == max_lab_index:
          match += 1
      total_count += 1
    return 100.0 * match / total_count

def _clss_accuracy_micro(labels, predicts, orig_label=1):
    """Compute accuracy for classification"""
    total_count = 0.
    match = 0.0
    for ind, label in enumerate(labels):
      max_lab_index = max_index(label)
      if max_lab_index == orig_label:
          max_pred_index = max_index(predicts[ind])
          if max_pred_index == max_lab_index:
              match += 1
          total_count += 1
    return 100.0 * match / total_count

import numpy as np
from sklearn import metrics
def _clss_auc(labels, predicts):
    """c Compute auc for classification"""
    fpr, tpr, thresholds = metrics.roc_curve(np.array(labels)[:, 1], np.array(predicts)[:, 1], pos_label=1.0)
    auc = metrics.auc(fpr, tpr)
    return 100.0 * auc




##################### EMB similarity score ##################
def gen_mask(len_lists, max_len):
    ret = []
    for a in len_lists:
        mask_array = np.zeros(max_len, dtype=int)
        mask_array[np.arange(a)] = 1
        ret.append(mask_array)
    return np.array(ret)

from sklearn.metrics.pairwise import cosine_similarity
def emb_cosine_dist(emb1, emb2, emb_len1, emb_len2):
    mask1 = gen_mask(emb_len1, len(emb1[0]))
    mask1 = np.expand_dims(mask1, axis=-1)
    mask2 = gen_mask(emb_len2, len(emb2[0]))
    mask2 = np.expand_dims(mask2, axis=-1)
    emb1 = np.multiply(emb1, mask1)
    emb2 = np.multiply(emb2, mask2)
    emb1 = np.sum(emb1, axis=1) / np.expand_dims(emb_len1, axis=-1)
    emb2 = np.sum(emb2, axis=1) / np.expand_dims(emb_len2, axis=-1)
    scores = []
    for ind, emb in enumerate(emb1):
        scores.append(cosine_similarity([emb], [emb2[ind]])[0][0])
    return scores

##################### ACPT score ##################
def _accept_score(references, translations, args, lim=100):
    if args.accept_name is None:
        return 0.0
    references_sent = [' '.join(reference) for reference in references[:lim]]
    translations_sent = [' '.join(translation) for translation in translations[:lim]]
    scores = acc_transformer.evaluate_accept(references_sent, translations_sent, args)
    return sum(scores)/len(scores)


##################### USE score ##################
def _use_scores(ref_changed, trans_changed, use_model, eval_num=200):
    if use_model is None:
        return 0.0
    cnt = 0
    sim_scores = []
    while cnt < min(len(ref_changed), eval_num):
        batch_src = ref_changed[cnt: cnt + 32]
        batch_tgt = trans_changed[cnt: cnt + 32]
        src_sent = [' '.join(a) for a in batch_src]
        tgt_sent = [' '.join(a) for a in batch_tgt]
        scores = use_model.semantic_sim(src_sent, tgt_sent)
        sim_scores.extend(scores)
        cnt += 32
    return sum(sim_scores) / len(sim_scores)



