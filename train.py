"""
Train or test a classification, auto-encoder, adversarial attack or defence model.
Usage:          python train.py ...
Input:          N/A
Output:         Trained models saved in directory specified by output_dir
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
from models.mySeq2Seq import Seq2SeqModel
from models.myClassifier import ClassificationModel
from models.myCNNClassifier import CNNClassificationModel
from models.myBertClassifier import BertClassificationModel
from misc.evaluate_attacks import evaluate_attack
from models.myAdversarialDefendingCopy import AdversarialModelCopy
import config
from timeit import default_timer
from misc import utils, input_data
import misc.eval_steps as eval_steps
from models.bert import modeling
import numpy as np

stop_words = [".", ",", "!", "...", "not", "n't"]

def maping_vocabs_bert(vocab_src, vocab_tgt):
    vocab_map = {}
    unk_id = 100
    for ind, word in enumerate(vocab_src):
        if word == '<sos>':
            vocab_map[ind] = vocab_tgt.index('[CLS]')
        elif word == '<eos>':
            vocab_map[ind] = vocab_tgt.index('[SEP]')
        elif word == '<unk>':
            vocab_map[ind] = vocab_tgt.index('[UNK]')
        elif word in vocab_tgt:
            vocab_map[ind] = vocab_tgt.index(word)
        else:
            vocab_map[ind] = unk_id
    return vocab_map

def maping_vocabs(vocab_src, vocab_tgt):
    vocab_map = {}
    unk_id = input_data.UNK_ID
    for ind, word in enumerate(vocab_src):
        if word in vocab_tgt:
            vocab_map[ind] = vocab_tgt.index(word)
        else:
            vocab_map[ind] = unk_id
    return vocab_map

def setStopWord(vocab):
    stop_word_index = []
    if args.use_stop_words:
        for word in stop_words:
            if word in vocab:
                stop_word_index.append(vocab.index(word))
        stop_word_index.append(input_data.UNK_ID)
        stop_word_index.append(input_data.SOS_ID)
        stop_word_index.append(input_data.EOS_ID)
    return stop_word_index

def load_data_iters(args):
    data_task = 'ae'
    if args.classification: data_task = 'clss'
    if args.adv: data_task = 'adv'
    if args.ae_vocab_file is not None: data_task = 'adv_counter_fitting'
    train_iter = input_data.get_dataset_iter(args, args.input_file, args.output_file, data_task,
                                             is_bert=(args.classification_model == 'BERT'))
    train_next = train_iter.get_next()
    dev_iter = input_data.get_dataset_iter(args, args.dev_file, args.dev_output, data_task, is_training=False,
                                           is_bert=(args.classification_model == 'BERT'))
    dev_next = dev_iter.get_next()
    return train_iter, train_next, dev_iter, dev_next

def create_models(args):
    model, model_dev, model_classifier = None, None, None
    if args.adv:
        # if args.copy:
        if args.classification_model == 'BERT':
            args.bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
        model = AdversarialModelCopy(args, mode='Train')
        model_dev = AdversarialModelCopy(args, mode="Infer", include_cls=False, embedding=model.get_embedding())
        model_classifier = AdversarialModelCopy(args, mode="Infer", include_ae=False, embedding=model.get_embedding())
    elif args.classification:
        if args.classification_model == 'RNN':
            utils.print_out('Initialise classification model: RNN')
            model = ClassificationModel(args, 'Train')
        elif args.classification_model == 'CNN':
            utils.print_out('Initialise classification model: CNN')
            model = CNNClassificationModel(args, mode='Train')
        elif args.classification_model == 'BERT':
            bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
            model = BertClassificationModel(args, bert_config, mode='Train')
            modeling.init_bert(args.bert_init_chk, word_embedding_trainable=True)
    else:
        model = Seq2SeqModel(args, mode="Train")
        model_dev = Seq2SeqModel(args, mode="Infer")
    return model, model_dev, model_classifier

def init_model(args):
    if args.load_model_cls is not None:
        vars = [i[0] for i in tf.train.list_variables(args.load_model_cls)]
        # cls_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier')
        # map_cls = {variable.op.name: variable for variable in cls_var_list if variable.op.name in vars}
        if args.defending:
            def_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='defending_classifier')
            map_def = {variable.op.name.replace('defending_classifier', 'my_classifier'): variable for variable in
                       def_var_list
                       if variable.op.name.replace('defending_classifier', 'my_classifier') in vars}
            tf.train.init_from_checkpoint(args.load_model_cls, map_def)
    if args.load_model_cls is not None:
        if args.use_defending_as_target:
            vars = [i[0] for i in tf.train.list_variables(args.load_model_cls)]
            cls_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier')
            map_def = {variable.op.name.replace('my_classifier', 'defending_classifier'): variable for variable in cls_var_list
                   if variable.op.name.replace('my_classifier', 'defending_classifier') in vars}
            tf.train.init_from_checkpoint(args.load_model_cls, map_def)
        else:
            vars = [i[0] for i in tf.train.list_variables(args.load_model_cls)]
            cls_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier')
            map_cls = {variable.op.name: variable for variable in cls_var_list if variable.op.name in vars}
            tf.train.init_from_checkpoint(args.load_model_cls, map_cls)
    if args.load_model_ae is not None:
        vars = [i[0] for i in tf.train.list_variables(args.load_model_ae)]
        ae_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_seq2seq')
        map_ae = {variable.op.name: variable for variable in ae_var_list if variable.op.name in vars}
        tf.train.init_from_checkpoint(args.load_model_ae, map_ae)
    if (args.classification and args.load_model is not None):
        vars = [i[0] for i in tf.train.list_variables(args.load_model)]
        cls_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier')
        map_cls = {variable.op.name: variable for variable in cls_var_list if variable.op.name in vars}
        tf.train.init_from_checkpoint(args.load_model, map_cls)

    # initialse the copy attention layer with pretrained bi_att model
    if args.adv and args.copy and args.classification_model != 'RNN' and args.load_copy_model is not None:
        vars = [i[0] for i in tf.train.list_variables(args.load_copy_model)]
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_copy_attention_layer')
        map_copy = {variable.op.name.replace('my_copy_attention_layer', 'my_classifier'): variable for variable in
                   var_list if variable.op.name.replace('my_copy_attention_layer', 'my_classifier') in vars}
        tf.train.init_from_checkpoint(args.load_copy_model, map_copy)


def train(args):
    vocab, _ = input_data.load_vocab(args.vocab_file)
    ae_vocab, _ = (args.ae_vocab_file, None) if args.ae_vocab_file is None else input_data.load_vocab(args.ae_vocab_file)
    args.stop_words = setStopWord(vocab) if args.ae_vocab_file is None else setStopWord(ae_vocab)
    args.vocab_map = None if args.ae_vocab_file is None else (maping_vocabs_bert(ae_vocab, vocab)
                                                                if args.classification_model == 'BERT'
                                                                else maping_vocabs(ae_vocab, vocab))

    train_iter, train_next, dev_iter, dev_next = load_data_iters(args)

    model, model_dev, model_classifier = create_models(args)

    utils.print_out('Training model constructed.')
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # initialise models with pretrained weights
        init_model(args)

        if args.use_model is not None:
            args.use_model.set_sess(sess)

        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sess.run(train_iter.initializer)

        if args.load_model_cls is not None and args.classification_model != 'BERT' and (not args.use_defending_as_target):
            saver_cls = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier'))
            saver_cls.restore(sess, args.load_model_cls)

        if args.adv and args.load_model is not None:
            saver_all = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            saver_all.restore(sess, args.load_model)

        utils.print_out('start training...')
        tf.get_default_graph().finalize()

        # init best infos
        step = 0
        last_improvement_step = -1
        best_loss = 1e6
        best_auc_max = -1
        best_auc_min = 1e6
        best_T = [100.0]*9

        upper_bounds = [95.0, 88.0, 78.0, 64.0, 54.0, 44.0, 34.0, 24.0, 14.0]
        lower_bounds = [90.0, 84.0, 74.0, 58.0, 48.0, 38.0, 28.0, 18.0, 8.0]


        while True:

            try:
                batch = sess.run(train_next)
                if args.copy:
                    copy_mask = eval_steps.get_copy_mask(sess, model, batch, np.max(batch[5]), n_top_k=args.top_k_attack)
                    batch = batch+(copy_mask,)

                results = sess.run(model.make_train_outputs(full_loss_step=(step % args.at_steps == 0), defence=args.defending),
                                       feed_dict=model.make_train_inputs(batch)) # Alternative training

            except tf.errors.OutOfRangeError:
                break

            if step % args.print_every_steps == 0:
                step_name = 'train'
                if args.defending and step % args.at_steps > 0:
                    step_name = 'defending'
                utils.print_out('Step: ' + str(step) + ', '+step_name+' loss=' + str(results[1]) +
                                (', ae_loss=' + str(results[4]) + ', cls_loss=' + str(results[5]) if len(results) > 5 else '') +
                                (', senti_loss='+str(results[7])+', aux_loss=' + str(results[6]) +', def_loss=' + str(results[8])
                                 if len(results) > 6 else '') +
                                (' *' if (results[1] < best_loss) else ''))

                if (results[1] < best_loss):
                    best_loss = results[1]

                if step % (10 * args.print_every_steps) == 0:

                    if args.adv:
                        cls_acc, cls_acc_pos, cls_acc_neg, changed_bleu = eval_steps.eval_adv(args, sess, dev_iter, model_dev, model_classifier,
                                                    dev_next, vocab, step, 10 * args.print_every_steps)

                        eval_score = cls_acc
                        eval_bleu = changed_bleu
                        if args.target_label is not None:
                            eval_score = cls_acc_neg if args.target_label == 1 else cls_acc_pos
                            # eval_bleu = neg_bleu if args.target_label == 1 else pos_bleu

                        # use accuracy as best selection measure in each threshold
                        thre_score = cls_acc

                        if args.save_checkpoints:
                            for i in range(9):
                                if eval_score >= lower_bounds[i] and eval_score < upper_bounds[i]:
                                    if thre_score < best_T[i]:
                                        best_T[i] = thre_score
                                        saver.save(sess, args.output_dir + '/' + 'nmt-T'+str(i)+'.ckpt')
                                        utils.print_out('Step: ' + str(step) + ' model saved for T'+str(i)+' *')

                        if (eval_score <= 0.0) or eval_score >= args.lowest_bound_score:
                            if eval_score < best_auc_min:
                                best_auc_min = eval_score
                                last_improvement_step = step
                        else:
                            break

                    elif args.classification:
                        auc, acc = eval_steps.eval_classification(args, sess, dev_iter, dev_next, model, vocab)
                        utils.print_out('Step: ' + str(step) + ', test acc=' + str(acc) + ', auc=' + str(auc))
                        eval_score = acc
                        if args.output_classes > 2:
                            eval_score = acc
                        if eval_score > best_auc_max:
                            best_auc_max = eval_score
                            last_improvement_step = step
                            saver.save(sess, args.output_dir + '/' + 'nmt.ckpt')
                    else:
                        acc, word_acc, rouge, bleu = eval_steps.eval_ae(sess, dev_iter, model_dev, dev_next, vocab, step,
                                                             50*args.print_every_steps)
                        utils.print_out('Step: ' + str(step) + ', test acc=' + str(acc) + ', word_acc=' + str(word_acc)
                                        + ', rouge=' + str(rouge) + ', bleu=' + str(bleu))
                        if bleu > best_auc_max:
                            best_auc_max = bleu
                            last_improvement_step = step
                            saver.save(sess, args.output_dir + '/' + 'nmt.ckpt')

            if args.total_steps is None:
                if step - last_improvement_step > args.stop_steps:
                    break
            else:
                if step >= args.total_steps:
                    break

            step += 1

        utils.print_out('finish training')

        if args.do_test:
            args.load_model = args.output_dir + '/' + 'nmt.ckpt-'+str(step)


def test(args):
    data_task = 'ae'
    if args.classification: data_task = 'clss'
    if args.adv: data_task = 'adv'
    if args.ae_vocab_file is not None: data_task = 'adv_counter_fitting'

    vocab, _ = input_data.load_vocab(args.vocab_file)
    ae_vocab, _ = (args.ae_vocab_file, None) if args.ae_vocab_file is None else input_data.load_vocab(args.ae_vocab_file)
    args.stop_words = setStopWord(vocab) if args.ae_vocab_file is None else setStopWord(ae_vocab)
    args.vocab_map = None if args.ae_vocab_file is None else (maping_vocabs_bert(ae_vocab, vocab)
                                                                if args.classification_model == 'BERT'
                                                                else maping_vocabs(ae_vocab, vocab))

    test_iter = input_data.get_dataset_iter(args, args.test_file, args.test_output, data_task,
                                            is_training=False, is_test=True,
                                            is_bert=(args.classification_model == 'BERT'))
    test_next = test_iter.get_next()

    step = 0
    if args.adv:
        model_test = AdversarialModelCopy(args, mode="Infer", include_cls=False)
        model_classifier = AdversarialModelCopy(args, mode="Infer", include_ae=False, embedding=model_test.get_embedding())
    elif args.classification:
        if args.classification_model == 'RNN':
            utils.print_out('Initialise classification model: RNN')
            model_test = ClassificationModel(args, mode='Train')
        elif args.classification_model == 'CNN':
            utils.print_out('Initialise classification model: CNN')
            model_test = CNNClassificationModel(args, mode='Train')
        elif args.classification_model == 'BERT':
            bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
            model_test = BertClassificationModel(args, bert_config, mode='Test')
    else:
        model_test = Seq2SeqModel(args, mode="Infer")

    utils.print_out('Testing model constructed.')

    saver = tf.train.Saver()

    with tf.Session() as sess:

        if args.classification and args.use_defending_as_target:
            vars = [i[0] for i in tf.train.list_variables(args.load_model)]
            def_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_classifier')
            map_def = {variable.op.name.replace('my_classifier', 'defending_classifier'): variable for variable in def_var_list
                   if variable.op.name.replace('my_classifier', 'defending_classifier') in vars}
            tf.train.init_from_checkpoint(args.load_model, map_def)

        if args.adv or (args.classification and not args.use_defending_as_target):
            vars = [i[0] for i in tf.train.list_variables(args.load_model)]
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            map_all = {variable.op.name: variable for variable in
                       var_list if variable.op.name in vars}
            tf.train.init_from_checkpoint(args.load_model, map_all)

        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sess.run(test_iter.initializer)

        # if args.adv or (args.classification and not args.defending):
        #     saver.restore(sess, args.load_model)

        if args.use_model is not None:
            args.use_model.set_sess(sess)

        if args.adv:
            eval_steps.eval_adv(args, sess, test_iter, model_test, model_classifier, test_next, vocab, step,
                                demo_per_step=1, is_train=False)

        elif args.classification:
            auc, acc = eval_steps.eval_classification(args, sess, test_iter, test_next, model_test, vocab)
            utils.print_out('Test: acc=' + str(acc) + ', auc=' + str(auc))

        else:
            acc, word_acc, rouge, bleu = eval_steps.eval_ae(sess, test_iter, model_test, test_next, vocab, step,
                                                 demo_per_step=1)
            utils.print_out('Test: acc=' + str(acc) + ', word_acc=' + str(word_acc)
                            + ', rouge=' + str(rouge) + ', bleu=' + str(bleu))



def test_adv_pos_neg(args):
    data_task = 'adv'
    if args.ae_vocab_file is not None: data_task = 'adv_counter_fitting'
    vocab, _ = input_data.load_vocab(args.vocab_file)
    ae_vocab, _ = (args.ae_vocab_file, None) if args.ae_vocab_file is None else input_data.load_vocab(args.ae_vocab_file)
    args.stop_words = setStopWord(vocab) if args.ae_vocab_file is None else setStopWord(ae_vocab)
    args.vocab_map = None if args.ae_vocab_file is None else (maping_vocabs_bert(ae_vocab, vocab)
                                                                if args.classification_model == 'BERT'
                                                                else maping_vocabs(ae_vocab, vocab))

    test_iter = input_data.get_dataset_iter(args, args.test_file, args.test_output, data_task,
                                            is_training=False, is_test=True,
                                            is_bert=(args.classification_model == 'BERT'))
    test_next = test_iter.get_next()

    step = 0
    model_test = AdversarialModelCopy(args, mode="Infer", include_cls=False)
    model_classifier = AdversarialModelCopy(args, mode="Infer", include_ae=False, embedding=model_test.get_embedding())

    utils.print_out('Testing model constructed.')

    saver = tf.train.Saver()
    sess_pos = tf.Session()
    sess_pos.run([tf.global_variables_initializer(), tf.tables_initializer()])
    sess_pos.run(test_iter.initializer)

    saver.restore(sess_pos, args.load_model_pos)

    _, _, dev_logits, dev_labels = eval_steps.run_classification(args, model_classifier, vocab, sess_pos, test_iter, test_next)

    target_predicts = np.argmax(dev_logits, axis=-1)
    pos_mask = (dev_labels[:, 1] == 1)
    neg_mask = (dev_labels[:, 1] == 0)
    correct_mask = (target_predicts == dev_labels[:, 1])
    pos_predict_mask = pos_mask & correct_mask
    neg_predict_mask = neg_mask & correct_mask
    keep_orig_mask = 1 - correct_mask

    sess_neg = tf.Session()
    sess_neg.run([tf.global_variables_initializer(), tf.tables_initializer()])
    saver.restore(sess_neg, args.load_model_neg)

    if args.use_model is not None:
        args.use_model.set_sess(sess_pos)

    start = default_timer()
    decoder_reference_list, decoder_prediction_list_pos, cls_labels, copy_masks_pos = \
        eval_steps.run_adv(args, model_test, sess_pos, test_iter, test_next)
    decoder_reference_list, decoder_prediction_list_neg, cls_labels, copy_masks_neg = \
        eval_steps.run_adv(args, model_test, sess_neg, test_iter, test_next)

    decoder_prediction_list = []
    for i in range((len(decoder_reference_list) // args.batch_size) + (1 if (len(decoder_reference_list) % args.batch_size > 0) else 0)):
        start, end = i*args.batch_size, (i+1)*args.batch_size
        decoder_prediction_batch = np.array(decoder_reference_list[start:end]) * np.expand_dims(keep_orig_mask[start:end], axis=1) + \
                              np.array(decoder_prediction_list_pos[start:end]) * np.expand_dims(pos_predict_mask[start:end], axis=1) \
                              + np.array(decoder_prediction_list_neg[start:end]) * np.expand_dims(neg_predict_mask[start:end], axis=1)
        decoder_prediction_list.extend(decoder_prediction_batch)

    end = default_timer()
    utils.print_out('Adversarial attack elapsed:' + '{0:.4f}'.format(end - start) + 's')

    cls_logits_def, cls_origs_def, cls_logits, cls_orig_logits, sent_embs, adv_sent_embs, \
           orig_alphas, trans_alphas, trans_alphas_def = \
        eval_steps.run_classifications(args, sess_pos, test_iter, decoder_prediction_list, model_classifier, test_next)

    evaluate_attack(args, step, decoder_reference_list, decoder_prediction_list,
                                                        cls_logits, cls_orig_logits, cls_labels, vocab,
                                                        sent_embs, adv_sent_embs,
                                                        is_test=True, orig_alphas=orig_alphas,
                                                        trans_alphas=trans_alphas,
                                                        cls_logits_def=cls_logits_def, cls_origs_def=cls_origs_def)

def main(args):

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)

    if args.do_cond_test:
        test_adv_pos_neg(args)



if __name__ == '__main__':
    args = config.add_arguments()
    main(args)
