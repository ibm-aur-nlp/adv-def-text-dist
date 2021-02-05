"""
Adversarial attack and defence model.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
import numpy as np
from misc import input_data
from tensorflow.contrib.distributions.python.ops import relaxed_onehot_categorical
from models.myCopyDecoder import CopyDecoder
import models.utils as utils
from tensorflow.python.ops import math_ops

class AdversarialModelCopy():
    def __init__(self, args, mode=None, include_ae=True, include_cls=True, embedding=None):

        self.mode = mode
        self.include_ae = include_ae
        self.include_cls = include_cls
        self.ae_bidirectional = True if args.enc_type == 'bi' else False
        self.cls_bidirectional = True if args.cls_enc_type == 'bi' else False
        self.args = args

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.beam_width = args.beam_width
        self.stop_words = args.stop_words

        self.old_ = True if ((self.args.test_file is not None and 'yelp' in self.args.test_file) or
                            (self.args.input_file is not None and 'yelp' in self.args.input_file)) else False

        if embedding is not None:
            self.embedding_encoder, self.cf_emb_decoder, self.para_emb_decoder, \
            self.ae_embedding_encoder, self.transfer_embedding_decoder = embedding
        else:
            self.embedding_encoder = None
            self.cf_emb_decoder = None
            self.para_emb_decoder = None
            self.ae_embedding_encoder = None
            self.transfer_embedding_decoder = None

        self._make_graph()

    def _make_graph(self):

        self._init_placeholders()
        self._init_decoder_train_connectors()

        # use pretrained classifier embedding for unk, <s> and </s>, keeping them fixed
        with tf.variable_scope("my_classifier", reuse=tf.AUTO_REUSE) as scope:
            self._init_embedding(trainable=False)

            # get copy_mask, reuse map from RNN models
            if self.args.copy and self.include_ae and self.args.classification_model == 'RNN':
                self._init_importance_scores()

        if self.args.copy and self.include_ae and self.args.classification_model != 'RNN':
            with tf.variable_scope("my_copy_attention_layer", reuse=tf.AUTO_REUSE) as scope:
                self._init_importance_scores()


        if self.include_ae:
            with tf.variable_scope("my_seq2seq", reuse=tf.AUTO_REUSE) as scope:
                self.encoder_embedding_inputs = tf.nn.embedding_lookup(
                    self.ae_embedding_encoder,
                    self.encoder_inputs)  # [batch size, sequence len, h_dim]
                self.decoder_embedding_inputs = tf.nn.embedding_lookup(
                    self.ae_embedding_encoder,
                    self.decoder_inputs)  # [batch size, sequence len, h_dim]

                self.ae_encoder_outputs, self.ae_encoder_state = self._init_encoder(self.encoder_inputs_length,
                                                                                    self.encoder_embedding_inputs,
                                                                                    self.ae_bidirectional,
                                                                                    self.args.enc_num_units,
                                                                                    self.args.enc_layers)

                self._init_decoder(trainable=True)

        if self.include_cls:

            if self.include_ae:
                # gumbel-softmax: mapping samples to embeddings
                with tf.device(utils.get_device_str(self.args.num_gpus, gpu_rellocate=True)):
                    dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(
                        temperature=self.args.gumbel_softmax_temporature,
                        logits=self.decoder_logits_train)
                    self.dist_sample = dist.sample()
                    self.dist_sample_argmax = tf.argmax(self.dist_sample, axis=-1)
                    if self.args.ae_vocab_file is not None:
                        tiled_encoder = tf.tile(tf.expand_dims(self.transfer_embedding_decoder, 0), [self.batch_size, 1, 1])
                    else:
                        tiled_encoder = tf.tile(tf.expand_dims(self.embedding_encoder, 0), [self.batch_size, 1, 1])
                    self.cls_encoder_emb_inp = tf.matmul(self.dist_sample, tiled_encoder)
                    if self.args.classification_model == 'BERT':
                        sos_embs = tf.tile(tf.expand_dims(tf.expand_dims(self.embedding_encoder[101], 0), 0), [self.batch_size, 1, 1])
                        self.cls_encoder_emb_inp = tf.concat([sos_embs, self.cls_encoder_emb_inp], axis=1)

                self.ae_dec_tgt_emb = tf.nn.embedding_lookup(
                    self.embedding_encoder,
                    self.decoder_outputs)  # [batch size, sequence len, h_dim]

                if self.args.gan:
                    target_lens = tf.concat([self.decoder_targets_length, self.decoder_targets_length], axis=0)
                    input_emb = tf.concat([self.ae_dec_tgt_emb, self.cls_encoder_emb_inp], axis=0)
                    self.disc_logits_pos, _, _ = self._init_classifier(input_emb, target_lens,
                                                                   scope_name='my_discriminator_pos',
                                                                   trainable=True)
                    self.disc_logits_neg, _, _ = self._init_classifier(input_emb, target_lens,
                                                                   scope_name='my_discriminator_neg',
                                                                   trainable=True)

            else:
                self.cls_encoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_encoder,
                    self.decoder_outputs)  # [batch size, sequence len, h_dim]
                self.eval_para_emb_inp = tf.nn.embedding_lookup(
                    self.para_emb_decoder,
                    self.decoder_outputs)

            self.cls_logits, self.cls_encoder_outputs, self.alphas = self._init_classifier(self.cls_encoder_emb_inp,
                                                                                           self.decoder_targets_length,
                                                                                           scope_name='my_classifier',
                                                                                           trainable=False)

            if self.args.defending:
                target_lens = self.decoder_targets_length
                input_emb = self.cls_encoder_emb_inp
                if self.include_ae:
                    target_lens = tf.concat([self.decoder_targets_length, self.decoder_targets_length], axis=0)
                    input_emb = tf.concat([self.cls_encoder_emb_inp, self.ae_dec_tgt_emb], axis=0)

                self.cls_logits_def, self.cls_encoder_outputs_def, self.alphas_def = self._init_classifier(
                    input_emb,
                    target_lens,
                    scope_name='defending_classifier',
                    trainable=True)

        if self.mode == "Train":
            self._init_optimizer()


    def _init_importance_scores(self):
        
        self.importance_score = tf.ones([self.batch_size, tf.reduce_max(self.decoder_targets_length)], dtype=tf.float32)
        if self.args.attention_copy_mask:
            copy_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder,
                self.decoder_outputs)  # [batch size, sequence len, h_dim]
            cls_encoder_outputs, _ = self._init_encoder(self.decoder_targets_length,
                                                        copy_emb_inp,
                                                        self.cls_bidirectional,
                                                        self.args.cls_enc_num_units,
                                                        self.args.cls_enc_layers,
                                                        trainable=False)
            with tf.variable_scope("classification", reuse=tf.AUTO_REUSE) as scope:
                alpha = self._init_cls_attention(cls_encoder_outputs, trainable=False)
            self.importance_score = tf.squeeze(alpha, axis=-1)

        # weigh in the stop_words factor
        if self.args.use_stop_words:
            self.importance_score = self.importance_score * self.stop_word_mask


    def _init_classifier(self, input_emb, target_lens, scope_name, trainable=True):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            # target_lens = tf.concat([self.decoder_targets_length, self.decoder_targets_length], axis=0)
            # input_emb = tf.concat([self.ae_dec_tgt_emb, self.cls_encoder_emb_inp], axis=0)
            if self.args.classification_model == 'RNN':
                cls_encoder_outputs, _ = self._init_encoder(target_lens,
                                                                 input_emb,
                                                                 self.cls_bidirectional,
                                                                 self.args.cls_enc_num_units,
                                                                 self.args.cls_enc_layers,
                                                                 trainable=trainable)
                logits, cls_encoder_outputs, alphas = self._init_classification(cls_encoder_outputs, trainable=trainable)
            elif self.args.classification_model == 'CNN': #CNN
                cls_encoder_outputs = self._init_cnn_encoder(input_emb, trainable=trainable)
                logits, cls_encoder_outputs, alphas = self._init_cnn_classification(cls_encoder_outputs, trainable=trainable)
            else:
                return None, None, None

            return logits, cls_encoder_outputs, alphas


    ################# CNN as Classifier ################
    def _init_cnn_encoder(self, cls_encoder_outputs, trainable=None, gpu_rellocate=False, initializer=tf.random_normal_initializer(stddev=0.1)):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE) as scope:
            with tf.device(utils.get_device_str(self.args.num_gpus, gpu_rellocate=gpu_rellocate)):
                for i, filter_size in enumerate(self.args.filter_sizes):
                    filter = tf.get_variable("filter_"+str(filter_size), [filter_size, 300 if i==0 else self.args.cls_enc_num_units, self.args.cls_enc_num_units],
                                             initializer=initializer, trainable=trainable)
                    conv = tf.nn.conv1d(cls_encoder_outputs if i==0 else h, filter, stride=2, padding="VALID")
                    conv = tf.contrib.layers.batch_norm(conv, is_training=True, scope='cnn_bn_'+str(filter_size))
                    b = tf.get_variable("b_"+str(filter_size), [self.args.cls_enc_num_units], trainable=trainable)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h, keep_prob=self.args.dropout_keep_prob)
        return h_drop

    def _init_cnn_classification(self, cls_encoder_outputs, trainable=None, gpu_rellocate=False):
        with tf.variable_scope("classification", reuse=tf.AUTO_REUSE) as scope:
            output_flatten = tf.reduce_mean(cls_encoder_outputs, axis=1)
            fc_output = tf.layers.dense(output_flatten, 1024, activation=tf.nn.relu, trainable=trainable)
            projection_layer = layers_core.Dense(units=self.args.output_classes, name="projection_layer", trainable=trainable)
            with tf.device(utils.get_device_str(self.args.num_gpus, gpu_rellocate=gpu_rellocate)):
                cls_logits = tf.nn.tanh(projection_layer(fc_output))  # [batch size, output_classes]
        return cls_logits, cls_encoder_outputs, None

    ################# RNN as Classifier ################
    def _init_classification(self, cls_encoder_outputs, trainable=None, gpu_rellocate=False):
        with tf.variable_scope("classification", reuse=tf.AUTO_REUSE) as scope:
            if self.args.cls_attention:
                alphas = self._init_cls_attention(cls_encoder_outputs, trainable=trainable)
                cls_encoder_outputs = cls_encoder_outputs * alphas
                output_rnn_last = tf.reduce_sum(cls_encoder_outputs, axis=1)  # [batch size, h_dim]
            else:
                alphas = None
                output_rnn_last = tf.reduce_mean(cls_encoder_outputs, axis=1)  # [batch size, h_dim]
            projection_layer = layers_core.Dense(units=self.args.output_classes, name="projection_layer",
                                                 trainable=trainable)
            with tf.device(utils.get_device_str(self.args.num_gpus, gpu_rellocate=gpu_rellocate)):
                cls_logits = tf.nn.tanh(projection_layer(output_rnn_last))  # [batch size, output_classes]
        return cls_logits, cls_encoder_outputs, alphas

    def _init_cls_attention(self, cls_encoder_outputs, trainable=None):
        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE) as scope:
            x = tf.reshape(cls_encoder_outputs, [-1, self.args.cls_enc_num_units * 2
            if self.cls_bidirectional else self.args.cls_enc_num_units])
            cls_attention_layer = layers_core.Dense(self.args.cls_attention_size,
                                                    name="cls_attention_layer", trainable=trainable)
            cls_attention_fc_layer = layers_core.Dense(1, name="cls_attention_fc_layer", trainable=trainable)
            with tf.device(utils.get_device_str(self.args.num_gpus)):
                x = tf.nn.relu(cls_attention_layer(x))
                x = cls_attention_fc_layer(x)
                logits = tf.reshape(x, [-1, tf.shape(cls_encoder_outputs)[1], 1])
                alphas = tf.nn.softmax(logits, dim=1)
        return alphas

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_inputs',
        )

        self.decoder_outputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_outputs',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets',
        )

        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )
        
        self.prem_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='prem_length',
        )

        self.classification_outputs = tf.placeholder(
            shape=(None, self.args.output_classes),
            dtype=tf.int32,
            name='classification_outputs',
        )

        self.stop_word_mask = tf.placeholder(
            shape=(None, None),
            dtype=tf.float32,
            name='stop_word_mask',
        )

        self.copy_mask = tf.placeholder(
            shape=(None, None),
            dtype=tf.float32,
            name='copy_mask',
        )

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            self.decoder_train_length = self.decoder_targets_length
            self.loss_weights = tf.ones(
                [self.batch_size, tf.reduce_max(self.decoder_train_length)],
                dtype=tf.float32)

    def _init_embedding(self, trainable=True):

        if self.embedding_encoder is None:
            if self.include_ae and self.args.ae_vocab_file is not None:
                self.embedding_encoder, self.ae_embedding_encoder, self.transfer_embedding_decoder = input_data._create_pretrained_embeddings_from_jsons(
                        vocab_file=self.args.ae_vocab_file, embed_file=self.args.ae_emb_file,
                        cls_vocab_file=self.args.vocab_file, cls_embed_file=self.args.emb_file,
                        cls_model=self.args.classification_model,
                    )
            else:
                self.embedding_encoder = input_data._create_pretrained_emb_from_txt(
                        vocab_file=self.args.vocab_file, embed_file=self.args.emb_file
                    )
                self.ae_embedding_encoder = self.embedding_encoder
                self.transfer_embedding_decoder = self.embedding_encoder

        self.cf_emb_decoder = self.embedding_encoder

        self.para_emb_decoder = self.embedding_encoder

    def get_embedding(self):
        return self.embedding_encoder, self.cf_emb_decoder, self.para_emb_decoder, \
               self.ae_embedding_encoder, self.transfer_embedding_decoder

    def _init_encoder(self, encoder_inputs_length, embedding_inputs, bidirectional,
                      enc_num_units, enc_layers, trainable=True, gpu_rellocate=False):
        with tf.variable_scope("Encoder") as scope:
            if bidirectional:
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(enc_num_units, utils.get_device_str(self.args.num_gpus, gpu_rellocate), trainable=trainable) for _ in range(enc_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(enc_num_units, utils.get_device_str(self.args.num_gpus, gpu_rellocate), trainable=trainable) for _ in range(enc_layers)])
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    embedding_inputs,
                    dtype=tf.float32,
                    sequence_length=encoder_inputs_length if trainable else None,
                )

                encoder_outputs = tf.concat(bi_outputs, -1)  # [batch size, sequence len, h_dim*2]
                bi_encoder_state = bi_state

                if self.args.enc_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(self.args.enc_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                encoder_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus, gpu_rellocate), trainable=trainable) for _ in range(self.args.enc_layers)])
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=embedding_inputs,
                    sequence_length=encoder_inputs_length if trainable else None,
                    dtype=tf.float32
                )  # [batch size, sequence len, h_dim]
        return encoder_outputs, encoder_state

    def _init_decoder(self, trainable=True, gpu_rellocate=False):

        def create_decoder_cell(trainable=True):
            cell = tf.contrib.rnn.MultiRNNCell(
                [utils.make_cell(self.args.dec_num_units, utils.get_device_str(self.args.num_gpus, gpu_rellocate), trainable=trainable) for _ in range(self.args.dec_layers)])

            if self.args.beam_width > 0 and self.mode == "Infer":
                dec_start_state = seq2seq.tile_batch(self.ae_encoder_state, self.beam_width)
                enc_outputs = seq2seq.tile_batch(self.ae_encoder_outputs, self.beam_width)
                enc_lengths = seq2seq.tile_batch(self.encoder_inputs_length, self.beam_width)
            else:
                dec_start_state = self.ae_encoder_state
                enc_outputs = self.ae_encoder_outputs
                enc_lengths = self.encoder_inputs_length

            if self.args.attention:
                attention_states = enc_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.args.dec_num_units,
                    attention_states,
                    memory_sequence_length = enc_lengths
                )

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell,
                    attention_mechanism,
                    attention_layer_size = self.args.dec_num_units
                )

                if self.args.beam_width > 0 and self.mode == "Infer":
                    initial_state = decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32)
                else:
                    initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)

                initial_state = initial_state.clone(cell_state = dec_start_state)
                # initial_state_beam = initial_state_beam.clone(cell_state=dec_start_state)
            else:

                decoder_cell = cell
                initial_state = dec_start_state
                # initial_state_beam = None

            return decoder_cell, initial_state

        with tf.variable_scope("Decoder") as scope:

            vocab_size = self.args.vocab_size if self.args.ae_vocab_file is None else self.args.ae_vocab_size

            projection_layer = tf.layers.Dense(vocab_size, use_bias=False,
                                               name="projection_layer", trainable=trainable)


            self.ae_encoder_state = tuple(self.ae_encoder_state[-2:])

            decoder_cell, initial_state = create_decoder_cell(trainable=trainable)

            if self.mode == "Train":
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embedding_inputs,
                    self.decoder_train_length)

                if self.args.copy:
                    training_decoder = CopyDecoder(
                        cell=decoder_cell,
                        helper=training_helper,
                        initial_state=initial_state,
                        copy_mask=self.copy_mask,
                        encoder_input_ids=self.decoder_targets,
                        vocab_size=vocab_size,
                        output_layer=projection_layer)
                else:
                    training_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=decoder_cell,
                        helper=training_helper,
                        initial_state=initial_state,
                        output_layer=projection_layer)

                (self.decoder_outputs_train,
                 self.decoder_state_train,
                 final_sequence_length) = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    impute_finished=True,
                    scope=scope
                )

                self.decoder_logits_train = self.decoder_outputs_train.rnn_output
                decoder_predictions_train = tf.argmax(self.decoder_logits_train, axis=-1)
                self.decoder_predictions_train = tf.identity(decoder_predictions_train)

            elif self.mode == "Infer":
                start_tokens = tf.tile(tf.constant([input_data.SOS_ID], dtype=tf.int32), [self.batch_size])

                if self.args.beam_width > 0:
                    inference_decoder_beam = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=self.embedding_encoder,
                        start_tokens=tf.ones_like(self.encoder_inputs_length) * tf.constant(input_data.SOS_ID,
                                                                                            dtype=tf.int32),
                        end_token=tf.constant(input_data.EOS_ID, dtype=tf.int32),
                        initial_state=initial_state,
                        beam_width=self.beam_width,
                        output_layer=projection_layer)

                    self.decoder_outputs_inference_beam, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=inference_decoder_beam,
                        maximum_iterations=tf.round(tf.reduce_max(self.decoder_targets_length)),
                        impute_finished=False,
                        scope=scope)
                    self.decoder_predictions_inference = tf.identity(tf.transpose(self.decoder_outputs_inference_beam.predicted_ids, perm=[2, 0, 1]))
                else:
                    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.ae_embedding_encoder,
                        start_tokens=start_tokens,
                        end_token=input_data.EOS_ID)  # EOS id

                    if self.args.copy:
                        inference_decoder = CopyDecoder(
                            cell=decoder_cell,
                            helper=inference_helper,
                            initial_state=initial_state,
                            copy_mask=self.copy_mask,
                            encoder_input_ids=self.decoder_targets,
                            vocab_size=vocab_size,
                            output_layer=projection_layer)
                    else:
                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                            cell=decoder_cell,
                            helper=inference_helper,
                            initial_state=initial_state,
                            output_layer=projection_layer)

                    self.decoder_outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        maximum_iterations=tf.round(tf.reduce_max(self.decoder_targets_length)),
                        impute_finished=False,
                        scope=scope)
                    self.decoder_logits_inference = self.decoder_outputs_inference.rnn_output
                    self.decoder_predictions_inference = tf.identity(self.decoder_outputs_inference.sample_id)

    def _normalise_weights(self, alphas_to_weights):
        return (alphas_to_weights - tf.expand_dims(tf.reduce_min(alphas_to_weights, axis=-1), axis=-1))\
                                    /(tf.expand_dims(tf.reduce_max(alphas_to_weights, axis=-1), axis=-1)
                                     - tf.expand_dims(tf.reduce_min(alphas_to_weights, axis=-1), axis=-1))

    def _gen_loss_attention(self, loss_mask):
        if self.args.cls_attention:
            with tf.variable_scope("my_classifier", reuse=True) as scope:
                ae_dec_tgt_encoding, _ = self._init_encoder(self.decoder_targets_length,
                                                            self.ae_dec_tgt_emb,
                                                            self.cls_bidirectional,
                                                            self.args.cls_enc_num_units,
                                                            self.args.cls_enc_layers,
                                                            trainable=False)

                # get attention alphas for decoder targets
                with tf.variable_scope("classification", reuse=True) as scope:
                    alphas = self._init_cls_attention(ae_dec_tgt_encoding, trainable=False)

            # mini-max normalization
            alphas_to_weights = tf.squeeze(alphas, axis=-1)
            if self.args.loss_attention_norm:
                alphas_to_weights = self._normalise_weights(alphas_to_weights)
            attention_weights = tf.multiply(alphas_to_weights, loss_mask)
            return attention_weights
        else:
            return loss_mask

    def _cls_enc_embedding(self):
        with tf.variable_scope("my_classifier", reuse=True) as scope:
            ae_dec_tgt_encoding, _ = self._init_encoder(self.decoder_targets_length,
                                                        self.ae_dec_tgt_emb,
                                                        self.cls_bidirectional,
                                                        self.args.cls_enc_num_units,
                                                        self.args.cls_enc_layers,
                                                        trainable=False)
            # get attention alphas for decoder targets
            if self.args.cls_attention:
                with tf.variable_scope("classification", reuse=True) as scope:
                    alphas = self._init_cls_attention(ae_dec_tgt_encoding, trainable=False)
                    ae_dec_tgt_encoding = ae_dec_tgt_encoding * alphas
        return ae_dec_tgt_encoding

    def relaxed_WMD(self, emb1, emb2, loss_mask):
        # cos-dist
        emb1 = tf.nn.l2_normalize(emb1)
        emb2 = tf.nn.l2_normalize(emb2)

        emb1_tile = tf.tile(tf.expand_dims(emb1, 1), [1, tf.reduce_max(self.decoder_targets_length), 1, 1])
        emb1_tile_t = tf.transpose(emb1_tile, perm=[0, 2, 1, 3])
        emb2_tile = tf.tile(tf.expand_dims(emb2, 1), [1, tf.reduce_max(self.decoder_targets_length), 1, 1])

        # cos-dist
        radial_diffs = math_ops.multiply(emb1_tile_t, emb2_tile)
        emb_diff_cos = 1 - math_ops.reduce_sum(radial_diffs, axis=-1)
        max_mask = (1 - loss_mask) * 1e6
        max_mask_tile = tf.tile(tf.expand_dims(max_mask, 1), [1, tf.reduce_max(self.decoder_targets_length), 1])
        max_mask_tile_t = tf.transpose(max_mask_tile, perm=[0, 2, 1])

        emb_row_min = tf.multiply(tf.reduce_min(emb_diff_cos + max_mask_tile, axis=-1), loss_mask)
        emb_col_min = tf.multiply(tf.reduce_min(emb_diff_cos + max_mask_tile_t, axis=1), loss_mask)
        emb_row = tf.reduce_sum(emb_row_min, axis=-1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
        emb_col = tf.reduce_sum(emb_col_min, axis=-1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
        emb_row_col_stack = tf.stack([emb_row, emb_col], axis=-1)
        emb_row_col_max = tf.reduce_max(emb_row_col_stack, axis=-1)
        WMD_dist = tf.reduce_mean(emb_row_col_max)
        return WMD_dist

    def _emb_dist_loss(self, emb1, emb2, dist='avgcos', attention_weights=None, loss_mask=None):

        emb_dist_loss = tf.constant(0.0)
        if dist == 'rWMD':
            emb_dist_loss = self.relaxed_WMD(emb1, emb2, loss_mask)

        if dist=='l2':
            emb_dist_loss = tf.nn.l2_loss(tf.multiply(emb1, tf.expand_dims(attention_weights, -1))
                                          - tf.multiply(emb2, tf.expand_dims(attention_weights, -1)))

        if dist=='avgl2':
            alphas_to_weights = attention_weights
            if self.args.cls_attention:
                alphas_to_weights = tf.squeeze(self.alphas, axis=-1)
                if self.args.loss_attention_norm:
                    alphas_to_weights = self._normalise_weights(alphas_to_weights)
            emb1_avg = tf.reduce_sum(tf.multiply(emb1, tf.expand_dims(attention_weights, -1)),
                                           axis=1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
            emb2_avg = tf.reduce_sum(tf.multiply(emb2, tf.expand_dims(alphas_to_weights, -1)),
                                            axis=1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
            emb_dist_loss = tf.reduce_mean(tf.nn.l2_loss(emb1_avg - emb2_avg))

        if dist=='cos':
            normalize_a = tf.nn.l2_normalize(emb1, 2)
            normalize_b = tf.nn.l2_normalize(emb2, 2)
            cos_sim_pairwise = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
            cos_sim_weighted = tf.reduce_sum(tf.multiply(cos_sim_pairwise, attention_weights), axis=-1) / tf.reduce_sum(attention_weights, axis=-1)
            emb_dist_loss  = -tf.reduce_mean(cos_sim_weighted)
            # emb_dist_loss = tf.compat.v1.losses.cosine_distance(normalize_a, normalize_b,
            #                                                     axis=-1, weights=tf.expand_dims(attention_weights, -1),
            #                                                     reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        if dist == 'avgcos':
            alphas_to_weights = attention_weights
            if self.args.cls_attention:
                alphas_to_weights = tf.squeeze(self.alphas, axis=-1)
            if self.args.loss_attention_norm:
                alphas_to_weights = self._normalise_weights(alphas_to_weights)
            # Averaging
            emb1_avg = tf.reduce_sum(tf.multiply(emb1, tf.expand_dims(attention_weights, -1)), axis=1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
            emb2_avg = tf.reduce_sum(tf.multiply(emb2, tf.expand_dims(alphas_to_weights, -1)), axis=1) / tf.expand_dims(tf.reduce_sum(loss_mask, axis=1), -1)
            # emb_dist_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.emb1_avg, axis=1),
            #                                                tf.nn.l2_normalize(self.emb2_avg, axis=1), axis=-1)
            emb_dist_loss = tf.reduce_mean(utils.cos_dist_loss(emb1_avg, emb2_avg))
        return emb_dist_loss

    def _init_ae_loss(self):
        self.seq_loss = tf.constant(0.0)
        self.aux_loss = tf.constant(0.0)

        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length),
            tf.reduce_max(self.decoder_targets_length),
            dtype=tf.float32)

        if self.args.loss_attention:
            attention_weights = self._gen_loss_attention(loss_mask)
        else:
            attention_weights = loss_mask

        self.seq_loss = tf.contrib.seq2seq.sequence_loss(
            self.decoder_logits_train,
            self.decoder_targets,
            loss_mask)

        if self.args.gan:
            # Adversarial loss according to discriminator
            real_labels = tf.ones(self.batch_size, dtype=tf.int32)
            fake_labels = tf.ones(self.batch_size, dtype=tf.int32)
            crossent_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_logits_pos,
                                                                      labels=tf.concat([real_labels, fake_labels],
                                                                                       axis=0))[self.batch_size:]
            crossent_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_logits_neg,
                                                                          labels=tf.concat([real_labels, fake_labels],
                                                                                           axis=0))[self.batch_size:]
            neg_flags = tf.cast(self.classification_outputs[:, -1], dtype=tf.float32)
            pos_flags = 1 - neg_flags
            loss_pos = tf.reduce_sum(math_ops.multiply(crossent_pos, pos_flags)) / tf.reduce_sum(pos_flags) \
                if self.args.target_label is None or self.args.target_label==0 else tf.constant(0.0)
            loss_neg = tf.reduce_sum(math_ops.multiply(crossent_neg, neg_flags)) / tf.reduce_sum(neg_flags) \
                if self.args.target_label is None or self.args.target_label==1 else tf.constant(0.0)
            self.aux_loss = loss_pos + loss_neg

        self.ae_loss = self.args.aux_lambda * self.seq_loss + (1-self.args.aux_lambda) * self.aux_loss

    def _init_adv_loss(self):
        if self.args.output_classes > 2:
            if self.args.label_beta is not None:
                bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.args.label_beta)
                loss = bce(self.classification_outputs, self.cls_logits)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_logits, labels=self.classification_outputs))
            self.cls_loss = -1.0 * loss
            return
        if self.args.label_beta is not None:
            bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=self.args.label_beta)
            if self.args.balance:
                neg_masks = tf.cast(self.classification_outputs[:, -1], dtype=tf.float32)
                pos_loss = bce(self.classification_outputs, self.cls_logits, sample_weight=1-neg_masks)
                neg_loss = bce(self.classification_outputs, self.cls_logits, sample_weight=neg_masks)
                self.cls_loss = math_ops.maximum(pos_loss, neg_loss)
            else:
                self.cls_loss = bce(self.classification_outputs, self.cls_logits)
        else:
            self.target_output = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
            crossent= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_logits, labels=self.target_output)

            if self.args.balance:
                target_output = tf.cast(self.classification_outputs[:, -1], dtype=tf.float32)
                self.cls_loss = math_ops.maximum(tf.reduce_sum(math_ops.multiply(crossent, target_output)) / tf.reduce_sum(
                    target_output), tf.reduce_sum( math_ops.multiply(crossent, (1 - target_output))) / tf.reduce_sum(
                        1 - target_output)
                    )
            else:
                self.cls_loss = tf.reduce_mean(crossent)

    def _init_disc_loss(self):

        real_labels = tf.ones(self.batch_size, dtype=tf.int32)
        fake_labels = tf.zeros(self.batch_size, dtype=tf.int32)
        crossent_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_logits_pos,
                                                                  labels=tf.concat([real_labels, fake_labels], axis=0))
        crossent_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_logits_neg,
                                                                  labels=tf.concat([real_labels, fake_labels], axis=0))
        neg_flags = tf.cast(self.classification_outputs[:, -1], dtype=tf.float32)
        neg_flags = tf.concat([neg_flags, neg_flags], axis=0)
        pos_flags = 1-neg_flags
        self.disc_loss_pos = tf.reduce_sum(math_ops.multiply(crossent_pos, pos_flags)) / tf.reduce_sum(pos_flags) \
            if self.args.target_label is None or self.args.target_label==0 else tf.constant(0.0)
        self.disc_loss_neg = tf.reduce_sum(math_ops.multiply(crossent_neg, neg_flags)) / tf.reduce_sum(neg_flags) \
            if self.args.target_label is None or self.args.target_label==1 else tf.constant(0.0)


    def _init_defending_loss(self):

        if self.args.output_classes > 2:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_logits_def,
                                                                             labels=tf.concat([self.classification_outputs, self.classification_outputs], axis=0)))
            self.def_loss = loss
        else:
            labels = 1-tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
            crossent= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_logits_def,
                                                                     labels=tf.concat([labels, labels], axis=0))
            self.def_loss = tf.reduce_mean(crossent)

    def _init_sent_loss(self):
        # cls_enc_emb: Sentiment
        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length),
            tf.reduce_max(self.decoder_targets_length),
            dtype=tf.float32)
        mask = loss_mask
        if self.args.copy:
            mask = mask * (1-self.copy_mask)
        senti_dec_tgt_emb = tf.nn.embedding_lookup(
            self.cf_emb_decoder,
            self.decoder_outputs)
        tiled_encoder = tf.tile(tf.expand_dims(self.cf_emb_decoder, 0), [self.batch_size, 1, 1])
        senti_dec_spl_emb = tf.matmul(self.dist_sample, tiled_encoder)
        self.sentiment_emb_dist_loss = self._emb_dist_loss(senti_dec_tgt_emb, senti_dec_spl_emb,
                                                           self.args.sentiment_emb_dist, mask, loss_mask)

    def _init_optimizer(self):
        # auto-encoder loss
        self._init_ae_loss()
        tf.summary.scalar('ae_loss', self.ae_loss)
        self.summary_op = tf.summary.merge_all()

        # sentiment loss
        self.sentiment_emb_dist_loss = tf.constant(0.0)
        if self.args.seq_lambda < 1.0 and (not self.args.defending):
            self._init_sent_loss()
            tf.summary.scalar('sent_loss', self.sentiment_emb_dist_loss)
            self.summary_op = tf.summary.merge_all()

        self.disc_loss = tf.constant(0.0)
        if self.args.gan:
            self._init_disc_loss()
            self.disc_loss = self.disc_loss_pos + self.disc_loss_neg

        self.def_loss = tf.constant(0.0)
        if self.args.defending:
            # defending loss
            self._init_adv_loss()
            self._init_defending_loss()
            tf.summary.scalar('adv_loss', self.cls_loss)
            tf.summary.scalar('def_loss', self.def_loss)
            self.summary_op = tf.summary.merge_all()
        else:
            # adversarial loss
            self._init_adv_loss()
            tf.summary.scalar('adv_loss', self.cls_loss)
            self.summary_op = tf.summary.merge_all()

        # weighted sum
        self.loss = self.args.ae_lambda * (self.args.seq_lambda * self.ae_loss +
                                       (1-self.args.seq_lambda) * self.sentiment_emb_dist_loss) + \
                (1-self.args.ae_lambda) * self.cls_loss

        learning_rate = self.args.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = self.optimizer.compute_gradients(self.loss,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "my_seq2seq"))
        capped_gradients = [
            (tf.clip_by_value(grad, -1.0 * self.args.max_gradient_norm, self.args.max_gradient_norm), var) for grad, var
            in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)

        # discriminator_loss only train_op_ae
        if self.args.gan:
            gradients_disc = self.optimizer.compute_gradients(self.disc_loss,
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                               "my_discriminator_pos")+
                                                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                               "my_discriminator_neg"))
            capped_gradients_disc = [
                (tf.clip_by_value(grad, -1.0 * self.args.max_gradient_norm, self.args.max_gradient_norm), var) for grad, var
                in gradients_disc if grad is not None]
            self.train_op_disc = self.optimizer.apply_gradients(capped_gradients_disc)

        if self.args.defending:
            self.loss_def = self.args.ae_lambda * (self.args.seq_lambda * self.ae_loss +
                                       (1-self.args.seq_lambda) * self.sentiment_emb_dist_loss) + \
                                        (1-self.args.ae_lambda) * self.def_loss

            gradients_def = self.optimizer.compute_gradients(self.loss_def)
            capped_gradients_def = [
                (tf.clip_by_value(grad, -1.0 * self.args.max_gradient_norm, self.args.max_gradient_norm), var)
                for grad, var
                in gradients_def if grad is not None]
            self.train_op_def = self.optimizer.apply_gradients(capped_gradients_def)


    # Inputs and Outputs for train and infer

    def make_train_inputs(self, x):
        target = x[6] if self.args.ae_vocab_file is not None else x[2]
        stop_word_mask = [[1.0] * len(target[0])] * len(target)
        stop_word_mask = np.array(stop_word_mask)
        if self.args.use_stop_words:
            for ind, example in enumerate(target):
                for j, word in enumerate(example):
                    if word in self.stop_words:
                        stop_word_mask[ind][j] = 0.0

        outputs = 1 - x[3]
        if self.args.target_label is not None:
            outputs = np.array([[0.0, 1.0]]*len(outputs)) if self.args.target_label==1 else np.array([[1.0, 0.0]]*len(outputs))
        if self.args.output_classes > 2:
            outputs = x[3]
        
        prem_length = x[5]

        return {
            self.encoder_inputs: x[0],
            self.decoder_inputs: x[1],
            self.decoder_outputs: x[2],
            self.classification_outputs: outputs,
            self.encoder_inputs_length: x[4],
            self.decoder_targets_length: x[5],
            self.prem_length: prem_length,
            self.decoder_targets: x[2] if self.args.ae_vocab_file is None else x[6],
            self.batch_size: len(x[0]),
            self.stop_word_mask: stop_word_mask,
            self.copy_mask: x[-1] if self.args.copy else stop_word_mask,
        }

    def get_copy_masks(self):
        return [self.copy_mask, self.importance_score]

    def make_train_outputs(self, full_loss_step=True, defence=False):
        if full_loss_step:
            return [self.train_op, self.loss, self.decoder_predictions_train, self.summary_op,
                self.ae_loss, self.cls_loss, self.aux_loss, self.sentiment_emb_dist_loss, self.def_loss,
                    self.classification_outputs]
        elif defence:
            return [self.train_op_def, self.loss_def, self.decoder_predictions_train, self.summary_op,
                    self.ae_loss, self.cls_loss, self.aux_loss, self.sentiment_emb_dist_loss, self.def_loss]
        else:
            return [self.train_op_disc, self.loss, self.decoder_predictions_train, self.summary_op,
                    self.ae_loss, self.cls_loss, self.aux_loss, self.sentiment_emb_dist_loss, self.def_loss]

    def make_infer_outputs(self):
        if self.args.copy:
            return [self.decoder_predictions_inference, self.copy_mask]
        else:
            return [self.decoder_predictions_inference]

    def make_classifier_input(self, x, prediction_outputs, prediction_output_lengths):
        return {
            self.encoder_inputs: x[0],
            self.decoder_inputs: x[1],
            self.decoder_outputs: prediction_outputs,
            self.classification_outputs: x[3],
            self.encoder_inputs_length: x[4],
            self.decoder_targets: x[6] if self.args.ae_vocab_file is not None else prediction_outputs,
            self.decoder_targets_length: prediction_output_lengths,
            self.batch_size: len(x[0])
        }

    def make_classifier_outputs(self):
        if self.args.cls_attention:
            return [self.cls_logits, self.eval_para_emb_inp, self.alphas]
        else:
            return [self.cls_logits, self.eval_para_emb_inp]

    def make_def_classifier_outputs(self):
        if self.args.cls_attention:
            return [self.cls_logits_def, self.eval_para_emb_inp, self.alphas_def]
        else:
            return [self.cls_logits_def, self.eval_para_emb_inp]

    def make_cls_encoder_outputs(self):
        return self.cls_encoder_outputs

    def make_eval_outputs(self):
        return self.loss

