"""
RNN-based classification model.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from misc import input_data
import numpy as np
import models.utils as utils


class ClassificationModel():
    def __init__(self, args, mode=None):

        self.mode = mode
        self.bidirectional = True if args.enc_type == 'bi' else False
        self.args = args
        self.batch_size = args.batch_size

        self._init_placeholders()

        self.encoder_outputs, self.cls_logits, self.acc, self.alphas = self._make_graph()

        if self.mode == "Train":
            self._init_optimizer()

    def _make_graph(self, encoder_embedding_inputs=None):

        with tf.variable_scope("my_classifier", reuse=tf.AUTO_REUSE) as scope:
            if encoder_embedding_inputs is None:
                self._init_embedding()
            encoder_outputs, _ = self._init_encoder(encoder_embedding_inputs=(self.encoder_embedding_inputs
                                                                              if encoder_embedding_inputs is None else
                                                                              encoder_embedding_inputs))

            with tf.variable_scope("classification") as scope:
                alphas = None
                if self.args.cls_attention:
                    with tf.variable_scope("attention") as scope:
                        x = tf.reshape(encoder_outputs, [-1, self.args.enc_num_units*2
                                                            if self.bidirectional else self.args.enc_num_units])
                        self.cls_attention_layer = layers_core.Dense(self.args.cls_attention_size, name="cls_attention_layer")
                        self.cls_attention_fc_layer = layers_core.Dense(1, name="cls_attention_fc_layer")
                        with tf.device(utils.get_device_str(self.args.num_gpus)):
                            x = tf.nn.relu(self.cls_attention_layer(x))
                            x = self.cls_attention_fc_layer(x)
                            logits = tf.reshape(x, [-1, tf.shape(encoder_outputs)[1], 1])
                            alphas = tf.nn.softmax(logits, dim=1)
                            encoder_outputs = encoder_outputs * alphas
                    output_rnn_last = tf.reduce_sum(encoder_outputs, axis=1)  #[batch size, h_dim]
                else:
                    output_rnn_last = tf.reduce_mean(encoder_outputs, axis=1)  # [batch size, h_dim]
                projection_layer = layers_core.Dense(units=self.args.output_classes, name="projection_layer")
                with tf.device(utils.get_device_str(self.args.num_gpus)):
                    cls_logits = tf.nn.tanh(projection_layer(output_rnn_last))     #[batch size, output_classes]
                    ybar = tf.argmax(cls_logits, axis=1, output_type=tf.int32)
                    self.categorical_logits = tf.one_hot(ybar, depth=2, on_value=1.0, off_value=0.0)
                    ylabel = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
                    count = tf.equal(ylabel, ybar)
                    acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        return encoder_outputs, cls_logits, acc, alphas

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
        
        self.classification_outputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='classification_outputs',
        )

    def _init_embedding(self, trainable=False):
        if trainable:
            with tf.variable_scope("my_word_embeddings", reuse=tf.AUTO_REUSE) as scope:
                emb_mat, emb_size = input_data.load_embed_json(self.args.emb_file, vocab_size=self.args.vocab_size)
                self.embedding_encoder = tf.get_variable(name="embedding_matrix", shape=emb_mat.shape,
                                                         initializer=tf.constant_initializer(emb_mat),
                                                         trainable=True)
        else:
            self.embedding_encoder = input_data._create_pretrained_emb_from_txt(
                vocab_file=self.args.vocab_file, embed_file=self.args.emb_file,
                trainable_tokens=3)

        self.encoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_encoder,
            self.encoder_inputs)   #[batch size, sequence len, h_dim]

    def _init_encoder(self, encoder_embedding_inputs=None, trainable=True):

        with tf.variable_scope("Encoder") as scope:
            if self.bidirectional:
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus), trainable=trainable) for _ in range(self.args.enc_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus), trainable=trainable) for _ in range(self.args.enc_layers)])
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    encoder_embedding_inputs,
                    dtype=tf.float32)
                    #sequence_length=self.encoder_inputs_length)

                encoder_outputs = tf.concat(bi_outputs, -1)  #[batch size, sequence len, h_dim*2]
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
                encoder_cell = tf.contrib.rnn.MultiRNNCell([utils.make_cell(self.args.enc_num_units,
                                                                            utils.get_device_str(self.args.num_gpus),
                                                                            trainable=trainable) for _ in range(self.args.enc_layers)])
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_embedding_inputs,
                    #sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32
                )    #[batch size, sequence len, h_dim]
        return encoder_outputs, encoder_state

    def _init_optimizer(self):

        if self.args.output_classes > 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.cls_logits, labels=self.classification_outputs))
        else:
            self.target_output = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.cls_logits, labels=self.target_output))

        if self.args.defending:
            orig, augmented = tf.split(self.encoder_embedding_inputs, num_or_size_splits=2, axis=0)
            self.aux_loss = tf.reduce_sum(tf.keras.losses.MSE(orig, augmented))
            self.loss += self.args.aux_lambda * self.aux_loss

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        
        learning_rate = self.args.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1.0*self.args.max_gradient_norm, self.args.max_gradient_norm), var) for grad, var in self.gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    # inputs and outputs for train/infer

    def make_train_inputs(self, x, X_data=None):
        x_input = x[0]
        if X_data is not None:
            x_input = X_data
        if len(self.args.def_train_set) == 1:
            x_input_def = x[-2]
            if (len(x_input[0]) - len(x_input_def[0])) > 0:
                x_input_def = np.concatenate([x_input_def, np.ones([len(x_input_def), len(x_input[0])-len(x_input_def[0])], dtype=np.int32)*input_data.EOS_ID],
                                             axis=1)
            if (len(x_input[0]) - len(x_input_def[0])) < 0:
                x_input = np.concatenate([x_input, np.ones([len(x_input), len(x_input_def[0])-len(x_input[0])], dtype=np.int32)*input_data.EOS_ID],
                                         axis=1)
            x_input = np.concatenate([x_input, x_input_def], axis=0)
            y_input = np.concatenate([x[1], x[1]], axis=0)
            x_lenghts = np.concatenate([x[2], x[-1]], axis=0)
        else:
            y_input = x[1]
            x_lenghts = x[2]
        return {
            self.encoder_inputs: x_input,
            self.classification_outputs: y_input,
            self.encoder_inputs_length: x_lenghts
        }

    def embedding_encoder_fn(self):
        return self.embedding_encoder

    def make_train_outputs(self, full_loss_step=True, defence=False):
        if self.args.cls_attention:
            return [self.train_op, self.loss, self.cls_logits, self.summary_op, self.alphas]
        else:
            return [self.train_op, self.loss, self.cls_logits, self.summary_op]

    def make_eval_outputs(self):
        return self.loss

    def make_test_outputs(self):
        if self.args.cls_attention:
            return [self.loss, self.cls_logits, self.acc, self.alphas, self.encoder_inputs, self.classification_outputs]
        else:
            return [self.loss, self.cls_logits, self.acc, self.encoder_inputs, self.classification_outputs]

    def make_encoder_output(self):
        return self.encoder_outputs

