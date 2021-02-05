"""
CNN-based classification model.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from misc import input_data
import numpy as np
import models.utils as utils


class CNNClassificationModel():
    def __init__(self, args, mode=None):

        self.mode = mode
        self.bidirectional = True if args.enc_type == 'bi' else False
        self.args = args
        self.batch_size = args.batch_size

        self._init_placeholders()

        self.encoder_outputs, self.cls_logits, self.acc, _ = self._make_graph()

        # if self.mode == "Train":
        self._init_optimizer()

    def _make_graph(self, encoder_embedding_inputs=None):

        with tf.variable_scope("my_classifier", reuse=tf.AUTO_REUSE) as scope:
            if encoder_embedding_inputs is None:
                self._init_embedding()
            encoder_outputs = self._init_encoder(encoder_embedding_inputs=(self.encoder_embedding_inputs
                                                 if encoder_embedding_inputs is None else
                                                 encoder_embedding_inputs))

            with tf.variable_scope("classification", reuse=tf.AUTO_REUSE) as scope:
                output_flatten = tf.reduce_mean(encoder_outputs, axis=1)
                fc_output = tf.layers.dense(output_flatten, 1024, activation=tf.nn.relu)
                projection_layer = layers_core.Dense(units=self.args.output_classes, name="projection_layer")
                with tf.device(utils.get_device_str(self.args.num_gpus)):
                    logits = tf.nn.tanh(projection_layer(fc_output))  # [batch size, output_classes]
                    ybar = tf.argmax(logits, axis=1, output_type=tf.int32)
                    ylabel = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
                    count = tf.equal(ylabel, ybar)
                    acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        return encoder_outputs, logits, acc, None

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

    def _init_embedding(self):
        self.embedding_encoder = input_data._create_pretrained_emb_from_txt(
            vocab_file=self.args.vocab_file, embed_file=self.args.emb_file)
        self.encoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_encoder,
            self.encoder_inputs)  # [batch size, sequence len, h_dim]

    def _init_encoder(self, encoder_embedding_inputs=None, initializer=tf.random_normal_initializer(stddev=0.1)):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE) as scope:
            with tf.device(utils.get_device_str(self.args.num_gpus)):
                for i, filter_size in enumerate(self.args.filter_sizes):
                    filter = tf.get_variable("filter_"+str(filter_size), [filter_size, 300 if i==0 else self.args.enc_num_units,
                                                                          self.args.enc_num_units],
                                             initializer=initializer)
                    conv = tf.nn.conv1d(encoder_embedding_inputs if i==0 else h, filter, stride=2, padding="VALID")
                    conv = tf.contrib.layers.batch_norm(conv, is_training=True if (self.mode=='Train') else False, scope='cnn_bn_'+str(filter_size))
                    b = tf.get_variable("b_"+str(filter_size), [self.args.enc_num_units])
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h, keep_prob=self.args.dropout_keep_prob)
        return h_drop

    def _init_optimizer(self):
        if self.args.output_classes > 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.cls_logits, labels=self.classification_outputs))
        else:
            self.target_output = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.cls_logits, labels=self.target_output))
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        learning_rate = self.args.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [
            (tf.clip_by_value(grad, -1.0 * self.args.max_gradient_norm, self.args.max_gradient_norm), var) for grad, var
            in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    # inputs and outputs for train/infer

    def make_train_inputs(self, x, X_data=None):
        x_input = x[0]
        if X_data is not None:
            x_input = X_data
        if len(self.args.def_train_set) > 0:
            x_input_def = x[-2]
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
        return [self.train_op, self.loss, self.cls_logits, self.summary_op]

    def make_eval_outputs(self):
        return self.loss

    def make_test_outputs(self):
        return [self.loss, self.cls_logits, self.acc, self.encoder_inputs, self.classification_outputs]

    def make_encoder_output(self):
        return self.encoder_outputs