"""
Bert-based classification model.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from misc import input_data
from models.bert import modeling


def get_device_str(num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:0"
    return device_str_output


class BertClassificationModel():
    def __init__(self, args, bert_config, mode=None):

        self.mode = mode
        self.bidirectional = True if args.enc_type == 'bi' else False
        self.args = args
        self.batch_size = args.batch_size

        self._make_graph(bert_config)

    def _make_graph(self, bert_config):

        self._init_placeholders()

        self.input_mask = tf.sequence_mask(
            tf.to_int32(self.encoder_inputs_length),
            tf.reduce_max(self.encoder_inputs_length),
            dtype=tf.int32)

        self.segment_ids = tf.sequence_mask(
            tf.to_int32(self.encoder_inputs_length),
            tf.reduce_max(self.encoder_inputs_length),
            dtype=tf.int32)

        self.segment_ids = 0 * self.segment_ids

        old_ = True if ((self.args.test_file is not None and 'yelp' in self.args.test_file) or
                        (self.args.input_file is not None and 'yelp' in self.args.input_file)) else False

        self.model = modeling.BertModel(
            config=bert_config,
            is_training=(self.mode == 'Train'),
            input_ids=self.encoder_inputs,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False,
            word_embedding_trainable=(self.mode == 'Train'),
        )

        encoder_outputs = self.model.get_pooled_output()

        with tf.variable_scope("classification") as scope:
            fc_output = tf.layers.dense(encoder_outputs, 1024, activation=tf.nn.relu)
            projection_layer = layers_core.Dense(units=self.args.output_classes, name="projection_layer")
            with tf.device(get_device_str(self.args.num_gpus)):
                self.logits = tf.nn.tanh(projection_layer(fc_output))  # [batch size, output_classes]

        # if self.mode == "Train":
        self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        self.segment_ids = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='segment_ids'
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


    def _init_optimizer(self):
        if self.args.output_classes > 2:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.classification_outputs))
        else:
            self.target_output = tf.cast(self.classification_outputs[:, -1], dtype=tf.int32)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.target_output))
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

    def make_train_inputs(self, x):
        return {
            self.encoder_inputs: x[0],
            self.classification_outputs: x[1],
            self.encoder_inputs_length: x[2]
        }

    def embedding_encoder_fn(self):
        return self.embedding_encoder

    def get_bert_embedding(self):
        return self.model.embedding_table

    def make_train_outputs(self, full_loss_step=True, defence=False):
        return [self.train_op, self.loss, self.logits, self.summary_op]

    def make_eval_outputs(self):
        return self.loss

    def make_test_outputs(self):
        return [self.loss, self.logits, self.encoder_inputs, self.classification_outputs]