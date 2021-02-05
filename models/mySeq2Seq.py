"""
RNN-based auto-encoder model.
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from misc import input_data
import models.utils as utils

class Seq2SeqModel():
    def __init__(self, args, mode = None):

        self.mode = mode
        self.bidirectional = True if args.enc_type == 'bi' else False
        self.args = args

        # self.batch_size = args.batch_size
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.beam_width = args.beam_width

        self._make_graph()

    def _make_graph(self):

        self._init_placeholders()
        self._init_decoder_train_connectors()

        with tf.variable_scope("my_seq2seq", reuse=tf.AUTO_REUSE) as scope:
            self._init_embedding()

            self._init_encoder()
            self._init_decoder()

        if self.mode == "Train":
            self._init_optimizer()

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
        
        self.decoder_targets_length = tf.placeholder(
            shape = (None,),
            dtype = tf.int32,
            name = 'decoder_targets_length',
        )

        self.classification_outputs = tf.placeholder(
            shape=(None, self.args.output_classes),
            dtype=tf.int32,
            name='classification_outputs',
        )
            
    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):                
            self.decoder_train_length = self.decoder_targets_length
            self.loss_weights = tf.ones(
                [self.batch_size, tf.reduce_max(self.decoder_train_length)], 
                dtype=tf.float32)

    def _init_embedding(self):
        self.embedding_encoder = input_data._create_pretrained_emb_from_txt(
            vocab_file=self.args.vocab_file, embed_file=self.args.emb_file)
        self.encoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_encoder, 
            self.encoder_inputs)  #[batch size, sequence len, h_dim]
        
        self.embedding_decoder = self.embedding_encoder
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_decoder, 
            self.decoder_inputs)  #[batch size, sequence len, h_dim]

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            if self.bidirectional:
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus)) for _ in range(self.args.enc_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus)) for _ in range(self.args.enc_layers)])
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    self.encoder_embedding_inputs,
                    dtype=tf.float32,
                    sequence_length=self.encoder_inputs_length)

                self.encoder_outputs = tf.concat(bi_outputs, -1) #[batch size, sequence len, h_dim*2]
                bi_encoder_state = bi_state

                if self.args.enc_layers == 1:
                    self.encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(self.args.enc_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    self.encoder_state = tuple(encoder_state)
            else:
                encoder_cell = tf.contrib.rnn.MultiRNNCell([utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus)) for _ in range(self.args.enc_layers)])
                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=self.encoder_embedding_inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32
                )    #[batch size, sequence len, h_dim]

    def _init_decoder(self):
        
        def create_decoder_cell():
            cell = tf.contrib.rnn.MultiRNNCell([utils.make_cell(self.args.enc_num_units, utils.get_device_str(self.args.num_gpus)) for _ in range(self.args.dec_layers)])
                
            if self.args.beam_width > 0 and self.mode == "Infer":
                dec_start_state = seq2seq.tile_batch(self.encoder_state, self.beam_width)
                enc_outputs = seq2seq.tile_batch(self.encoder_outputs, self.beam_width)
                enc_lengths = seq2seq.tile_batch(self.encoder_inputs_length, self.beam_width)
            else:
                dec_start_state = self.encoder_state
                enc_outputs = self.encoder_outputs
                enc_lengths = self.encoder_inputs_length

            if self.args.attention:
                attention_states = enc_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.args.dec_num_units,
                    attention_states,
                    memory_sequence_length = enc_lengths)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell,
                    attention_mechanism,
                    attention_layer_size = self.args.dec_num_units)

                if self.args.beam_width > 0 and self.mode == "Infer":
                    initial_state = decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32)
                else:
                    initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)

                initial_state = initial_state.clone(cell_state = dec_start_state)
            else:

                decoder_cell = cell
                initial_state = dec_start_state

            return decoder_cell, initial_state
        
        with tf.variable_scope("Decoder") as scope:

            projection_layer = tf.layers.Dense(self.args.vocab_size, use_bias=False, name="projection_layer")

            self.encoder_state = tuple(self.encoder_state[-2:])

            decoder_cell, initial_state = create_decoder_cell()

            if self.mode == "Train":
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embedding_inputs,
                    self.decoder_train_length)


                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = decoder_cell,
                    helper = training_helper,
                    initial_state = initial_state,
                    output_layer=projection_layer)

                (self.decoder_outputs_train,
                self.decoder_state_train,
                final_sequence_length) = tf.contrib.seq2seq.dynamic_decode(
                        decoder = training_decoder,
                        impute_finished = True,
                        scope = scope
                )

                self.decoder_logits_train = self.decoder_outputs_train.rnn_output
                decoder_predictions_train = tf.argmax(self.decoder_logits_train, axis=-1)
                self.decoder_predictions_train = tf.identity(decoder_predictions_train)

            elif self.mode == "Infer":
                start_tokens = tf.tile(tf.constant([input_data.SOS_ID], dtype=tf.int32), [self.batch_size])

                if self.args.beam_width > 0:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell          = decoder_cell,
                            embedding     = self.embedding_decoder,
                            start_tokens  = tf.ones_like(self.encoder_inputs_length) * tf.constant(input_data.SOS_ID, dtype = tf.int32),
                            end_token     = tf.constant(input_data.EOS_ID, dtype = tf.int32),
                            initial_state = initial_state,
                            beam_width    = self.beam_width,
                            output_layer  = projection_layer)

                    self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = inference_decoder,
                        maximum_iterations = tf.round(tf.reduce_max(self.decoder_targets_length)) * 2,
                        impute_finished = False,
                        scope = scope)

                    self.decoder_predictions_inference = tf.identity(self.decoder_outputs_inference.predicted_ids)
                else:
                    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder,
                        start_tokens = start_tokens,
                        end_token=input_data.EOS_ID) # EOS id

                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell = decoder_cell,
                        helper = inference_helper,
                        initial_state = initial_state,
                        output_layer = projection_layer)

                    self.decoder_outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = inference_decoder,
                        maximum_iterations = tf.round(tf.reduce_max(self.decoder_targets_length)) * 2,
                        impute_finished = False,
                        scope = scope)

                    self.decoder_predictions_inference = tf.identity(self.decoder_outputs_inference.sample_id)

    def _init_optimizer(self):
        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length), 
            tf.reduce_max(self.decoder_targets_length),
            dtype = tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.decoder_logits_train,
            self.decoder_outputs,
            loss_mask)
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        
        learning_rate = self.args.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1.0*self.args.max_gradient_norm, self.args.max_gradient_norm), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)


    # Inputs and Outputs for train and infer
    def make_train_inputs(self, x):
        return {
            self.encoder_inputs: x[0],
            self.decoder_inputs: x[1],
            self.decoder_outputs: x[2],
            self.classification_outputs: x[3],
            self.encoder_inputs_length: x[4],
            self.decoder_targets_length: x[5],
            self.batch_size: len(x[0])
        }

    def make_train_outputs(self, full_loss_step=True, defence=False):
        return [self.train_op, self.loss, self.decoder_predictions_train, self.summary_op]

    def make_infer_outputs(self):
        return self.decoder_predictions_inference

    def make_eval_outputs(self):
        return self.loss
