import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None,
                  max_across_timesteps=False):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
      logits: A 3D Tensor of shape
        [batch_size x sequence_length x num_decoder_symbols] and dtype float.
        The logits correspond to the prediction across all classes at each
        timestep.
      targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype
        int. The target represents the true class at each timestep.
      weights: A 2D Tensor of shape [batch_size x sequence_length] and dtype
        float. Weights constitutes the weighting of each prediction in the
        sequence. When using weights as masking set all valid timesteps to 1 and
        all padded timesteps to 0.
      average_across_timesteps: If set, sum the cost across the sequence
        dimension and divide by the cost by the total label weight across
        timesteps.
      average_across_batch: If set, sum the cost across the batch dimension and
        divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".
  Returns:
	    A scalar float Tensor: The average log-perplexity per symbol (weighted).

	  Raises:
	    ValueError: logits does not have 3 dimensions or targets does not have 2
	                dimensions or weights does not have 2 dimensions.
	  """
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
	                     "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
	                     "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
	                     "tensor")
    with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
        num_classes = array_ops.shape(logits)[2]
        probs_flat = array_ops.reshape(logits, [-1, num_classes])
        targets = array_ops.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=probs_flat)
        else:
            crossent = softmax_loss_function(probs_flat, targets)
        crossent = crossent * array_ops.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent)
            total_size = math_ops.reduce_sum(weights)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        else:
            batch_size = array_ops.shape(logits)[0]
            sequence_length = array_ops.shape(logits)[1]
            crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
        if average_across_timesteps and not average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[1])
            total_size = math_ops.reduce_sum(weights, axis=[1])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if not average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[0])
            total_size = math_ops.reduce_sum(weights, axis=[0])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if max_across_timesteps:
            crossent = math_ops.reduce_max(crossent, axis=[1])
            crossent = math_ops.reduce_mean(crossent, axis=[0])
        return crossent

def hinge_loss(logits, targets, delta):
    logits = math_ops.to_float(logits)
    targets = math_ops.to_float(targets)
    correct_label_scores = math_ops.reduce_sum(math_ops.multiply(logits, 1-targets), axis=-1)
    incorrect_label_scores = math_ops.reduce_sum(math_ops.multiply(logits, targets), axis=-1)
    incrrect_correct_different = (incorrect_label_scores - correct_label_scores)
    target_output = tf.cast(targets[:, -1], dtype=tf.float32)
    loss = math_ops.maximum(delta - tf.reduce_sum(math_ops.multiply(incrrect_correct_different, target_output)) / tf.reduce_sum(target_output),
                            delta - tf.reduce_sum(math_ops.multiply(incrrect_correct_different, (1-target_output))) / tf.reduce_sum(1-target_output)
                            )
    # loss = tf.reduce_mean(math_ops.maximum(0.0, delta - incrrect_correct_different))
    return loss

def cos_dist_loss(emb1, emb2):
    normalize_a = tf.nn.l2_normalize(emb1, 1)
    normalize_b = tf.nn.l2_normalize(emb2, 1)
    cos_distance = 1 - tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
    return cos_distance

def get_device_str(num_gpus, gpu_rellocate=False):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:0"
    if num_gpus > 1 and gpu_rellocate:
        device_str_output = "/gpu:1"
    return device_str_output

def make_cell(rnn_size, device_str, trainable=True):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, trainable=trainable)
    enc_cell = tf.contrib.rnn.DeviceWrapper(enc_cell, device_str)
    print("  %s, device=%s" % (type(enc_cell).__name__, device_str))
    return enc_cell