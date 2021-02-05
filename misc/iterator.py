"""
Input data iteration
Author:         Ying Xu
Date:           Jul 8, 2020
"""

import tensorflow as tf

def get_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True,
                 min_len=0):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > min_len, tf.size(tgt) > min_len))

    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.concat(([sos_id], src, [eos_id]), 0),
                              tf.concat(([sos_id], src), 0),
                              tf.concat((src, [eos_id]), 0)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src,
                            tf.concat(([sos_id], tgt), 0),
                            tf.concat((tgt, [eos_id]), 0)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    pad_id = eos_id
    if sos == '[CLS]':
        pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_input
                tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_id,  # src
                eos_id,  # tgt_input
                eos_id,  # tgt_output
                0,  # src_len -- unused
                0))  # tgt_len -- unused

    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
          # Calculate bucket_width by maximum source sequence length.
          # Pairs with length [0, bucket_width) go to bucket 0, length
          # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
          # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
          if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
          else:
            bucket_width = 10

          # Bucket sentence pairs by the length of their source sentence and target
          # sentence.
          bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
          return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
          return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter


def get_adv_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 sep,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
    sep_id = tf.cast(vocab_table.lookup(tf.constant(sep)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(tf.strings.to_number(tgt, tf.float32), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.concat(([sos_id], src, [eos_id]), 0),
                              tf.concat(([sos_id], src), 0),
                              tf.concat((src, [eos_id]), 0),
                              tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src,
                            tf.concat(([sos_id], src), 0),
                            tf.concat((src, [eos_id]), 0),
                            tgt),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_in, src_out, tgt: (
            src, src_in, src_out, tgt, tf.size(src), tf.size(src_out),
            tf.cast(tf.reduce_min(tf.where(tf.equal(src_out, sep_id))), tf.int32)
        ),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    pad_id = eos_id
    if sos == '[CLS]':
        pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # src_input
                tf.TensorShape([None]),  # src_output
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([]),  # tgt_len
                tf.TensorShape([]),  # prem length
            ),
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_id,  # src
                eos_id,  # src_input
                eos_id,  # src_output
                0,  # tgt -- unused
                0,  # src_len -- unused
                0,  # src_output_len -- unused
                0,  # prem_len -- unused
            ))

    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, unused_3, unused4, src_len, tgt_len, prem_len):
          # Calculate bucket_width by maximum source sequence length.
          # Pairs with length [0, bucket_width) go to bucket 0, length
          # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
          # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
          if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
          else:
            bucket_width = 10

          # Bucket sentence pairs by the length of their source sentence and target
          # sentence.
          bucket_id = src_len // bucket_width
          return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
          return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter


def get_cls_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 sep,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
    sep_id = tf.cast(vocab_table.lookup(tf.constant(sep)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.size(src) > 0)


    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(tf.strings.to_number(tgt, tf.float32), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.concat(([sos_id], src, [eos_id]), 0), tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.concat((src, [eos_id]), 0), tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            src, tgt, tf.size(src), tf.cast(tf.reduce_min(tf.where(tf.equal(src, sep_id))), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    pad_id = eos_id
    if sos == '[CLS]':
        pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]), # src_len
                tf.TensorShape([])
            ),
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_id,  # src
                0,  # tgt -- unused
                0, # src_len -- unused
                0
            ))


    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, src_len, unused_3):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter


def get_cls_def_iterator(src_datasets,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip(tuple(src_datasets) + (tgt_dataset, ))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def, tgt: (
            tf.string_split([src]).values, tf.string_split([src_def]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, src_def, tgt: tf.size(src) > 0)


    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, src_def, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def, tgt: (src[:src_max_len], src_def, tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                                   tf.cast(vocab_table.lookup(src_def), tf.int32),
                          tf.cast(tf.strings.to_number(tgt, tf.float32), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def, tgt: (tf.concat(([sos_id], src, [eos_id]), 0),
                                       tf.concat(([sos_id], src_def, [eos_id]), 0), tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def, tgt: (tf.concat((src, [eos_id]), 0), tf.concat((src_def, [eos_id]), 0), tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def, tgt: (
            src, tgt, tf.size(src), src_def, tf.size(src_def)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    pad_id = eos_id
    if sos == '[CLS]':
        pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]),
                tf.TensorShape([None]),  # src_def
                tf.TensorShape([]),
            ),  # src_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_id,  # src
                0,  # tgt -- unused
                0,
                pad_id,
                0
            )
        )  # src_len -- unused


    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, src_len, unused_3, unused_4):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter


def get_cls_multi_def_iterator(src_datasets,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip(tuple(src_datasets) + (tgt_dataset, ))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (
            tf.string_split([src]).values,
            tf.string_split([src_def1]).values,
            tf.string_split([src_def2]).values,
            tf.string_split([src_def3]).values,
            tf.string_split([src_def4]).values,
            tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, src_def1, src_def2, src_def3, src_def4, tgt: tf.size(src) > 0)


    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (src[:src_max_len],
                                                                      src_def1, src_def2, src_def3, src_def4, tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                                                                  tf.cast(vocab_table.lookup(src_def1), tf.int32),
                                                                  tf.cast(vocab_table.lookup(src_def2), tf.int32),
                                                                  tf.cast(vocab_table.lookup(src_def3), tf.int32),
                                                                  tf.cast(vocab_table.lookup(src_def4), tf.int32),
                                                                  tf.cast(tf.strings.to_number(tgt, tf.float32), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (tf.concat(([sos_id], src, [eos_id]), 0),
                                                                      tf.concat(([sos_id], src_def1, [eos_id]), 0),
                                                                      tf.concat(([sos_id], src_def2, [eos_id]), 0),
                                                                      tf.concat(([sos_id], src_def3, [eos_id]), 0),
                                                                      tf.concat(([sos_id], src_def4, [eos_id]), 0), tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (tf.concat((src, [eos_id]), 0),
                                                                      tf.concat((src_def1, [eos_id]), 0),
                                                                      tf.concat((src_def2, [eos_id]), 0),
                                                                      tf.concat((src_def3, [eos_id]), 0),
                                                                      tf.concat((src_def4, [eos_id]), 0),
                                                                      tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_def1, src_def2, src_def3, src_def4, tgt: (
            src, tgt, tf.size(src), src_def1, src_def2, src_def3, src_def4,
            tf.size(src_def1),tf.size(src_def2), tf.size(src_def3), tf.size(src_def4)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    pad_id = eos_id
    if sos == '[CLS]':
        pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]),
                tf.TensorShape([None]),  # src_def1
                tf.TensorShape([None]),  # src_def2
                tf.TensorShape([None]),  # src_def3
                tf.TensorShape([None]),  # src_def4
                tf.TensorShape([]),     # src_def_len1
                tf.TensorShape([]),  # src_def_len2
                tf.TensorShape([]),  # src_def_len3
                tf.TensorShape([]),  # src_def_len4
            ),  # src_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_id,  # src
                0,  # tgt -- unused
                0,
                pad_id,
                pad_id,
                pad_id,
                pad_id,
                0,
                0,
                0,
                0
            )
        )  # src_len -- unused


    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, src_len, unused_31, unused_32,unused_33,unused_34, unused_41, unused_42, unused_43, unused_44):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter


def get_adv_cf_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 num_epochs,
                 sos,
                 eos,
                 sep,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 is_training=True,
                 ae_vocab_table=None):

    num_parallel_calls = 4

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(ae_vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(ae_vocab_table.lookup(tf.constant(eos)), tf.int32)
    sep_id = tf.cast(ae_vocab_table.lookup(tf.constant(sep)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if is_training:
        src_tgt_dataset = src_tgt_dataset.shard(1, 0)
        src_tgt_dataset = src_tgt_dataset.repeat(num_epochs)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed, True)

    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len is not None:
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: (tf.size(src) < src_max_len+1))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(ae_vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(tf.strings.to_number(tgt, tf.float32), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    if sos == '[CLS]':
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, cls_src, tgt: (src,
                              tf.concat(([1], src), 0),
                              tf.concat((src, [2]), 0),
                              tgt,
                              tf.concat(([sos_id], cls_src, [eos_id]), 0)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
          lambda src, cls_src, tgt: (src,
                            tf.concat(([sos_id], src), 0),
                            tf.concat((src, [eos_id]), 0),
                            tgt,
                            tf.concat((cls_src, [eos_id]), 0)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, src_in, src_out, tgt, cls_src_in: (
            src, src_in, cls_src_in, tgt, tf.size(src), tf.size(src_out), src_out,
            tf.cast(tf.reduce_min(tf.where(tf.equal(src_out, sep_id))), tf.int32)
        ),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    if sos == '[CLS]':
        ae_pad_id = 2
        cls_pad_id = tf.cast(vocab_table.lookup(tf.constant('[PAD]')), tf.int32)
    else:
        ae_pad_id = eos_id
        cls_pad_id = eos_id

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # src_input
                tf.TensorShape([None]),  # src_output
                tf.TensorShape([None]),  # tgt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([]),
                tf.TensorShape([None]),  # dec_target
                tf.TensorShape([]),  # prem_len
            ),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                ae_pad_id,  # src
                ae_pad_id,  # src_input
                cls_pad_id,  # src_output
                0,  # tgt -- unused
                0,  # src_len -- unused
                0,
                ae_pad_id, # dec_target
                0,  # prem_len -- unused
            ), )  # src_output_len -- unused

    if num_buckets > 1 and is_training:

        def key_func(unused_1, unused_2, unused_3, unused4, src_len, tgt_len, cls_src_in, prem_len):
          # Calculate bucket_width by maximum source sequence length.
          # Pairs with length [0, bucket_width) go to bucket 0, length
          # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
          # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
          if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
          else:
            bucket_width = 10

          # Bucket sentence pairs by the length of their source sentence and target
          # sentence.
          bucket_id = src_len // bucket_width
          return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
          return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    return batched_iter