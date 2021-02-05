import numpy as np
import sys, six

def prepare_batch(inputs):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    max_sequence_length = max(sequence_lengths)
        
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], 
                                  dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    return inputs_batch_major, sequence_lengths

        
def batch_generator(x, y):            
    while True:
        i = np.random.randint(0, len(x))
        yield [x[i], y[i]]
        
def input_generator(x, y, batch_size):
    gen_batch = batch_generator(x, y)

    x_batch = []
    y_batch = []
    for i in range(batch_size):
        a, b= next(gen_batch)
        x_batch += [a]
        y_batch += [b]
    return x_batch, y_batch

def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  if six.PY2:
    sys.stdout.write(s.encode("utf-8"))
  else:
    sys.stdout.buffer.write(s.encode("utf-8"))

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def readlines(input_file):
    ret = []
    for line in open(input_file, 'r'):
        ret.append(line.strip())
    return ret

def write_lines(arr_list, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join(arr_list))
    return

import json
def write_numpy_array(emb_mat, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(emb_mat.tolist(), outfile)