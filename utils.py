import os
import numpy as np
import tensorflow as tf

symbols = dict()
symbols['EOS_CHAR'], symbols['EOS_INDEX'] = '<EOS>', 0
symbols['UNK_CHAR'], symbols['UNK_INDEX'] = '<UNK>', 1
symbols['SOS_CHAR'], symbols['SOS_INDEX'] = '<SOS>', 2


def force_mkdir(dir_path):
  try:
    os.mkdir(dir_path)
  except:
    pass


def loadGlove(embedding_file, params):
  EOS_CHAR, SOS_CHAR, UNK_CHAR = symbols['EOS_CHAR'], symbols['SOS_CHAR'], symbols['UNK_CHAR']
  vocab = [EOS_CHAR, UNK_CHAR, SOS_CHAR]
  embedding = [np.zeros((params.embed_dim,)), np.random.normal(size=(params.embed_dim,)), np.ones((params.embed_dim,))]
  if embedding_file.endswith('txt'):
    file = open(embedding_file, 'r+')
    for index, line in enumerate(file.readlines()):
      row = line.strip().split(' ')
      vocab.append(row[0])
      embedding.append([float(x) for x in row[1:]])
    print('Glove word vectors are Loaded!')
    file.close()
  return vocab, np.asarray(embedding)


def get_bleu(sess, batch_size, bleu_score):
  bleu_score_temp = []
  while True:
    try:
      bleu_score_temp.append(sess.run(bleu_score, feed_dict={batch_size: 1}))
    except tf.errors.OutOfRangeError:
      break
  return sum(bleu_score_temp) / len(bleu_score_temp)


def rev_vocab(vocab):
  rev_vocab = dict()
  for index, val in enumerate(vocab):
    rev_vocab[index] = val
  return rev_vocab

def generate_output(output, output_file, embedding_file):
  vocab, embedding = loadGlove(embedding_file)
  reverse_vocab = rev_vocab(vocab)
  output_file = open(output_file, 'w+')
  for line in output:
    temp_summary = np.vectorize(reverse_vocab.get)(line)
    a = ' '.join(temp_summary[:-1]) + '\n'
    output_file.write(a)
  output_file.close()
