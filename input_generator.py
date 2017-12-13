'''
This file is used by the basic_rnn_summarizer and basic_rnn_evaluation script to generate the input data
'''

import pickle
import utils
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from sklearn.model_selection import train_test_split

EOS_CHAR, EOS_INDEX = utils.symbols['EOS_CHAR'], utils.symbols['EOS_INDEX']
UNK_CHAR, UNK_INDEX = utils.symbols['UNK_CHAR'], utils.symbols['UNK_INDEX']
SOS_CHAR, SOS_INDEX = utils.symbols['SOS_CHAR'], utils.symbols['SOS_INDEX']

def get_inputs(paras_file, titles_file, embedding_file, params):
  '''
  This function returns a dictionary of input texts, inputs required for tensorflow operation and input
  placeholder and operations
  :param paras_file: This the file for the input text
  :param titles_file: This is the file for the input summaries
  :param embedding_file: This is the embedding file like Glove, word2vec, etc
  :param params: These are the flags required for the input text batch iterator
  :return:
    inputs: This is the dictionary corresponding to the input training and validation texts and summaries
    inputs_tf: This is the dictionary with the tensorflow variables which will be used as inputs to the RNN
               summarizer
    inputs_ph_op: This is the dictionary with the tensorflow placeholders and operations which we will use to
                  declare the session
  '''
  def _input_parse_function(para, title):
    '''
    This function is used to parse the input
    :param para: This is the input paragraph
    :param title: This is the input summary
    :return:
      A tuple for para as well as for title. This tuple consists of following two things:
       i) A list consisting of all the input words
       ii) Number of words in the input
    '''
    def parse_input(text, src=None):
      words = tf.string_split([text]).values
      size = tf.size(words)
      words = vocab.lookup(words)
      if src == 'Target':
        words = tf.concat([tf.constant(SOS_INDEX, dtype=tf.int64, shape=[1, ]), words], axis=0)
      return (words, size)

    return (parse_input(para), parse_input(title, src='Target'))

  if paras_file.endswith('.pickle') or paras_file.endswith('pkl'):
    input_paras = pickle.load(open(paras_file,'rb'))
    input_titles = pickle.load(open(titles_file, 'rb'))

  print("Data is loaded. It has {} rows".format(len(input_paras)))
  input_paras, val_paras, input_titles, val_titles = train_test_split(input_paras, input_titles,
                                                                      test_size=params.test_size,
                                                                      train_size=params.train_size, shuffle=False)
  ###################
  ## Getting VOCAB ##
  ###################
  vocab, embedding = utils.loadGlove(embedding_file, params)
  embedding_W = tf.Variable(tf.constant(0.0, shape=embedding.shape), trainable=False, name='embedding_w')
  embedding_ph = tf.placeholder(tf.float32, embedding.shape)
  embedding_init = embedding_W.assign(embedding_ph)
  # embedding_W = tf.Variable(embedding, trainable=False, name='embedding')
  vocab = tf.contrib.lookup.index_table_from_tensor(mapping=vocab, default_value=UNK_INDEX)

  #######################
  ## Data manipulation ##
  #######################

  paras_ph = tf.placeholder(tf.string, shape=(None,))
  titles_ph = tf.placeholder(tf.string, shape=(None,))
  batch_size = tf.placeholder(tf.int32, shape=())
  data = Dataset.from_tensor_slices((paras_ph, titles_ph))
  data = data.map(_input_parse_function, num_parallel_calls=8).prefetch(params.batch_size * 10)
  data = data.padded_batch(tf.cast(batch_size, dtype=tf.int64),
                           padded_shapes=((tf.TensorShape([None]),
                                           tf.TensorShape([])),
                                          (tf.TensorShape([None]),
                                           tf.TensorShape([]))),
                           padding_values=((tf.to_int64(EOS_INDEX), 0),
                                           (tf.to_int64(EOS_INDEX), 0)))
  iterator = data.make_initializable_iterator()
  (para_batch, para_length), (title_batch, title_length) = iterator.get_next()
  para_embedding = tf.nn.embedding_lookup(embedding_W, para_batch)
  title_embedding = tf.nn.embedding_lookup(embedding_W, title_batch)
  inputs = dict()
  inputs['input_paras'] = input_paras
  inputs['val_paras'] = val_paras
  inputs['input_titles'] = input_titles
  inputs['val_titles'] = val_titles
  inputs['embedding'] = embedding
  inputs_tf = dict()
  inputs_tf['para_embedding'] = para_embedding
  inputs_tf['title_embedding'] = title_embedding
  inputs_tf['para_batch'], inputs_tf['para_length'] = para_batch, para_length
  inputs_tf['title_batch'], inputs_tf['title_length'] = title_batch, title_length
  inputs_tf['embedding_W'] = embedding_W
  inputs_tf['batch_size'] = batch_size # Repeating batch_size as it is also required for the model building
  inputs_tf['iterator'] = iterator
  inputs_ph_op = dict()
  inputs_ph_op['paras_ph'], inputs_ph_op['titles_ph'] = paras_ph, titles_ph
  inputs_ph_op['embedding_ph'], inputs_ph_op['embedding_init'] = embedding_ph, embedding_init
  inputs_ph_op['batch_size'] = batch_size
  return inputs, inputs_tf, inputs_ph_op
