import tensorflow as tf
import utils

EOS_CHAR, EOS_INDEX = utils.symbols['EOS_CHAR'], utils.symbols['EOS_INDEX']
UNK_CHAR, UNK_INDEX = utils.symbols['UNK_CHAR'], utils.symbols['UNK_INDEX']
SOS_CHAR, SOS_INDEX = utils.symbols['SOS_CHAR'], utils.symbols['SOS_INDEX']


def RNNModel(inputs, params, is_training=True, multirnn=True):
  para_embedding = inputs['para_embedding']
  para_length = inputs['para_length']
  title_embedding = inputs['title_embedding']
  title_length = inputs['title_length']
  embedding_W = inputs['embedding_W']
  batch_size = inputs['batch_size']
  vocab_size = embedding_W.get_shape().as_list()[0]

  #############
  ## Encoder ##
  #############
  if multirnn:
    encoder_cells = []
    for _ in range(params.num_units):
      cell = tf.contrib.rnn.GRUCell(params.hidden_dim)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - params.dropout)
      encoder_cells.append(cell)
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)
  else:
    encoder_cell = tf.contrib.rnn.GRUCell(params.hidden_dim)
  encoder_init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
  encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, para_embedding,
                                                           initial_state=encoder_init_state,
                                                           sequence_length=para_length,
                                                           dtype=tf.float32)
  if is_training:
    helper = tf.contrib.seq2seq.TrainingHelper(title_embedding, title_length)
  else:
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_W,
                                                      start_tokens=tf.tile([SOS_INDEX], tf.reshape(batch_size, (1,))),
                                                      end_token=EOS_INDEX)

  #############
  ## Decoder ##
  #############
  if multirnn:
    decoder_cells = []
    for _ in range(params.num_units):
      cell = tf.contrib.rnn.GRUCell(params.hidden_dim)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - params.dropout)
      decoder_cells.append(cell)
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
  else:
    decoder_cell = tf.contrib.rnn.GRUCell(params.hidden_dim)
  decoder_output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size)

  decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_output_cell, helper=helper, initial_state=encoder_final_state)
  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                   output_time_major=False,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=60)
  return outputs
