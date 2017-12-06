import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.contrib import layers
from tensorflow.contrib.data import Dataset, Iterator
import bleu
from sklearn.model_selection import train_test_split


flags = tf.app.flags
slim = tf.contrib.slim

EOS_CHAR, EOS_INDEX = '<EOS>', 0
UNK_CHAR, UNK_INDEX = '<UNK>', 1
SOS_CHAR, SOS_INDEX = '<SOS>', 2
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('hidden_dim', 512, '')
flags.DEFINE_integer('num_units', 3, '')
flags.DEFINE_integer('embed_dim', 100, '')
flags.DEFINE_integer('learning_rate', 0.001, '')
flags.DEFINE_float('clip_gradient_norm', 4, '')
flags.DEFINE_integer('epochs', 10000, '')
flags.DEFINE_string('embedding_file', 'new_glove.txt', '')
flags.DEFINE_string('titles_file', 'AbsSumm_title_60k.pkl', '')
flags.DEFINE_string('paras_file', 'AbsSumm_text_60k.pkl', '')

FLAGS = flags.FLAGS
data_root_dir = './workspace'
paras_file = FLAGS.paras_file
titles_file = FLAGS.titles_file
# embedding_file = 'glove.6B.100d.txt'
embedding_file = FLAGS.embedding_file
ckpt_dir = './checkpoints'

workspace_path = lambda file_path: os.path.join(data_root_dir, file_path)
paras_file, titles_file, embedding_file = workspace_path(paras_file), \
  workspace_path(titles_file), workspace_path(embedding_file)

try:
  os.mkdir(ckpt_dir)
except:
  pass

if paras_file.endswith('.pickle') or paras_file.endswith('pkl'):
  input_paras = pickle.load(open(paras_file,'rb'))
  input_titles = pickle.load(open(titles_file, 'rb'))

print("Data is loaded. It has {} rows".format(len(input_paras)))
input_paras, val_paras, input_titles, val_titles = train_test_split(input_paras, input_titles,
                                                                    test_size=5000, train_size=60000, shuffle=False)

###################
## Getting VOCAB ##
###################
def loadGlove(embedding_file):
  vocab = [EOS_CHAR, UNK_CHAR, SOS_CHAR]
  embedding = [np.zeros((FLAGS.embed_dim,)), np.random.normal(size=(FLAGS.embed_dim,)), np.ones((FLAGS.embed_dim,))]
  if embedding_file.endswith('txt'):
    file = open(embedding_file, 'r+')
    for index, line in enumerate(file.readlines()):
      row = line.strip().split(' ')
      vocab.append(row[0])
      embedding.append([float(x) for x in row[1:]])
    print('Glove word vectors are Loaded!')
    file.close()
  return vocab, np.asarray(embedding)

def get_bleu(sess, batch_size):
  bleu_score_temp = []
  while True:
    try:
      bleu_score_temp.append(sess.run(blue_score, feed_dict={batch_size: 1}))
    except tf.errors.OutOfRangeError:
      break
  return sum(bleu_score_temp) / len(bleu_score_temp)


def rev_vocab(vocab):
  return {idx: key for key, idx in vocab.iteritems()}

vocab, embedding = loadGlove(embedding_file)
embedding_W = tf.Variable(tf.constant(0.0, shape=embedding.shape), trainable=False, name='embedding_w')
embedding_placeholder = tf.placeholder(tf.float32, embedding.shape)
embedding_init = embedding_W.assign(embedding_placeholder)
# embedding_W = tf.Variable(embedding, trainable=False, name='embedding')

vocab = tf.contrib.lookup.index_table_from_tensor(mapping=vocab, default_value=UNK_INDEX)

#######################
## Data manipulation ##
#######################
def _input_parse_function(para, title):
  def parse_input(text, src=None):
    words = tf.string_split([text]).values
    size = tf.size(words)
    words = vocab.lookup(words)
    if src == 'Target':
      words = tf.concat([tf.constant(SOS_INDEX, dtype=tf.int64, shape=[1,]), words], axis=0)
    return (words, size)
  return (parse_input(para), parse_input(title, src='Target'))

paras_ph = tf.placeholder(tf.string, shape=(None,))
titles_ph = tf.placeholder(tf.string, shape=(None,))
# is_training = tf.placeholder(tf.bool, shape=())
is_training = True
batch_size = tf.placeholder(tf.int32, shape=())
# paras_ph = input_paras
# titles_ph = input_titles
data = Dataset.from_tensor_slices((paras_ph, titles_ph))
data = data.map(_input_parse_function, num_parallel_calls=8).prefetch(FLAGS.batch_size * 10)
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

###########
## Model ##
###########

# [TODO: Add the multirnncell code]
# encoder_cells = [tf.contrib.rnn.GRUCell(FLAGS.hidden_dim)] * FLAGS.num_units
# multi_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)
# initial_state = multi_encoder_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
# encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(multi_encoder_cell, para_embedding,
#                                                          sequence_length=para_length,
#                                                          dtype=tf.float32)

encoder_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_dim)
encoder_init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, para_embedding,
                                                         initial_state=encoder_init_state,
                                                         sequence_length=para_length,
                                                         dtype=tf.float32)

# Decoder
if is_training:
  helper = tf.contrib.seq2seq.TrainingHelper(title_embedding, title_length)
else:
  helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_W,
                                                    start_tokens=tf.tile([SOS_INDEX], tf.reshape(batch_size, (1,))),
                                                    end_token=EOS_INDEX)

decoder_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_dim)
decoder_output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, embedding.shape[0])
decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_output_cell, helper=helper, initial_state=encoder_final_state)
outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                 output_time_major=False,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=60)
# TODO: Add variable for maximum iterations
blue_score = bleu.bleu_score(predictions=outputs.sample_id, labels=title_batch[:, 1:])

##################
## Optimisation ##
##################
weights = tf.cast(tf.sequence_mask(title_length), tf.float32)
loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=title_batch[:, 1:],
                                                      weights=weights, average_across_timesteps=False))
loss_summary = tf.summary.scalar('training_loss', loss)
global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

##############
## Training ##
##############

sess = tf.Session()
tf.tables_initializer().run(session=sess)
sess.run(tf.global_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
train_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
saver = tf.train.Saver()
train_bleu = tf.Variable(0.0, name='Train_bleu')
train_bleu_summary = tf.summary.scalar('Training Bleu', train_bleu)
val_bleu = tf.Variable(0.0, name='Validation_bleu')
val_bleu_summary = tf.summary.scalar('Validation Bleu', val_bleu)

for epoch in range(FLAGS.epochs):
  print("Training for epoch: {}".format(epoch))
  sess.run(iterator.initializer, feed_dict={paras_ph: input_paras,
                                            titles_ph: input_titles,
                                            batch_size: FLAGS.batch_size})
  while True:
    try:
      _, tb_summary = sess.run([optimizer, loss_summary], feed_dict={batch_size: FLAGS.batch_size})
      train_writer.add_summary(tb_summary, global_step=global_step.eval(session=sess))
      print("Global Step: {}".format(sess.run(global_step)))
    except tf.errors.OutOfRangeError:
      break
  if epoch % 10 == 0:
    saver.save(sess, ckpt_dir)
    # sess.run(iterator.initializer, feed_dict={paras_ph: input_paras,
    #                                           titles_ph: input_titles,
    #                                           batch_size: 1})
    # train_bleu_temp = get_bleu(sess, batch_size)
    # sess.run(train_bleu.assign(train_bleu_temp))
    # train_bleu_summ = sess.run(train_bleu_summary)
    # train_writer.add_summary(train_bleu_summ, global_step=epoch)
    # print('Training bleu {}'.format(train_bleu_temp))
    sess.run(iterator.initializer, feed_dict={paras_ph: val_paras,
                                              titles_ph: val_titles,
                                              batch_size: 1})
    val_bleu_temp = get_bleu(sess, batch_size)
    sess.run(val_bleu.assign(val_bleu_temp))
    val_bleu_summ = sess.run(val_bleu_summary)
    train_writer.add_summary(val_bleu_summ, global_step=epoch)
    print('Validation bleu {}'.format(val_bleu_temp))




# slim.learning.train(train_op=train_op,
#                     logdir=FLAGS.ckpt_dir,
#                     number_of_steps=FLAGS.max_number_of_steps,
#                     saver=saver,
#                     save_summaries_secs=FLAGS.save_summaries_secs,
#                     save_interval_secs=FLAGS.save_internal_secs)
# a, b, e, c, d, f = sess.run([para_batch, title_batch, title_batch[:, 1:], outputs.rnn_output, outputs.sample_id,
#                           blue_score])
# # a = 1
#
#
# ################
# ## Evaluation ##
# ################
# # TODO[Remove the Blue score

