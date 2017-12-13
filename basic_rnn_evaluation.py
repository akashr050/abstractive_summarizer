import tensorflow as tf
import os
import utils
import bleu
import input_generator
from model import RNNModel

flags = tf.app.flags
slim = tf.contrib.slim

output_file = './workspace/title_out.txt'
IS_TRAINING = True
EOS_CHAR, EOS_INDEX = utils.symbols['EOS_CHAR'], utils.symbols['EOS_INDEX']
UNK_CHAR, UNK_INDEX = utils.symbols['UNK_CHAR'], utils.symbols['UNK_INDEX']
SOS_CHAR, SOS_INDEX = utils.symbols['SOS_CHAR'], utils.symbols['SOS_INDEX']


flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('hidden_dim', 512, '')
flags.DEFINE_integer('num_units', 3, '')
flags.DEFINE_integer('embed_dim', 100, '')
flags.DEFINE_integer('learning_rate', 0.001, '')
flags.DEFINE_float('clip_gradient_norm', 4, '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_string('embedding_file', 'new_glove.txt', '')
flags.DEFINE_string('titles_file', 'AbsSumm_title_60k.pkl', '')
flags.DEFINE_string('paras_file', 'AbsSumm_text_60k.pkl', '')
flags.DEFINE_float('dropout', 0.2, '')
flags.DEFINE_integer('test_size', 5000, '')
flags.DEFINE_integer('train_size', 60000, '')
flags.DEFINE_integer('maximum_iterations', 60, '')

FLAGS = flags.FLAGS
data_root_dir = './workspace'
paras_file = FLAGS.paras_file
titles_file = FLAGS.titles_file
# embedding_file = 'glove.6B.100d.txt'
embedding_file = FLAGS.embedding_file
ckpt_dir = './checkpoints'
utils.force_mkdir(ckpt_dir)

workspace_path = lambda file_path: os.path.join(data_root_dir, file_path)
paras_file, titles_file, embedding_file = workspace_path(paras_file), \
  workspace_path(titles_file), workspace_path(embedding_file)

input_data, inputs_for_tf, input_placeholders = input_generator.get_inputs(
  paras_file, titles_file, embedding_file, FLAGS)

###########
## Model ##
###########
outputs = RNNModel(inputs_for_tf, FLAGS, is_training=IS_TRAINING, multirnn=False)
blue_score = bleu.bleu_score(predictions=outputs.sample_id,
                             labels=inputs_for_tf['title_batch'][:, 1:])

##############
## Training ##
##############
sess = tf.Session()
tf.tables_initializer().run(session=sess)
sess.run(tf.global_variables_initializer())
sess.run(input_placeholders['embedding_init'], feed_dict={
  input_placeholders['embedding_ph']: input_data['embedding']})
saver = tf.train.import_meta_graph('checkpoints.meta')
saver.restore(sess, tf.train.latest_checkpoint('.'))

sess.run(inputs_for_tf['iterator'].initializer, feed_dict={
  input_placeholders['paras_ph']: input_data['input_paras'],
  input_placeholders['titles_ph']: input_data['input_titles'],
  input_placeholders['batch_size']: FLAGS.batch_size})
outputs_to_write = []
i = 0
while True:
  try:
    outputs_to_write.extend(sess.run(outputs.sample_id, feed_dict={
      input_placeholders['batch_size']: 1}))
    print("Step: {}".format(i))
    i = i+1
  except tf.errors.OutOfRangeError:
    break

utils.generate_output(outputs_to_write, output_file, embedding_file)






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

