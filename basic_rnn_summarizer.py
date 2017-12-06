import tensorflow as tf
import os
import utils
import bleu
import input_generator
from model import RNNModel

flags = tf.app.flags
slim = tf.contrib.slim

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
outputs = RNNModel(inputs_for_tf, FLAGS, is_training=True, multirnn=True)

##################
## Optimisation ##
##################
weights = tf.cast(tf.sequence_mask(inputs_for_tf['title_length']), tf.float32)
loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output,
                                                      targets=inputs_for_tf['title_batch'][:, 1:],
                                                      weights=weights,
                                                      average_across_timesteps=False))
loss_summary = tf.summary.scalar('training_loss', loss)
global_step = tf.Variable(0, trainable=False, name='global_step')
blue_score = bleu.bleu_score(predictions=outputs.sample_id,
                             labels=inputs_for_tf['title_batch'][:, 1:])
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss,
                                                                 global_step=global_step)
##############
## Training ##
##############
sess = tf.Session()
tf.tables_initializer().run(session=sess)
sess.run(tf.global_variables_initializer())
sess.run(input_placeholders['embedding_init'], feed_dict={
  input_placeholders['embedding_ph']: input_data['embedding']})
train_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
saver = tf.train.Saver()
train_bleu = tf.Variable(0.0, name='Train_bleu')
train_bleu_summary = tf.summary.scalar('Training Bleu', train_bleu)
val_bleu = tf.Variable(0.0, name='Validation_bleu')
val_bleu_summary = tf.summary.scalar('Validation Bleu', val_bleu)

for epoch in range(FLAGS.epochs):
  print("Training for epoch: {}".format(epoch))
  sess.run(inputs_for_tf['iterator'].initializer, feed_dict={
    input_placeholders['paras_ph']: input_data['input_paras'],
    input_placeholders['titles_ph']: input_data['input_titles'],
    input_placeholders['batch_size']: FLAGS.batch_size})
  while True:
    try:
      _, tb_summary = sess.run([optimizer, loss_summary], feed_dict={
        input_placeholders['batch_size']: FLAGS.batch_size})
      train_writer.add_summary(tb_summary, global_step=global_step.eval(session=sess))
      print("Global Step: {}".format(sess.run(global_step)))
    except tf.errors.OutOfRangeError:
      break
  saver.save(sess, ckpt_dir)
  if epoch % 50 == 0:
    # sess.run(iterator.initializer, feed_dict={paras_ph: input_paras,
    #                                           titles_ph: input_titles,
    #                                           batch_size: 1})
    # train_bleu_temp = get_bleu(sess, batch_size)
    # sess.run(train_bleu.assign(train_bleu_temp))
    # train_bleu_summ = sess.run(train_bleu_summary)
    # train_writer.add_summary(train_bleu_summ, global_step=epoch)
    # print('Training bleu {}'.format(train_bleu_temp))
    sess.run(inputs_for_tf['iterator'].initializer, feed_dict={
      input_placeholders['paras_ph']: input_data['input_paras'],
      input_placeholders['titles_ph']: input_data['input_titles'],
      input_placeholders['batch_size']: 1})
    val_bleu_temp = utils.get_bleu(sess, input_placeholders['batch_size'], blue_score)
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

