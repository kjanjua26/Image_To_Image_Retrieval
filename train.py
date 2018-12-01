import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
from dataset import BatchGenerator, get_mnist, get_fashion
from model import *
import os
from preprocess import get_split
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 50000, 'Total training iter')
flags.DEFINE_integer('step', 500, 'Save after ... iteration')

mnist = get_fashion()
gen = BatchGenerator(mnist.train.images, mnist.train.labels)
test_im = np.array([im.reshape((28,28,1)) for im in mnist.test.images])
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left')
right = tf.placeholder(tf.float32, [None, 28, 28, 1], name='right')
with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2
left_output = inference(left, reuse=False)
right_output = inference(right, reuse=True)
loss = contrastive_loss2(left_output, right_output, label, margin)
global_step = tf.Variable(0, trainable=False)
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
inf_norm = tf.norm(reg_ws, np.inf)
#reg = tf.sqrt(tf.reduce_sum(tf.square(reg_ws))) # regularizer: forbenius norm as regularization
tot_loss = loss + 0.01 * inf_norm
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(tot_loss, global_step=global_step)

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#setup tensorboard	
	tf.summary.scalar('step', global_step)
	tf.summary.scalar('loss', loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train.log', sess.graph)

	#train iter
	for i in range(FLAGS.train_iter):
		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

		_, l, summary_str = sess.run([train_step, loss, merged], 
			feed_dict={left:b_l, right:b_r, label: b_sim})
		
		writer.add_summary(summary_str, i)
		print("\r#%d - Loss"%i, l)

		
		if (i + 1) % FLAGS.step == 0:
			#generate test
			feat = sess.run(left_output, feed_dict={left:test_im})
			
			labels = mnist.test.labels
			# plot result
			f = plt.figure(figsize=(16,9))
			for j in range(10):
			    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
			    	'.', c=c[j],alpha=0.8)
			plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
			plt.savefig('img/%d.jpg' % (i + 1))

	saver.save(sess, "model/model.ckpt")





