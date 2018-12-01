import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from math import log
from keras import backend as K
flags = tf.app.flags
FLAGS = flags.FLAGS
import tflearn
from tensorflow.contrib.layers import flatten
import tensorflow.contrib.layers as initializers

def inference(input_images, reuse=False):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=32, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv1_2')
            #x = tf.nn.relu(x)
            #x = tf.layers.batch_normalization(x)
            x = slim.max_pool2d(x, scope='pool1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv2_2')
            #x = tf.nn.relu(x)
            #x = tf.layers.batch_normalization(x)
            x = slim.max_pool2d(x, scope='pool2')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), reuse=reuse, scope='conv3_2')
            #x = tf.nn.relu(x)
            x = slim.max_pool2d(x, scope='pool3')
            x = slim.flatten(x, scope='flatten')
            #x = tf.layers.batch_normalization(x) # added batch normalization layer to test
            embds = tf.nn.l2_normalize(x, 1, 1e-10, name='embeddings')
            x = 0.01 * embds # scaled layer
            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, reuse=reuse, scope='fc1')
            #x = tflearn.prelu(feature)
            #x = slim.fully_connected(x, num_outputs=10, activation_fn=None, reuse=reuse, scope='fc2')
    return feature

def lenet(x, reuse=False):
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16, 
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def mynet(input, reuse=False):
	with tf.name_scope("model"):
		with tf.variable_scope("conv1") as scope:
			net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv2") as scope:
			net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv3") as scope:
			net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv4") as scope:
			net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv5") as scope:
			net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		net = tf.contrib.layers.flatten(net)
	
	return net


def contrastive_loss2(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		softmax_1 = tf.nn.softmax(model1)
		ce_1 = -tf.reduce_sum(y * tf.log(softmax_1), 1)
		ce_1 = 0.1*ce_1
		coss_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(model1, 0), tf.nn.l2_normalize(model2, 0), dim=0)
		softmax_2 = tf.nn.softmax(model2)
		ce_2 = -tf.reduce_sum(y * tf.log(softmax_2), 1)
		ce_2 = ce_2*0.1

		d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
		tmp = y * tf.square(d)
		tmp2 = (1 - y) * tf.square(tf.maximum((	margin - d),0))
		k_norm = tf.pow(tf.reduce_sum(tf.pow(model1-model2,3)),0.3)
		
		tmp_coss = y * tf.square(coss_dist)
		tmp2_coss = (1 - y) * tf.square(tf.maximum((margin - coss_dist),0))
		
		tmp_k = y * tf.square(k_norm)
		tmp2_k = (1 - y) * tf.square(tf.maximum((margin - k_norm),0))
		
		loss = tf.reduce_mean(tmp+tmp2)/2 + tf.reduce_mean(ce_1+ce_2)/2 + tf.reduce_mean(tmp_coss + tmp2_coss)/2 
		return loss
		#margin_2 = 1
		#return K.mean(model1 * K.square(model2) + (1 - model1) * K.square(K.maximum(margin_2 - model2, 0)))/2

def contrastive_loss(left_output, right_output, y, margin):
 	label = tf.to_float(y)
 	one = tf.constant(1.0)
 	d = tf.reduce_sum(tf.square(tf.subtract(left_output, right_output)),1)
 	d_sqrt = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(left_output, right_output)),1))
 	first_part = tf.multiply(one-label, d)# (Y-1)*(d)
 	max_part = tf.square(tf.maximum(margin-d_sqrt, 0))
 	second_part = tf.multiply(label, max_part)  # (Y) * max(margin - d, 0)
 	loss = 0.5 * tf.reduce_mean(first_part + second_part)
 	return loss

def marginal_loss(model1, model2, y, margin, threshold):

	margin_ = 1/(tf.pow(margin,2)-margin)
	tmp = (1. - y)
	euc_dist = tf.reduce_sum(tf.square(tf.nn.l2_normalize(model1) - tf.nn.l2_normalize(model2)), 1)
	#euc_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(model1, 0), tf.nn.l2_normalize(model2, 0), dim=0)
	thres_dist = threshold - euc_dist
	mul_val = tf.multiply(tmp, thres_dist)
	sum_ = tf.reduce_sum(mul_val)
	softmax_1 = tf.nn.softmax(model1)
	ce_1 = -tf.reduce_sum(y * tf.log(softmax_1), 1)
	ce_1 = 0.1*ce_1
	softmax_2 = tf.nn.softmax(model2)
	ce_2 = -tf.reduce_sum(y * tf.log(softmax_2), 1)
	ce_2 = ce_2*0.1

	return tf.multiply(margin_, sum_) + tf.reduce_mean(ce_1+ce_2)/2






