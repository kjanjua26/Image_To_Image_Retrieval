import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import pandas as pd
import scipy.misc
from dataset import get_mnist, get_fashion
from model import *
from PIL import Image
from scipy.spatial.distance import cdist
from matplotlib import gridspec
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#(x_train, y_train),(x_test, y_test) = get_cifar()

mnist = get_fashion()
train_images = np.array([im.reshape((28,28,1)) for im in mnist.train.images])
test_images = np.array([im.reshape((28,28,1)) for im in mnist.test.images])
len_test = len(mnist.test.images)
len_train = len(mnist.train.images)

print(len_test, len_train)


def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
    	scipy.misc.imsave(str(idx)+'.jpg', data[idxs[i],:,:,0])
    	ax = fig.add_subplot(gs[0,i])
    	ax.imshow(data[idxs[i],:,:,0])
    	ax.axis('off')
    plt.savefig('retrieval.png')
    plt.show()

img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='img')
net = inference(img_placeholder, reuse=False)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model.ckpt")
    
    train_feat = sess.run(net, feed_dict={img_placeholder:train_images[:10000]})
idx = np.random.randint(0, len_test)
print(idx)
im = test_images[idx]
#show the test image
show_image(idx, test_images)
print ("This is image from id:", idx)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model.ckpt")
    search_feat = sess.run(net, feed_dict={img_placeholder:[im]})
    
#calculate the cosine similarity and sort
dist = cdist(train_feat, search_feat, 'cosine')
rank = np.argsort(dist.ravel())
n = 10
#show the top n similar image from train data
show_image(rank[:n], train_images)
print ("retrieved ids:", rank[:n])
