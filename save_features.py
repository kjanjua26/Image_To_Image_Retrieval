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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
fname = open('nolayer-test-feat-file.txt', 'w')
print('Extracting features')
for i in range(len_test):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model/model.ckpt")
        test_feat = sess.run(net, feed_dict={img_placeholder:[test_images[i]]})
        fname.write(str(test_feat))
        fname.write(',')
        fname.write(str(mnist.test.labels[i]))
        fname.write('\n')
    print(mnist.test.labels[i])
fname.close()
