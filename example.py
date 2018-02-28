
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import input_data
import pathnet
import os
import numpy as np
import time
import random
from PIL import Image
import scipy.misc as misc
import imagenet_data

data_folder_task1 = '../data_set/imagenet/task1'
data_folder_task2 = '../data_set/imagenet/task2'


filename, label = imagenet_data.create_file_queue(data_folder_task1)
filename2, label2 = imagenet_data.create_file_queue(data_folder_task2)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(10):
    batch_images, batch_labels = imagenet_data.read_batch(sess, filename, label, 10, data_folder_task1)
#print(np.array(batch_images).shape)
#print(batch_labels[0])
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
for i in range(10):
    batch_images, batch_labels = imagenet_data.read_batch(sess, filename2, label2, 10, data_folder_task2)
#print(np.array(batch_images).shape)
#print(batch_labels[0])

coord.request_stop()
coord.join(threads)