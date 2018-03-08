from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys,os,time
import subprocess
import scipy.io as sio
import tensorflow as tf
from six.moves import urllib
import cifar10
import pathnet
import numpy as np
from candidate import Candidate

FLAGS = None

def load_cifar10_data():
  # Get CIFAR 10  dataset
  cifar10.maybe_download_and_extract();
  tr_label_cifar10=np.zeros((50000,10),dtype=float);
  ts_label_cifar10=np.zeros((10000,10),dtype=float);
  for i in range(1,6):
    file_name=os.path.join(FLAGS.cifar_data_dir,"data_batch_"+str(i)+".bin");
    f = open(file_name,"rb");
    data=np.reshape(bytearray(f.read()),[10000,3073]);
    if(i==1):
      tr_data_cifar10=data[:,1:]/255.0;
    else:
      tr_data_cifar10=np.append(tr_data_cifar10,data[:,1:]/255.0,axis=0);
    for j in range(len(data)):
      tr_label_cifar10[(i-1)*10000+j,data[j,0]]=1.0;
  file_name=os.path.join(FLAGS.cifar_data_dir,"test_batch.bin");
  f = open(file_name,"rb");
  data=np.reshape(bytearray(f.read()),[10000,3073]);
  for i in range(len(data)):
    ts_label_cifar10[i,data[i,0]]=1.0;
  ts_data_cifar10=data[:,1:]/255.0;
  data_num_len_cifar10=len(tr_label_cifar10);
  
  return tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10

def train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate, max_steps):

  #candidate.display_structure()
  # define local variables
  tr_data1=tr_data_cifar10;
  tr_label1=tr_label_cifar10;
  data_num_len1=data_num_len_cifar10;
  max_data_len= int(data_num_len1 / FLAGS.batch_num) # avoid [a:b], a will greater than b

  
  L = int(candidate.feature_layer_num + candidate.fc_layer_num + 1) # +1 for first conv layer
  M = int(candidate.module_num)
  F = int(candidate.filter_num) * 2  # due to filter number must be an even number
  FC = int(candidate.fc_layer_num)
  FL = candidate.feature_layer_array
  
  
  ## TASK 1
  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 32*32*3], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 32, 32, 3])

  ## need to change when involve task2
  # geopath_examples
  geopath=pathnet.geopath_initializer(L, M);

  ## need to change when involve task2
  # fixed weights list
  fixed_list=np.ones((L, M),dtype=str);
  for i in range(L):
    for j in range(M):
      fixed_list[i,j]='0';    

  # record weights, biases  and sum_weights that need to change
  weights_list=np.zeros((L, M),dtype=object);
  biases_list=np.zeros((L, M),dtype=object);
  sum_weights_list=np.zeros((L,M), dtype=object)
  for i in range(L):
    for j in range(M):
      _initial1 = tf.truncated_normal(shape=[1], mean=1, stddev=0.1)
      _initial2 = tf.truncated_normal(shape=[1], mean=1, stddev=0.1)
      sum_weights_list[i,j]= [tf.Variable(_initial1), tf.Variable(_initial2)]

  ## model define
  layer_modules_list=np.zeros(M,dtype=object)

  # first layer: conv 
  for j in range(M):
    layer_modules_list[j], weights_list[0,j], biases_list[0,j] = pathnet.conv_module(sum_weights_list[0][j][1] * image_shaped_input, F, [5,5], geopath[0,j], 1,  'conv_layer'+str(0+1)+"_"+str(j+1))

  net=np.sum(map(lambda (a,b):a*b[0], zip(layer_modules_list , sum_weights_list[0])))/ M

  # feature abstract layers
  for i in range(len(FL)):
    if FL[i] == 0:
      for j in range(M):
        layer_modules_list[j], weights_list[i + 1,j], biases_list[i + 1,j] = pathnet.res_fire_layer(sum_weights_list[i+1,j][1] * net, geopath[i + 1,j], 'res_fire_layer'+str(i+2)+"_"+str(j+1))
    else:
      # check dimension_reduction input whether to small 1*1
      if int(net.get_shape()[1]) == 1 and int(net.get_shape()[2]) == 1:
        candidate.disable_mask[i] = 1
        continue

      for j in range(M):
        layer_modules_list[j], weights_list[i + 1,j], biases_list[i + 1,j] = pathnet.Dimensionality_reduction_module(sum_weights_list[i+1,j][1] * net, geopath[i + 1,j], 'dimension_reduction_layer'+str(i+2)+"_"+str(j+1))    
    net=np.sum(map(lambda (a,b):a*b[0], zip(layer_modules_list , sum_weights_list[i+1])))/ M

  # full connection layer
    # reshape
  _shape = net.shape[1:]
  _length = 1
  for _i in _shape:
      _length *= int(_i)
  net=tf.reshape(net,[-1,_length])

    # full connection
  for i in range(L)[len(FL)+1:]:
    for j in range(M):
      layer_modules_list[j], weights_list[i,j], biases_list[i, j] = pathnet.fc_layer(sum_weights_list[i][j][1] * net, F, geopath[i,j], 'fc_layer'+str(i+1)+"_"+str(j+1))
    net=np.sum(map(lambda (a,b):a*b[0], zip(layer_modules_list , sum_weights_list[i])))/ M   

  # output layer
  y, output_weights ,output_biases = pathnet.nn_layer(net, 10, 'output_layer'+str(i))
    
  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  
  # Need to learn variables
  var_list_to_learn=[]+output_weights+output_biases;
  for i in range(L):
    # disabled layer don't have argvs to learn
    if i > 0 and i < candidate.maxFr+1 and candidate.disable_mask[i-1] == 1:
      continue
    for j in range(M):
      if (fixed_list[i,j]=='0'):
        var_list_to_learn+=weights_list[i,j]+biases_list[i,j]+sum_weights_list[i,j]
  
  # GradientDescent 
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);

  # Accuracy 
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # init
  tf.global_variables_initializer().run()

  
  # Learning & Evaluating
    # Shuffle the data
  #idx=range(len(tr_data1))
  #np.random.shuffle(idx)
  #tr_data1=tr_data1[idx]
  #tr_label1=tr_label1[idx]

  step_list = [max_data_len for i in range(int(max_steps/max_data_len))] + [max_steps%max_data_len]
  counter = 0
  acc_geo_tr = 0
  #print(step_list)
  #print(max_data_len)
  #print("max_steps: "+max_steps)
  for s in step_list: 
    idx=range(len(tr_data1))
    np.random.shuffle(idx)
    tr_data1=tr_data1[idx]
    tr_label1=tr_label1[idx]
    for k in range(s):
      _, acc_geo_tmp = sess.run([train_step, accuracy], feed_dict={x:tr_data1[k*FLAGS.batch_num :(k+1)*FLAGS.batch_num,:],
                                                                y_:tr_label1[k*FLAGS.batch_num :(k+1)*FLAGS.batch_num,:]})

      acc_geo_tr+=acc_geo_tmp
      counter+=1
      if(counter > 100 and counter%1000 ==0 ):
        print("step %d, single_acc %f" % (counter , acc_geo_tmp))


  sess.close()
    
  return acc_geo_tr / max_steps


def main(_):
  # create log dir but not used
  FLAGS.log_dir+="cifar/"
  FLAGS.log_dir+=str(int(time.time()))
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  # read cifar10 dataset
  tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10 = load_cifar10_data()


  # ceate inti candidates
  candidate1 = Candidate()
  
  # set your structurs here
  candidate1.feature_layer_num = 3
  candidate1.feature_layer_array = [1, 0, 1] 
  candidate1.fc_layer_num = 0       
  candidate1.module_num = 2         
  candidate1.filter_num = 10 

  final_acc = train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate1, FLAGS.max_step)
  print("best structure1 avg_acc "+ str(final_acc))
  candidate1.display_structure()


  # ceate inti candidates
  candidate2 = Candidate()
  
  # set your structurs here
  candidate2.feature_layer_num = 4
  candidate2.feature_layer_array = [1, 0, 1, 0] 
  candidate2.fc_layer_num = 1       
  candidate2.module_num = 3         
  candidate2.filter_num = 10 

  final_acc = train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate2, FLAGS.max_step)
  print("best structure2 avg_acc "+ str(final_acc))
  candidate2.display_structure()


  # ceate inti candidates
  candidate3 = Candidate()
  
  # set your structurs here
  candidate3.feature_layer_num = 5
  candidate3.feature_layer_array = [1, 0, 1, 0, 1] 
  candidate3.fc_layer_num = 2       
  candidate3.module_num = 4         
  candidate3.filter_num = 10 

  final_acc = train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate3, FLAGS.max_step)
  print("best structure3 avg_acc "+ str(final_acc))
  candidate3.display_structure()


    # ceate inti candidates
  candidate4 = Candidate()
  
  # set your structurs here
  candidate4.feature_layer_num = 5
  candidate4.feature_layer_array = [1, 0, 1, 0, 1] 
  candidate4.fc_layer_num = 2       
  candidate4.module_num = 5         
  candidate4.filter_num = 16 

  final_acc = train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate4, FLAGS.max_step)
  print("best structure4 avg_acc "+ str(final_acc))
  candidate4.display_structure()


    # ceate inti candidates
  candidate5 = Candidate()
  
  # set your structurs here
  candidate5.feature_layer_num = 7
  candidate5.feature_layer_array = [1, 0, 1, 0, 1, 0, 0] 
  candidate5.fc_layer_num = 3       
  candidate5.module_num = 3         
  candidate5.filter_num = 16

  final_acc = train(tr_data_cifar10, tr_label_cifar10, data_num_len_cifar10, candidate5, FLAGS.max_step)
  print("best structure5 avg_acc "+ str(final_acc))
  candidate5.display_structure()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--cifar_data_dir', type=str, default='/dataset/cifar_data/cifar-10-batches-bin',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/DetaNet/',
                      help='Summaries log directry')

  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')

  parser.add_argument('--T', type=int, default=10,
                      help='The Number of epoch per each geopath')
  parser.add_argument('--batch_num', type=int, default=256,
                      help='The Number of batches per each geopath')
  parser.add_argument('--candi', type=int, default=10,
                      help='The Number of Candidates of geopath, should greater than 4')
  parser.add_argument('--max_generations', type = int,default = 20,
                      help='The Generation Number of Evolution')
  parser.add_argument('--max_step', type = int,default = 100000,
                      help='The max training step of final structure')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
