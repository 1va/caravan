""" Simple, end-to-end, LeNet-5-like convolutional model.
"""
from __future__ import print_function
import os
import sys
import time
#import urllib
import tensorflow.python.platform
import numpy as np
import pymongo
from sklearn import cross_validation
import tensorflow as tf
from bson import objectid
from PIL import Image

#VALIDATION_SIZE = 300    # min(400,datalimit/4)  # Size of the validation set.
BATCH_SIZE = 20
NUM_EPOCHS = 30
IMAGE_SIZE = 300
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 70*70
SEED = 6647#8  # Set to None for random seed.
state=1


def load_dataset():
    X = np.genfromtxt('tmp_images/assemble_x.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    y = np.genfromtxt('tmp_images/assemble_y.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    ids = np.genfromtxt('tmp_images/assemble_ids.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    X = X.reshape(X.shape[0],IMAGE_SIZE, IMAGE_SIZE,NUM_CHANNELS)/np.float32(PIXEL_DEPTH)-.5
    return X, y, X, y, X, y, ids

def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels, ids = load_dataset()
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, NUM_LABELS))
  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.
  validation_data_node = tf.constant(validation_data)
  test_data_node = tf.constant(test_data)
  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
  depth1=32
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, depth1],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([depth1]))
  depth2=64
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, depth1, depth2],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[depth2]))
  depth3=256
  hidden2_size = 24/4  #((IMAGE_SIZE-4)/2-4)/2
  fc1_weights = tf.Variable(  # fully connected, depth 512.    ! but input nodes kept in the shape of square !  for future assembly into larger image
      tf.truncated_normal([hidden2_size, hidden2_size, depth2, depth3],
                          stddev=0.1,
                          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[depth3]))
  depth4=4
  fc2_weights = tf.Variable(
      tf.truncated_normal([1, 1, depth3, depth4],
                          stddev=0.1,
                          seed=SEED))
  fc2_weights_partial = tf.Variable(
      tf.truncated_normal([1, 1, depth3, 1],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[depth4]))
  fc2_biases_partial = tf.Variable(tf.constant(0.1, shape=[1]))
  assembly_weights = tf.Variable(
      tf.truncated_normal([5, 5, depth4, 1],
                          stddev=0.01,
                          seed=SEED))
  assembly_biases = tf.Variable(tf.constant(0.1, shape=[1]))
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    conv1 = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv2 = tf.nn.conv2d(pool1,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Fully connected layer on pretrained patches 24x24 for single caravan sites.
    hidden_pool = tf.nn.conv2d(pool2,
                        fc1_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    hidden = tf.nn.relu(tf.nn.bias_add(hidden_pool, fc1_biases))
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    single_pool = tf.nn.conv2d(hidden,
                        fc2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    single = tf.nn.tanh(tf.nn.bias_add(single_pool, fc2_biases))
    out_pool = tf.nn.conv2d(single,
                        assembly_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    out_shape = out_pool.get_shape().as_list()
    reshape = tf.reshape( tf.nn.bias_add(out_pool, assembly_biases),
        [out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]])
    if train:
      print('Dimensions of network Tensors: [minibatch size, ..dims.. , channels]')
      print(data.get_shape().as_list(), '->',
            conv1.get_shape().as_list(), '->', pool1.get_shape().as_list(), '->',
            conv2.get_shape().as_list(), '->', pool2.get_shape().as_list(), '->',
            hidden.get_shape().as_list(), '->', single_pool.get_shape().as_list(), '->',
            out_shape)
    return reshape

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits, train_labels_node))
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                  tf.nn.l2_loss(assembly_weights) + tf.nn.l2_loss(assembly_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)
  # Predictions for the minibatch, validation set and test set.
  train_prediction = tf.nn.sigmoid(logits)
  # We'll compute them only once in a while by calling their {eval()} method.
  validation_prediction = tf.nn.sigmoid(model(validation_data_node))
  test_prediction = tf.nn.sigmoid(model(test_data_node))
  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    if (state==1):
      tf.initialize_all_variables().run()
      saver = tf.train.Saver({'conv1_weights':conv1_weights, 'conv1_biases':conv1_biases,
                            'conv2_weights':conv2_weights, 'conv2_biases':conv2_biases,
                            'fc1_weights':fc1_weights, 'fc1_biases':fc1_biases,
                           'fc2_weights':fc2_weights_partial, 'fc2_biases':fc2_biases_partial})
      # Load pretrained parameters for single 24*24 patches.
      saver.restore(s, "net_sigmoid Feb 10.ckpt")
      s.run(fc2_weights.assign(np.concatenate([fc2_weights_partial.eval(), fc2_weights_partial.eval(),fc2_weights_partial.eval(),fc2_weights_partial.eval()],3)))
      s.run(fc2_biases.assign(np.concatenate([fc2_biases_partial.eval(), fc2_biases_partial.eval(),fc2_biases_partial.eval(),fc2_biases_partial.eval()],0)))
      a = np.ones((5,5,4,1))/10
      s.run(assembly_weights.assign(assembly_weights.eval()+a))
    if (state==2):
        saver = tf.train.Saver({'conv1_weights':conv1_weights, 'conv1_biases':conv1_biases,
                            'conv2_weights':conv2_weights, 'conv2_biases':conv2_biases,
                            'fc1_weights':fc1_weights, 'fc1_biases':fc1_biases,
                            'fc2_weights':fc2_weights, 'fc2_biases':fc2_biases,
                            'assembly_weights':assembly_weights, 'assembly_biases':assembly_biases})
        saver.restore(s, "net_assemble"+time.ctime()[3:10]+".ckpt")
        saver = tf.train.Saver({'conv1_weights':conv1_weights, 'conv1_biases':conv1_biases,
                            'conv2_weights':conv2_weights, 'conv2_biases':conv2_biases,
                            'fc1_weights':fc1_weights, 'fc1_biases':fc1_biases})
        saver.restore(s, "net_sigmoid"+time.ctime()[3:10]+".ckpt")

    print('Initialized! Training set size = %d' % train_size)
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      if step % (train_size / BATCH_SIZE) == 0:  ind =  np.random.permutation(train_size)
      batch_data = train_data[ind[offset:(offset + BATCH_SIZE)], :, :, :]
      batch_labels = train_labels[ind[offset:(offset + BATCH_SIZE)]]
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % 6 == 0:
        print('%s: Epoch %.2f' % (time.ctime(), float(step) * BATCH_SIZE / train_size))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))

        #validation_predictions = validation_prediction.eval()
        #print('Validation loss: %.2f%%' %)

        sys.stdout.flush()
        # Finally print the result!
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver({'conv1_weights':conv1_weights, 'conv1_biases':conv1_biases,
                            'conv2_weights':conv2_weights, 'conv2_biases':conv2_biases,
                            'fc1_weights':fc1_weights, 'fc1_biases':fc1_biases,
                            'fc2_weights':fc2_weights, 'fc2_biases':fc2_biases,
                            'assembly_weights':assembly_weights, 'assembly_biases':assembly_biases})
    # Save the variables to disk.
    save_path = saver.save(s, "net_assemble"+time.ctime()[3:10]+".ckpt")
    print ("Model saved in file: ", save_path)
    test_predictions = test_prediction.eval()
    print(test_predictions.shape)
    np.savetxt('tmp_images/assemble4.csv', test_predictions,  fmt='%.6f', delimiter=', ')

if __name__ == '__main__':
  tf.app.run()


"""
  # Restore variables from disk.
  saver.restore(sess, "net1.ckpt")
  print "Model restored."
"""
