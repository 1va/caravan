"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.
"""
from __future__ import print_function
import os
import sys
import time
#import urllib
#import tensorflow.python.platform
import numpy as np
import pymongo
from sklearn import cross_validation
import tensorflow as tf

datalimit= 3500
dataskip= 0
VALIDATION_SIZE = min(300,datalimit/4)  # Size of the validation set.
BATCH_SIZE = 64
NUM_EPOCHS = 80
IMAGE_SIZE = 300
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = 66478  # Set to None for random seed.


def db2np(db_trans, limit=None, skip=0):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
    Extract the labels into a 1-hot matrix [image index, label index].
    """
    if (limit==None):
        limit=db_trans.count()
    cursor=db_trans.find().limit(limit=limit).skip(skip=skip)
    X=np.zeros(limit*IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS).reshape(limit,IMAGE_SIZE, IMAGE_SIZE,NUM_CHANNELS).astype(dtype='float32')

    y=np.zeros(limit).astype(dtype='uint8')
    i=0
    for one_image in cursor:
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        X[i,:,:,:] = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)/np.float32(255)-.5
        y[i]=int(one_image['class'])
        i += 1
    return X, (np.arange(NUM_LABELS) == y[:, None]).astype(np.float32)


def load_dataset(limit=None, skip=0):
    db_trans = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]
    X, y = db2np(db_trans,limit=limit, skip=skip)
    sss = cross_validation.StratifiedShuffleSplit(y[:,1], n_iter=1, test_size=VALIDATION_SIZE, random_state=SEED)
    for train_index, test_index in sss:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
    return X_train, y_train, X_val, y_val, X_val, y_val


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100*(1- np.float32(np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0])
def accu_table(predictions,labels,toplabel=1):
    """Return list contingency table (as a list) of predictions and 1-hot labels."""
    predP = np.argmax(predictions,1) == toplabel
    trueP = np.argmax(labels,1) ==toplabel
    return np.sum(predP*trueP),np.sum(predP*(1-trueP)), np.sum((1-predP)*trueP),np.sum((1-predP)*(1-trueP))
def f1_score(predictions, labels, toplabel=1):
    """Return the F1 score based on dense predictions and 1-hot labels."""
    TP, FP, FN, TN  = accu_table(predictions, labels, toplabel)
    precision = TP/(TP+FP+.01)
    recall = TP/(TP+FN+.01)
    return 2*precision*recall/(precision+recall+.01)*100


def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_dataset(limit=datalimit,skip=dataskip)
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
  hidden2_size = ((IMAGE_SIZE-4)/2-4)/2
  depth3=32
  conv3_weights = tf.Variable(
      tf.truncated_normal([55, 55, depth2, depth3],
                          stddev=0.1,
                          seed=SEED))
  conv3_biases = tf.Variable(tf.constant(0.1, shape=[depth3]))
  hidden3_size = (hidden2_size-54)/2
  fc_weights = tf.Variable(
      tf.truncated_normal([hidden3_size**2 * depth3, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    #print(conv.get_shape().as_list())
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    #print(pool.get_shape().as_list())
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    #print(conv.get_shape().as_list())
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    #print(pool.get_shape().as_list())
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      pool = tf.nn.dropout(pool, 0.5)
    conv = tf.nn.conv2d(pool,
                        conv3_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    #print(conv.get_shape().as_list())
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    #print(pool.get_shape().as_list())
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    if train:
      reshape = tf.nn.dropout(reshape, 0.5)
    return (tf.matmul(reshape, fc_weights) + fc_biases)

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits, train_labels_node))
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_biases)+
                  tf.nn.l2_loss(fc_weights) + tf.nn.l2_loss(fc_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.005,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)
  # Predictions for the minibatch, validation set and test set.
  train_prediction = tf.nn.softmax(logits)
  # We'll compute them only once in a while by calling their {eval()} method.
  validation_prediction = tf.nn.softmax(model(validation_data_node))
  test_prediction = tf.nn.softmax(model(test_data_node))
  # Add ops to restore 1st two convolution layers.
  saver = tf.train.Saver([conv1_biases, conv1_weights, conv2_biases, conv2_weights])
  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    saver.restore(s, "mnistConvLayers32&64.ckpt")
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      #print(fc_biases.eval(s))
      print(': %.3f :' % l ),
      if step % 20 == 0:
        print('%s: Epoch %.2f' % (time.ctime(), float(step) * BATCH_SIZE / train_size))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%% and F1-score: %.1f%%' %
                 (error_rate(predictions, batch_labels), f1_score(predictions, batch_labels)))
        validation_predictions = validation_prediction.eval()
        print('Validation error: %.1f%% and F1-score: %.1f%%' %
              (error_rate(validation_predictions, validation_labels),
              f1_score(validation_predictions, validation_labels)))
        print(accu_table(validation_predictions, validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_predictions = test_prediction.eval()
    print('Test error: %.1f%% and F1-score: %.1f%%' %
          (error_rate(test_predictions, test_labels),
           f1_score(test_predictions, test_labels)))
    print(accu_table(test_predictions, test_labels))
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Save the variables to disk.
    save_path = saver.save(s, "net2"+time.ctime()[3:10]+".ckpt")
    print ("Model saved in file: ", save_path)
if __name__ == '__main__':
  tf.app.run()


"""
  # Restore variables from disk.
  saver = tf.train.Saver()
  with tf.Session() as s:
    tf.initialize_all_variables().run()
    print(conv1_biases.eval(s))
    saver.restore(s, "tensorflow_nnet/net2 Nov 22.ckpt")
    print(conv1_biases.eval(s))

  print "Model restored."
"""
