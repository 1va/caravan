"""
Description:
Author: Iva
Date: Nov 2015
Python version: 2.7
"""

import lasagne   #lightweigh nnet in theano
#from sklearn import cross_validation

IMG_SIZE=24
NUM_CHANNELS=3

def db2np(db_trans, limit=None, skip=0):
    import numpy as np
    if (limit==None):
        limit=db_trans.count()
    cursor=db_trans.find().limit(limit=limit).skip(skip=skip)
    X=np.zeros(limit*NUM_CHANNELS*IMG_SIZE*IMG_SIZE).reshape(limit,NUM_CHANNELS,IMG_SIZE, IMG_SIZE).astype(dtype='float32')
    y=np.zeros(limit).astype(dtype='uint8')
    i=0
    for one_image in cursor:
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        X[i,:,:,:] = img_array.reshape(1,NUM_CHANNELS,IMG_SIZE, IMG_SIZE)/np.float32(256)
        y[i]=int(one_image['class'])
        i += 1
    return X, y

def load_dataset(limit=None, skip=0):
    #from get_data.build_trainingset import db2np
    import pymongo
    from sklearn import cross_validation
    db_trans = pymongo.MongoClient("192.168.0.99:30000")["google"]['trainingset_single']   #["transformedset"]
    X, y = db2np(db_trans,limit=limit, skip=skip)
    sss = cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=.1, random_state=3476)
    for train_index, test_index in sss:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
    return X_train, y_train, X_val, y_val, X, y

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, IMG_SIZE, IMG_SIZE),
                                        input_var=input_var)
    print('Input layer shape:'+str(network.output_shape))
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    print('Conv2DLayer shape:'+str(network.output_shape))
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print('MaxPool2DLayer shape:'+str(network.output_shape))
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    print('Conv2DLayer shape:'+str(network.output_shape))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print('MaxPool2DLayer shape:'+str(network.output_shape))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    print('DenseLayer shape:'+str(network.output_shape))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    print('Output layer shape:'+str(network.output_shape))

    return network


build_net = build_cnn