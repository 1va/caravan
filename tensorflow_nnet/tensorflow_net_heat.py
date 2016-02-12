""" Simple, end-to-end, LeNet-5-like convolutional model.
"""
from __future__ import print_function
import urllib2 as urllib
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image


datalimit=100
dataskip=0
NET_FILE ="net1 Feb  8.ckpt"


IMAGE_SIZE = 300
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
ZOOM_LEVEL=17
NUM_LABELS = 2
SEED = 6647#8  # Set to None for random seed.

def google_query(latitude,longitude, zoom_level=ZOOM_LEVEL, img_size=IMAGE_SIZE):
    beginning = "https://maps.googleapis.com/maps/api/staticmap?"
    api_key = 'key=AIzaSyADEXfeHpcdEXfsEWF_F7iX5bKzVSnlkk8'
    longitude = round(longitude, 6)
    latitude = round(latitude, 6)
    center = "center=" + str(latitude) + "," + str(longitude)
    zoom = "zoom=%d" % zoom_level
    img_format = "format=png"
    map_type = "maptype=satellite"
    imag_size = 'size='+str(img_size+40)+'x'+ str(img_size+40)
    return "&".join([beginning, center, zoom, img_format, map_type, imag_size, api_key])

def get_one(lng, lat, db=False, db_train=None):
     """
     This functon retrieves image at given coordinates as a numpy matrix
     :param lng:  around -50
     :param lat:  around 0
     :param db_train: when getting previously downloaded images
             repricated due to rounding error in coordinates and the need of precise position
     :return: numpy matrix representing image
     """
     if db:
       one_image = db_train.find_one ({'coordinates': {'$near' :{'$geometry' : {'type':'Point', 'coordinates': [lng,lat]}, '$maxDistance' : 10 }}})
       img_array = np.fromstring(one_image["image"], dtype='uint8')
     else:
       one_point = google_query(lat, lng, zoom_level=ZOOM_LEVEL)
       try:
           file = urllib.urlopen(one_point)
           b = BytesIO(file.read())
           img = Image.open(b)
           img2 = img.convert(mode= 'RGB')   # 'RGB' or 'L' (Luma transformation to black&white)
           #img2.thumbnail((img_size+40, img_size+40), Image.ANTIALIAS) # no need?
       except IOError as e:
           print ('Bad image: %s. Coordinates: %.4f, %.4f' % (e, lat, lng))
       img_array = np.asarray(img2, dtype='uint8')[20:(IMAGE_SIZE+20),20:(IMAGE_SIZE+20),:]
     return img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

def load_dataset(limit=50, db=False):
    coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False)
    #db_trans = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_S"]
    X=np.zeros(limit*IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS).reshape(limit,IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS).astype(dtype='float32')
    for i in range(limit):
        X[i,:,:,:] = get_one(coords[i,1], coords[i,2], db=False)/np.float32(PIXEL_DEPTH)-.5
    return X


def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  data = load_dataset(limit=datalimit)
  #X = np.genfromtxt('tmp_images/assemble_x.csv', delimiter=',', skip_header= False).astype(dtype='float32')
  #data = X.reshape(X.shape[0],IMAGE_SIZE, IMAGE_SIZE,NUM_CHANNELS)/np.float32(PIXEL_DEPTH)-.5
  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.
  test_data_node = tf.constant(data)
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
  fc2_weights = tf.Variable(
      tf.truncated_normal([1, 1, depth3, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
  assembly_size = IMAGE_SIZE/2/2-5
  assembly_weights = tf.Variable(
      tf.truncated_normal([assembly_size, assembly_size, NUM_LABELS, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  assembly_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
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
    out_pool = tf.nn.bias_add(single_pool, fc2_biases)
    out_shape = out_pool.get_shape().as_list()
    reshape = tf.reshape(out_pool,
        [out_shape[0] * out_shape[1] * out_shape[2] , out_shape[3]])
    if train:
      print('Dimensions of network Tensors: [minibatch size, ..dims.. , channels]')
      print(data.get_shape().as_list(), '->',
            conv1.get_shape().as_list(), '->', pool1.get_shape().as_list(), '->',
            conv2.get_shape().as_list(), '->', pool2.get_shape().as_list(), '->',
            hidden.get_shape().as_list(), '->', single_pool.get_shape().as_list(), '->',
            out_shape)
    #return tf.nn.bias_add(reshape, assembly_biases)
    out_softmax = tf.nn.softmax(reshape)
    return tf.reshape(out_softmax,[out_shape[0],  out_shape[1] * out_shape[2], out_shape[3]])

  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    saver = tf.train.Saver({'conv1_weights':conv1_weights, 'conv1_biases':conv1_biases,
                            'conv2_weights':conv2_weights, 'conv2_biases':conv2_biases,
                            'fc1_weights':fc1_weights, 'fc1_biases':fc1_biases,
                            'fc2_weights':fc2_weights, 'fc2_biases':fc2_biases})
    # Load pretrained parameters for single 24*24 patches.
    saver.restore(s, NET_FILE)
    print('Initialized!')
    test_predictions = np.array(model(test_data_node).eval()[:,:,1])
    # Finally save the result!
    print(test_predictions.shape)
    np.savetxt('tmp_images/assemble1.csv', test_predictions,  fmt='%.6f', delimiter=', ')
    #np.savetxt('tmp_images/ids.csv', data_matrix,  fmt='%.6f', delimiter=', ')

if __name__ == '__main__':
  tf.app.run()
