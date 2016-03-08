""" Simple, end-to-end, LeNet-5-like convolutional model.
"""
from __future__ import print_function
import urllib2 as urllib
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import dihedral

BATCH_SIZE=64
NET_FILE ="net_single Feb 25.ckpt"
#NET_FILE ="net1 Feb  8.ckpt"


koef=1
ZOOM_LEVEL = 16+koef
PIXELperLABEL = 2
IMAGE_SIZE= 300*koef
SINGLE_SIZE = 24*koef
LABEL_SIZE = 1+ (IMAGE_SIZE-SINGLE_SIZE)/PIXELperLABEL  # = 1+69*koef
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 1
SEED = 6647#8  # Set to None for random seed.
state=1

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
     :param lng:  around 0
     :param lat:  around 50
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
       except IOError as e:
           print ('Bad image: %s. Coordinates: %.4f, %.4f' % (e, lat, lng))
           time.sleep(1)
           try:
             file = urllib.urlopen(one_point)
             b = BytesIO(file.read())
             img = Image.open(b)
             img2 = img.convert(mode= 'RGB')   # 'RGB' or 'L' (Luma transformation to black&white)
           except IOError as e:
               print ('Repeated bad image: %s. Coordinates: %.4f, %.4f' % (e, lat, lng))

       img_array = np.asarray(img2, dtype='uint8')[20:(IMAGE_SIZE+20),20:(IMAGE_SIZE+20),:]
     return img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

def load_dataset(limit=BATCH_SIZE, coords=None, db=False):
    if coords is None:
       coords = np.genfromtxt('tmp_images/assemble_ids.csv', delimiter=',', skip_header= False)[:,1:3]
    limit=min(limit, coords.shape[0])
    #db_trans = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_S"]
    X=np.zeros(limit*IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS).reshape(limit,IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS).astype(dtype='float32')
    for i in range(limit):
        X[i,:,:,:] = get_one(coords[i,0], coords[i,1], db=db)/np.float32(PIXEL_DEPTH)-.5
    return X

def get_centre(data):
    padding = (IMAGE_SIZE-SINGLE_SIZE)/2
    return data[:, padding:(padding+SINGLE_SIZE), padding:(padding+SINGLE_SIZE), :]

def main(csv=True, coords=None, gps=False):  # pylint: disable=unused-argument
  if coords is None:
       coords = np.array([[ -1.353887,  50.965639],
       [ -1.386731,  50.935744],
       [ -1.401907,  50.925657],
       [ -1.368191,  50.914509],
       [ -1.354101,  50.90988 ],
       [ -1.34853 ,  50.902636],
       [ -0.964885,  50.81373 ]])
  n = coords.shape[0]

  print('%s: Loading nnet...' % time.ctime())
  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.
  #test_data_node = tf.constant(data)
  test_data_node = tf.placeholder( tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  test_data_node2 = tf.placeholder( tf.float32, shape=(n%BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  test_centre_node = tf.placeholder( tf.float32, shape=(BATCH_SIZE, SINGLE_SIZE, SINGLE_SIZE, NUM_CHANNELS))
  test_centre_node2 = tf.placeholder( tf.float32, shape=(n%BATCH_SIZE, SINGLE_SIZE, SINGLE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
  depth1=SINGLE_SIZE*SINGLE_SIZE*NUM_CHANNELS
  conv1_weights = tf.constant( # shape: [SINGLE_SIZE, SINGLE_SIZE, NUM_CHANNELS, depth1]
      np.reshape(np.diag([1 for _ in range(depth1)]).astype('float32'),[SINGLE_SIZE, SINGLE_SIZE, NUM_CHANNELS, depth1]))
  #centre_weights =tf.constant( # shape: [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS, depth1]
  #    get_centre(depth1))
  #conv1_biases = tf.constant(np.zeros([depth1], dtype='float32'))
  # conv2_weights = tf.Variable(tf.concat(3,[tf.ones([1,1,depth1,1])*2*(.5-i%2) for i in range(2*depth1)]))
  conv2_weights = tf.concat(3,[tf.ones([1,1,depth1,1]),-tf.ones([1,1,depth1,1])])
  kernel_weights = tf.ones([SINGLE_SIZE, SINGLE_SIZE, NUM_CHANNELS, 1])  # gaussian or linear kernel to use instead avg_pool

  def model(data, centre_data,  train=False):
    """The Model definition."""
    # 1. extract features from centre and every (other) patch in the same way (first attempt: identity)
    conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, PIXELperLABEL, PIXELperLABEL, 1], padding='VALID')
    centre = tf.nn.conv2d(centre_data, conv1_weights, strides=[1, PIXELperLABEL, PIXELperLABEL, 1], padding='VALID')
    square_dim = conv1.get_shape().as_list()[2]
    rep_centre = tf.concat(1, [tf.concat(2,[centre for _ in range(square_dim)]) for _ in range(square_dim)])
    layer1 = tf.add(conv1, -rep_centre)
    # 2. compute absolute values of the differences (there must be a better way ...)
    conv2 = tf.nn.depthwise_conv2d(layer1, conv2_weights, strides = [1,1,1,1], padding='SAME')
    # conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, depth1], padding='VALID')
    conv2_shape = conv2.get_shape().as_list()
    layer2 = tf.reshape(conv2,[conv2_shape[0], conv2_shape[1] * conv2_shape[2] *conv2_shape[3]/2, 2, 1])
    pool1 = tf.nn.max_pool(layer2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
    pool1_shape = pool1.get_shape().as_list()
    layer3 = tf.reshape(pool1,[pool1_shape[0], pool1_shape[1]/depth1, depth1, 1])
    # 3. aggregate the absolute pixel differences (first attempt: average)
    out_pool = tf.nn.avg_pool(layer3, ksize=[1, 1, depth1, 1], strides=[1, 1, depth1, 1], padding='SAME')
    out_shape = out_pool.get_shape().as_list()
    print('Dimensions of network Tensors: [minibatch size, ..dims.. , channels]')
    print(data.get_shape().as_list(), '->',
            conv1.get_shape().as_list(), '->', layer1.get_shape().as_list(), '->',
            conv2.get_shape().as_list(), '->', layer2.get_shape().as_list(), '->',
            pool1.get_shape().as_list(), '->', layer3.get_shape().as_list(), '->',
            out_shape)
    return tf.reshape(out_pool, [out_shape[0], out_shape[1]])

  # Create a local session to run this computation.
  with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    num_batches = np.floor(n/BATCH_SIZE).astype('int16')
    print('%s: Initialized! (Loading data in %d minibatches))' % (time.ctime(), num_batches+1))
    if gps:
        from get_data.map_coverage import MercatorProjection, G_Point, G_LatLng
        from get_data.labels_GPS import labels_GPS, labels_suspect, labels_GPS_list
        suspect = np.zeros((0,2))
    else:
        suspect = np.zeros((0,LABEL_SIZE**2))

    for step in range(num_batches):
      offset = (step * BATCH_SIZE)
      batch_data = load_dataset(coords=coords[offset:(offset + BATCH_SIZE), :])
      feed_dict = {test_data_node: batch_data,  test_centre_node: get_centre(batch_data)}
      # Run the graph and fetch some of the nodes.
      test_predictions = s.run(model(test_data_node, test_centre_node), feed_dict=feed_dict)
      print('.'),
      if gps:
          suspect = np.concatenate([suspect, labels_GPS_list(labels= np.array(test_predictions[:,:,1]), coords=coords[offset:(offset + BATCH_SIZE), :], pixels= PIXELperLABEL, zoom=ZOOM_LEVEL)], axis=0 )
      else:
          suspect = np.concatenate([suspect, np.array(test_predictions[:,:,1])], axis=0)
      if (step+1)%10==0:
          print('%s: Processing batch %d out of %d' %(time.ctime(), step, num_batches+1))
          np.savetxt('tmp_images/suspect_tmp'+time.ctime()[3:10]+'.csv', suspect,  fmt='%.6f', delimiter=', ')

    if n%BATCH_SIZE > 0 :
      offset = num_batches * BATCH_SIZE
      batch_data = load_dataset(coords=coords[offset:n, :])
      feed_dict = {test_data_node2: batch_data, test_centre_node2: get_centre(batch_data)}
      # Run the graph and fetch some of the nodes.
      test_predictions = s.run(model(test_data_node2, test_centre_node2), feed_dict=feed_dict)
      if gps:
          suspect = np.concatenate([suspect, labels_GPS_list(labels= np.array(test_predictions[:,:,1]), coords=coords[offset:n, :], pixels= PIXELperLABEL, zoom=ZOOM_LEVEL) ], axis=0 )
      else:
          suspect = np.concatenate([suspect, np.array(test_predictions)], axis=0)

    # Finally save the result!
    print('%s: Result with %d rows and  %d columns. ' % (time.ctime(), suspect.shape[0], suspect.shape[1]))
    if csv:
      np.savetxt('tmp_images/centre_heat.csv', suspect,  fmt='%.6f', delimiter=', ')
    else: return(suspect)
    #np.savetxt('tmp_images/ids.csv', data_matrix,  fmt='%.6f', delimiter=', ')
    s.close()

if __name__ == '__main__':
  tf.app.run()


#if __name__ == '__main__':
#    kwargs = {}
#    if len(sys.argv) > 1:
#        kwargs['num_epochs'] = int(sys.argv[1])
#    net = main(**kwargs)
#    pickle_net(net)