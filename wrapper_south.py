'''
This is a first wrapper of my neural network algo.

It takes list of suspected gps coordinates and classify surrounding based on google satellite images.
It returns a list of gps of identified varavan sites, haha

'''
from __future__ import print_function, division
import urllib2 as urllib
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image


koef=1
ZOOM_LEVEL = 16+koef
PIXELperLABEL = 4
IMAGE_SIZE= 300*koef*2
SINGLE_SIZE = 24*koef
LABEL_SIZE = 1+ (IMAGE_SIZE-SINGLE_SIZE)/PIXELperLABEL  # = 1+69*koef
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = 6647#8  # Set to None for random seed.
state=1
BATCH_SIZE=64

from get_data.map_coverage import MercatorProjection, G_Point, G_LatLng
from get_data.labels_GPS import snap2grid, snap_list, labels_GPS, local_max, labels_suspect
from get_data.labels_GPS import labels_GPS_list2   # unknown error!
from tensorflow_nnet.tensorflow_net_heat import google_query, get_one
from tensorflow_nnet.tensorflow_net_heat import main as tensorflow_net_heat_batch

def test_run():
    olga=np.genfromtxt('coord_lists/Olgas_list_gps.csv', delimiter=',', skip_header= True, dtype='float32')
    southwesto = olga[np.logical_and(olga[:,1]<51.3, olga[:,0]<-5), :]
    southwest = snap_list(southwesto, zoom=ZOOM_LEVEL, img_size=IMAGE_SIZE)
    print('Original num of sites: %d. Num of pictures to download: %d' %(southwesto.shape[0], southwest.shape[0]))
    n = southwest.shape[0]-240
    suspect =  tensorflow_net_heat_batch(csv=False, coords=coords, gps=True)
    return suspect

suspect =  test_run()
np.savetxt('tmp_images/suspect_southwest.csv', suspect,  fmt='%.6f', delimiter=', ')
#np.savetxt('tmp_images/olga_southwest.csv', southwest,  fmt='%.6f', delimiter=', ')

coords=np.array([snap2grid([-5.329931, 50.115492], zoom=17, img_size=600), snap2grid([-5.173187,  50.002691], zoom=17, img_size=600),snap2grid([ -5.17279291,  50.00429916], zoom=17, img_size=600)], dtype='float32')

np.savetxt('tmp_images/coords_2.csv', coords,  fmt='%.6f', delimiter=', ')
np.savetxt('tmp_images/labels_2.csv', labels,  fmt='%.6f', delimiter=', ')

suspect=np.genfromtxt('tmp_images/suspect_southwest2.csv', delimiter=',', skip_header= True, dtype='float32')
suspect1=np.genfromtxt('tmp_images/suspect_southwest1.csv', delimiter=',', skip_header= True, dtype='float32')
suspect =  np.concatenate([suspect1, suspect], axis=0)
np.savetxt('tmp_images/suspect_southwest.csv', suspect,  fmt='%.6f', delimiter=', ')
