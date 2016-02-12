"""
Description:    Testing Image conversions and bytes format for mongoDB in python 2.7
Author:         Iva
Date:           19/10/2015
Python version: 2.7.10 (venv2)
"""

import numpy as np
import pymongo
import bson
import urllib2 as urllib
from PIL import Image
from io import BytesIO
import random
import time

IMG_SIZE = 600
ZOOM_LEVEL = 18

def google_query(latitude,longitude, zoom_level=ZOOM_LEVEL, img_size=IMG_SIZE):
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

# define constant difference between grid points if using zoom_level=19
DEFAULT_LAT_DIFFERENCE_AT_19 = 0.0008#0.0010
DEFAULT_LONG_DIFFERENCE_AT_19 = 0.0010#0.0015

###########################################################
def coord_box(coordinates, lat_diff=DEFAULT_LAT_DIFFERENCE_AT_19, long_diff=DEFAULT_LONG_DIFFERENCE_AT_19, size=1):
    """
    Substitute each coordinate with (2*size+1)^2 new coordinates in a square grid around the original.
    :param lat_diff:    latitude step in the grid
    :param long_diff:   longitude spet in the grid
    :param size:        how big neighborhood to take
    """
    newcoord = []
    for coord in coordinates:
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                newcoord.append([coord[0]+i*lat_diff, coord[1]+j*long_diff])
    return newcoord

def download_list(lst, db, classification, zoom_level=ZOOM_LEVEL, img_size=IMG_SIZE):
    count = 0
    for latitude, longitude in lst:
        one_point = google_query(latitude, longitude, zoom_level=zoom_level)
        try:
           file = urllib.urlopen(one_point)
           b = BytesIO(file.read())
           img = Image.open(b)
           img2 = img.convert(mode= 'RGB')   # 'RGB' or 'L' (Luma transformation to black&white)
           #img2.thumbnail((img_size+40, img_size+40), Image.ANTIALIAS) # no need?
        except IOError as e:
           print ('Bad image: %s. Coordinates: %.4f, %.4f' % (e, latitude, longitude))

        image_array = np.asarray(img2, dtype='uint8')[20:(img_size+20),20:(img_size+20),:].reshape(1,(img_size**2)*3)
        image_byte = bson.binary.Binary(image_array.tostring())

        doc = {"coordinates": [longitude, latitude],
               "image": image_byte,
               "class": classification}

        db.insert_one(doc)

        if (count%100 ==0):
            print('.'),
        count += 1
    return db.count()

if False:
  db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]
#caravans = np.genfromtxt('coord_lists/GPS_5000caravans.csv', delimiter=',', skip_header= False, dtype='float')
  caravans = coord_box(np.genfromtxt('coord_lists/GPS_osm_596caravans.csv', delimiter=',', skip_header= False, dtype='float'))
  controls = np.genfromtxt('coord_lists/GPS_5000controls.csv', delimiter=',', skip_header= False, dtype='float')
else:
  db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_L"]
  caravans = np.genfromtxt('coord_lists/GPS_osm_596caravans.csv', delimiter=',', skip_header= False, dtype='float')
  controls = np.genfromtxt('coord_lists/GPS_5000controls.csv', delimiter=',', skip_header= False, dtype='float')[1:600,]

if True:
    db_train.drop()
    print(time.ctime())
    download_list(caravans, db_train, classification=True, zoom_level=ZOOM_LEVEL)
    print(time.ctime())
    download_list(controls, db_train, classification=False, zoom_level=ZOOM_LEVEL)
    print(time.ctime())


if False:
    one_point = google_query(50.7677717,-0.8529036)
    file = urllib.urlopen(one_point)

    b = BytesIO(file.read())
    img = Image.open(b)

    img2 = img.convert('RGB')  # 'RGB' or 'L' (Luma transformation to black&white)
    #img2.thumbnail((IMG_SIZE+40, IMG_SIZE+40), Image.ANTIALIAS) # no need?

    image_array = np.asarray(img, dtype='uint8')[20:(IMG_SIZE+20),20:(IMG_SIZE+20)].reshape(1,IMG_SIZE**2)
    image_byte = bson.binary.Binary(image_array.tostring())

    doc = {"_id":'pokus', "image": image_byte}
    db.insert_one(doc)
    db.remove({"_id":'pokus'})

    one_image = db.find_one()
    image_array = np.fromstring(one_image["image"], dtype='uint8').reshape(IMG_SIZE, IMG_SIZE)
    img = Image.fromarray(image_array.reshape(IMG_SIZE, IMG_SIZE), 'L')
    img.save("images/graymap2.png")
