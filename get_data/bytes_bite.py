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

IMG_SIZE = 300
db = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]

def google_query(latitude,longitude, zoom_level=18, img_size=IMG_SIZE):
    beginning = "https://maps.googleapis.com/maps/api/staticmap?"
    api_key = 'key=AIzaSyADEXfeHpcdEXfsEWF_F7iX5bKzVSnlkk8'
    longitude = round(longitude, 4)
    latitude = round(latitude, 4)
    center = "center=" + str(latitude) + "," + str(longitude)
    zoom = "zoom=%d" % zoom_level
    img_format = "format=png"
    map_type = "maptype=satellite"
    imag_size = 'size='+str(img_size+40)+'x'+ str(img_size+40)
    return "&".join([beginning, center, zoom, img_format, map_type, imag_size, api_key])

# define constant difference between grid points if using zoom_level=19
DEFAULT_LAT_DIFFERENCE_AT_19 = 0.0008#0.0010
DEFAULT_LONG_DIFFERENCE_AT_19 = 0.00010#0.0015

###########################################################
def coord_box(coordinates, lat_diff=DEFAULT_LAT_DIFFERENCE_AT_19, long_diff=DEFAULT_LONG_DIFFERENCE_AT_19, size=1):
    """
    Substitute each coordinate with (2*size+1)^2 new coordinates in a square grid around the original.
    :param lat_diff:    latitude step in the grid
    :param long_diff:   longitude spet in the grid
    :param size:        how big neighborhood to take

    :type lat_diff:     float
    :type long_diff:    float
    :type size:         int
    """
    newcoord = []
    for coord in coordinates:
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                newcoord.append([coord[0]+i*lat_diff, coord[1]+j*long_diff])
    return newcoord

caravan_parks = [[50.6259796,-2.2706743], #durdle door
                 [50.689801,-2.3341522],  #warmwell
                 [50.7523135,-2.0617302], #huntnick
                 [50.7041072,-1.1035238], #nodes point
                 [50.700116,-1.1138009], #st. helens
                 [50.7963016,-0.9838095], #hayling island
                 [50.7988633,-0.9804728], #oven campsite
                 [50.7826322,-0.9472032], #eliots estate
                 [50.7831533,-0.9574026], #fishery creek
                 [50.9058588,-1.1627122], #rookesbury
                 [51.0093301,-1.5739032], #hillfarm
                 [50.9622607,-1.6225851], #greenhill
                 [50.8515685,-1.2839778], # dybles SUSPICIOS
                 [50.7358116,-1.5499394], # hurst view
                 [50.8218972,-0.3123287] # beach park
                ]

urban_areas = [[50.9555502,-1.6420727],
               [50.9171478,-1.4334934],
               [50.9059521,-1.4211532],
               [50.8137387,-1.0789],
               [50.822926,-1.0513507],
               [50.8471096,-1.2983196]
              ]

coast_areas = [[50.8962143,-1.3970956],
               [50.832687,-1.369941],
               [50.7848938,-1.3537767],
               [50.7507503,-1.5300394],
               [50.8192504,-0.3309942],
               [50.825415,-0.294987]
              ]

random.seed(1234)
random_areas = [[random.uniform(50.8484,52.0527), random.uniform(-2.75874, 0.4485376)]for i in range(20)]

def download_list(lst, db, classification,zoom_level=18, img_size=IMG_SIZE):
    for latitude, longitude in lst:
        one_point = google_query(latitude, longitude, zoom_level=zoom_level)
        file = urllib.urlopen(one_point)
        b = BytesIO(file.read())
        img = Image.open(b)

        img2 = img.convert('L') #why? -> unpack
        img2.thumbnail((img_size+40, img_size+40), Image.ANTIALIAS) # no need?

        image_array = np.asarray(img2, dtype='uint8')[20:(img_size+20),20:(img_size+20)].reshape(1,img_size**2)
        image_byte = bson.binary.Binary(image_array.tostring())

        doc = {"coordinates": [longitude, latitude],
               "image": image_byte,
               "class": classification}
        db.insert_one(doc)
    return db.count()

db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]
if False:
    db_train.remove()
    download_list(coord_box(caravan_parks), db_train, classification=True, zoom_level=19)
    controls = coord_box(urban_areas + coast_areas + random_areas)
    download_list(controls, db_train, classification=False, zoom_level=19)

if False:
    one_point = google_query(50.7677717,-0.8529036)
    file = urllib.urlopen(one_point)

    b = BytesIO(file.read())
    img = Image.open(b)

    img2 = img.convert('L') #why? -> unpack
    img2.thumbnail((IMG_SIZE+40, IMG_SIZE+40), Image.ANTIALIAS) # no need?

    image_array = np.asarray(img2, dtype='uint8')[20:(IMG_SIZE+20),20:(IMG_SIZE+20)].reshape(1,IMG_SIZE**2)
    image_byte = bson.binary.Binary(image_array.tostring())

    doc = {"_id":'pokus', "image": image_byte}
    db.insert_one(doc)
    db.remove({"_id":'pokus'})

    one_image = db.find_one()
    image_array = np.fromstring(one_image["image"], dtype='uint8').reshape(IMG_SIZE, IMG_SIZE)
    img = Image.fromarray(image_array.reshape(IMG_SIZE, IMG_SIZE), 'L')
    img.save("images/graymap2.png")
