"""
Description:
Author: Iva
Date: 14. 10. 2015
Python version: 3.4
"""

#Prerequisite: image downloaded from web using bytes_bite
#text files containing coordinates of single sites in the images
import pymongo
import numpy as np
import bson
from PIL import Image
#from bytes_bite import google_query
import urllib2 as urllib
from io import BytesIO


ZOOM_LEVEL = 18
IMG_SIZE = 600
SINGLE_SIZE = 24#/2
NUM_CHANNELS = 3
db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_L"]
db_single = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_single_L"]

'''
geoindexing:
db_train.create_index( [( 'coordinates' , pymongo.GEOSPHERE )] )
db.<collection>.find( { <location field> :
                         { $near :
                           { $geometry :
                              { type : "Point" ,
                                coordinates : [ <longitude> , <latitude> ] } ,
                             $maxDistance : <distance in meters>
                      } } } )
'''


from os import listdir
from os.path import isfile, join
mypath='coord_lists/click/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyind = [int(f[2:4])-1 for f in onlyfiles]   #  substract one to adjust for R vs python indexing (start from 1 vs 0)
n = len(onlyfiles)

coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False)

def db2np(lng, lat, db=False, db_train=db_train):
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
       img_array = np.asarray(img2, dtype='uint8')[20:(IMG_SIZE+20),20:(IMG_SIZE+20),:].reshape(1,(IMG_SIZE**2)*3)
     return img_array.reshape(IMG_SIZE, IMG_SIZE,3)

def insert_singles(click, big_image, coordinates, classification, db_single=db_single, trans=False):
    for j in range(click.shape[0]):
        x = IMG_SIZE-1-click[j,1]
        y = click[j,0]
        if x>=SINGLE_SIZE and y>=SINGLE_SIZE and x<=(IMG_SIZE-SINGLE_SIZE) and y<=(IMG_SIZE-SINGLE_SIZE):
            new_image = big_image[(x-SINGLE_SIZE):(x+SINGLE_SIZE),(y-SINGLE_SIZE):(y+SINGLE_SIZE), :].reshape(1,((SINGLE_SIZE*2)**2)*3)
                          #np.asarray(img2, dtype='uint8')[20:(img_size+20),20:(img_size+20),:].reshape(1,(img_size**2)*3)
            if (trans):
                image_list = transform_img(new_image)
            else:
                image_list = [new_image]
            for new_image in image_list:
                image_byte = bson.binary.Binary(new_image.tostring())
                doc = {"coordinates": coordinates,
                   "click": [round(x),round(y)],
                   "image": image_byte,
                   "class": classification}
                db_single.insert_one(doc)
                #print('.')

def transform_img(img_array, size= SINGLE_SIZE*2):
    """
    returns list of eight transformations of img_array (all Pi/4 rotations & reflection)
        :param img_array:           list of lists representing square matrix
        :param size:                numcolumns/numrows of the square array (picture)
    """
    img_array = img_array.reshape(size, size, NUM_CHANNELS)
    img_transformed = [img_array, img_array[::-1,::-1,:]]               # rotate by 180deg
    for i in range(0, 2):
        img_transformed.append(img_transformed[i].transpose(1,0,2))      # reflect by standard diagonal
    for i in range(0, 4):
        img_transformed.append(img_transformed[i][range(size)[::-1],:,:])  # reflect by first (vertical?) axes
    for i in range(0,8):
        img_transformed[i] = img_transformed[i].ravel()
    return img_transformed


db_single.drop()

for i in range(n):
    big_image = db2np(coords[onlyind[i],1], coords[onlyind[i],2])
    click = np.genfromtxt(mypath+onlyfiles[i], delimiter=',', skip_header= True, dtype='uint16')*2
    insert_singles(click, big_image, coordinates=[round(coords[onlyind[i],1],6), round(coords[onlyind[i],2],6)], classification = True, trans=True)
    print(db_single.count())

print(db_single.count())

shift=600
click=np.array([[i*27,j*27] for i in range(1,9) for j in range(1,9)  ])
for i in range(400):
     big_image = db2np(coords[shift+i,1], coords[shift+i,2], db=False, db_train=db_train)
     insert_singles(click, big_image, coordinates=[round(coords[shift+i,1],6), round(coords[shift+i,2],6)], classification = False)

print(db_single.count())

'''

i=2
img_array = db2np(db_train, coords[onlyind[i],1], coords[onlyind[i],2])
img_trans = transform_img(img_array,300)
for i in range(8):
 img = Image.fromarray(img_trans[i].reshape(300, 300, NUM_CHANNELS), 'RGB')
 img.save("tmp_images/testm"+str(i)+".png")

click = np.genfromtxt(mypath+onlyfiles[i], delimiter=',', skip_header= True, dtype='uint16')
for j in range(click.shape[0]):
   x = IMG_SIZE-1-click[j,1]
   y = click[j,0]
   if x>=SINGLE_SIZE and y>=SINGLE_SIZE and x<=(IMG_SIZE-SINGLE_SIZE) and y<=(IMG_SIZE-SINGLE_SIZE):
        new_image = big_image[(x-SINGLE_SIZE):(x+SINGLE_SIZE),(y-SINGLE_SIZE):(y+SINGLE_SIZE), :].reshape(1,((SINGLE_SIZE*2)**2)*3)
        img = Image.fromarray(new_image.reshape(24, 24, NUM_CHANNELS), 'RGB')
        img.save("tmp_images/testm"+str(j)+".png")
        print(j)


'''

def export_images():
  db_single = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset_single"]
  cursor=db_single.find() #limit(limit=limit).skip(skip=skip)
  i=0
  one_image = db_single.find_one()
  for one_image in cursor:
      img_array = np.fromstring(one_image["image"], dtype='uint8').reshape(24*2, 24*2, 3)
      img = Image.fromarray(img_array, 'RGB')
      img.save("tmp_images/"+str(one_image['class'])+"/img"+str(i)+".png")
      i+=1