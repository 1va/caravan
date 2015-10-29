"""
Description:
Author: Iva
Date: 14. 10. 2015
Python version: 3.4
"""

#Prerequisite: image downloaded from web using bytes_bite
import pymongo
import numpy as np
import cv2
import bson
from PIL import Image


db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]

def transform_img(img_array, size=600):
    """
    returns list of eight transformations of img_array (all Pi/4 rotations & reflection)
        :param img_array:           list of lists representing square matrix
        :param size:                numcolumns/numrows of the square array (picture)
    """
    img_transformed = [img_array.reshape(size, size), img_array[::-1].reshape(size, size)]
    for i in range(0, 2):
        img_transformed.append(img_transformed[i].transpose())
    for i in range(0, 4):
        img_transformed.append(img_transformed[i][::-1])
    for i in range(0,8):
        img_transformed[i] = img_transformed[i].ravel()
    return img_transformed

def transform_train(mongo_collection):
    """
    retrieves all pictures from specified collection and inserts 7 new transformation of each of them
        :param mongo_collection:    mongo db collection to retrieve images from
    """
    print('Original collection size is %d images (%d positive).' % (mongo_collection.count(), mongo_collection.find( { 'class': True } ).count()))
    cursor = list(mongo_collection.find())
    for one_image in cursor:
        # convert from binary to numpy array and transform
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        img_transformed = transform_img(img_array)

        # tell user what has been downloaded
        print('Transforming image at %2.4f_%2.4f' % tuple(one_image["coordinates"]))

        # insert new transformed image into db_collection
        for img_trans in img_transformed[1:]:
            one_image['image'] = img_trans.reshape((1, img_trans.size)).tobytes()
            one_image.pop('_id')
            mongo_collection.insert_one(one_image)
            # reconnect and return error code    !!!add the code!!!!

    print('Current collection size is %d images (%d positive).' % (mongo_collection.count(), mongo_collection.find( { 'class': True } ).count()))
    return None

def build_dataset(mongo_collection, patch_size=600, orig_size=600, nb_classes=2, edgedetect=True, transform=True):
    from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
    patch_size = min(patch_size, orig_size)
    trim = round((orig_size-patch_size)/2)
    #ds = SupervisedDataSet(patch_size**2, 1)
    ds = ClassificationDataSet(patch_size**2, target=1, nb_classes=nb_classes)
    cursor = list(mongo_collection.find())
    for one_image in cursor:
        # convert from binary to numpy array and transform
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        if edgedetect:
            img_array = cv2.Canny(img_array,150,200)
        img_crop = img_array.reshape(orig_size, orig_size)[trim:(trim+patch_size),trim:(trim+patch_size)]
        classification = float(one_image["class"])
        if transform:
            transformed=transform_img(img_crop.ravel(), patch_size)
        else:
            transformed=[img_crop.ravel()]
        for one_img in transformed:
            ds.addSample(one_img.ravel(),classification)
    print('New dataset contains %d images (%d positive).' % (len(ds), sum(ds['target'])))
    return ds

#ds = build_dataset(db_train, transform=False)

db_trans = pymongo.MongoClient("192.168.0.99:30000")["google"]["transformedset"]

def transform_db(db_train, db_trans, edgedetect=True, transform=True):
    cursor = list(db_train.find())
    for one_image in cursor:
        img_array = np.fromstring(one_image["image"], dtype='uint8')
        if edgedetect:
            img_array = cv2.Canny(img_array,150,200)
        if transform:
            transformed=transform_img(img_array.ravel())
        else:
            transformed=[img_array.ravel()]
        for one_img in transformed:
            image_byte = bson.binary.Binary(one_img.tostring())
            doc = {"image": image_byte, 'class': one_image["class"]}
            db_trans.insert_one(doc)
    return [db_train.count(),db_trans.count()]

#transform_db(db_train, db_trans)
#ds = build_dataset(db_trans, edgedetect=False, transform=False)

if False:  # image prepocessing experimentation
    #blur = cv2.GaussianBlur(image_array,(5,5),0)
    edges = cv2.Canny(image_array,50,220)
    img = Image.fromarray(edges.reshape(600, 600), 'L')
    img.save("images/edgemap.png")
    edges = cv2.Canny(image_array,150,200)
    img = Image.fromarray(edges[1:80,1:80], 'L')
    img.save("images/edge2map.png")

    thresh = cv2.adaptiveThreshold(image_array,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(image_array,150,255,0)
    ret,thresh = cv2.threshold(image_array,127,255,cv2.THRESH_TOZERO)
    img = Image.fromarray(thresh.reshape(600, 600), 'L')
    img.save("images/threshmap.png")

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    temp=255*np.ones((600,600),dtype='uint8')
    cv2.drawContours(temp, contours, -1, (0,255,0), 1)
    img = Image.fromarray(temp.reshape(600, 600), 'L')
    img.save("images/tempmap.png")

