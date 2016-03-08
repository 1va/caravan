"""
Description:
Author: Iva
Date: 14. 10. 2015
Python version: 3.4
"""

import numpy as np
SINGLE_SIZE = 2

def transform_img(img_array, size= SINGLE_SIZE):
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

def dihedral(size = SINGLE_SIZE):
    img = np.array(range(size**2))
    img = transform_img(img, size= size)
    for i in range(len(img)):
        img[i] =  np.array([range(size**2) == t for t in img[i]], dtype='float32')
    return img