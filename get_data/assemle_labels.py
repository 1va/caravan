"""
Description:
Author: Iva
Date: 14. 10. 2015
Python version: 3.4
"""

IMG_SIZE = 300
KERNEL_SIZE = 10
LABEL_SIZE = 70

import numpy as np
from scipy import signal
from os import listdir
from os.path import isfile, join
coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False)
mypath='coord_lists/click/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyind = [int(f[2:4])-1 for f in onlyfiles]   #  substract one to adjust for R vs python indexing (start from 1 vs 0)
n = len(onlyfiles)

def heat_labels_gauss(click, img_size=IMG_SIZE, k_size=KERNEL_SIZE, label_size=LABEL_SIZE):
    # take list of pixel coordinates and return 70x70 heatmap
    img = np.zeros((img_size, img_size))
    for j in range(click.shape[0]):
        x = img_size-1-click[j,1]
        y = click[j,0]
        img[x,y]=1
    kernel = np.outer(signal.gaussian(img_size+1, k_size), signal.gaussian(img_size+1, k_size))
    img = signal.convolve2d(img,kernel, mode='same')
    offset = (img_size-img_size/label_size*(label_size-1))/2
    step = img_size/label_size
    return img[offset:(img_size-offset+step):step, offset:(img_size-offset+step):step]

def heat_labels(click, mode='linear', img_size=IMG_SIZE, k_size=KERNEL_SIZE, label_size=LABEL_SIZE):
    # take list of pixel coordinates and return 70x70 heatmap
    img = np.zeros((img_size+2*k_size, img_size+2*k_size))
    if (mode=='gauss'):
        kernel = signal.gaussian(2*k_size-1, k_size/2)
    else:
        kernel = np.array(range(1,k_size+1)+range(k_size-1,0,-1),dtype='float32')/k_size
    kernel = np.outer(kernel,kernel)
    r = np.array(range(-k_size, k_size-1))+k_size
    for j in range(click.shape[0]):
        x = img_size-1-click[j,1] +r
        y = click[j,0] +r
        img[np.ix_(x,y)] = np.maximum(img[np.ix_(x,y)],kernel)
    offset = (img_size-img_size/label_size*(label_size-1))/2
    step = img_size/label_size
    r = np.array(range(offset,(img_size-offset+step),step))+k_size
    t = img[np.ix_(r,r)]*3
    t[t>.98] = .98+(t[t>.98]-.98)/100
    return(t)

labels =  np.zeros((n, LABEL_SIZE**2))
for i in range(n):
    click = np.genfromtxt(mypath+onlyfiles[i], delimiter=',', skip_header= True, dtype='uint16')
    labels[i,] = heat_labels(click).ravel()
y = np.concatenate([labels, np.zeros((5, LABEL_SIZE**2)) ], axis=0)
np.savetxt('tmp_images/assemble_y.csv', y,  fmt='%.6f', delimiter=', ')

def get_images(ind, coords):
    #needs: google_query from butes_bite.py, db2np from single_sites.py
    images = np.zeros((len(ind),3*IMG_SIZE**2))
    for i in range(len(ind)):
         images[i,:] = db2np(coords[ind[i],1], coords[ind[i],2]).ravel()
         print(ind[i])
    return images

ids = np.concatenate([onlyind, range(601, 606)])
x = get_images(np.concatenate(ids, coords)
z = np.concatenate([np.transpose([ids]),coords[ids,1:3]], axis=1 )

np.savetxt('tmp_images/assemble_x.csv', x,  fmt='%.6f', delimiter=', ')
np.savetxt('tmp_images/assemble_y.csv', y,  fmt='%.6f', delimiter=', ')
np.savetxt('tmp_images/assemble_ids.csv', z,  fmt='%.6f', delimiter=', ')