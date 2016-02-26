"""
Description:    Get GPS coordinates from labels (classification output)
Author:         Iva
Date:           11/2/2016
Python version: 2.7.10 (venv2)
"""
from __future__ import division
import math
import numpy as np
from get_data.map_coverage import MercatorProjection, G_Point, G_LatLng

koef=1
MERCATOR_RANGE = 256
ZOOM_LEVEL = 16+koef
PIXELperLABEL = 4
IMAGE_SIZE= 300*koef*2
SINGLE_SIZE = 24*koef
LABEL_SIZE = 1+ (IMAGE_SIZE-SINGLE_SIZE)/PIXELperLABEL  # = 1+69*koef


def labels_suspect(labels, pixels = PIXELperLABEL, treshold=.9):
    n = np.sqrt(np.prod(labels.shape)).astype('int')
    ind = local_max(labels, n) > treshold
    grid = np.round((np.array(range(n-2))-(n-3)/2.) * pixels)
    xc = np.array([grid for i in range(n-2)])
    yc = np.transpose(xc)
    return xc[ind].ravel(), yc[ind].ravel()


def local_max(labels, n):
    x = labels.reshape(n,n)
    z = (x[2:n,:] + x[1:(n-1),:] + x[0:(n-2),:])/3
    z = (z[:,2:n] + z[:,1:(n-1)] + z[:,0:(n-2)])/3
    x[1:(n-1),1:(n-1)] += np.random.randn(n-2,n-2)*z/1000000
    y = np.amax([(x[2:n,:]), (x[1:(n-1),:]), (x[0:(n-2),:])], axis=0)
    y = np.amax([(y[:,2:n]), (y[:,1:(n-1)]), (y[:,0:(n-2)])], axis=0)
    res = (y==x[1:(n-1),1:(n-1)]).astype('int')*z  # (z+x[1:(n-1),1:(n-1)])/2
    return res


def labels_GPS(labels, center_gps, pixels = PIXELperLABEL, zoom = ZOOM_LEVEL):
    scale = 2.**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(G_LatLng(center_gps[1], center_gps[0]))
    suspect_x, suspect_y = labels_suspect(labels, pixels)
    np_gps = np.zeros((len(suspect_x),2))
    for i in range(len(suspect_x)):
        suspectPX = G_Point(centerPx.x+suspect_x[i]/scale, centerPx.y+suspect_y[i]/scale)
        suspect_gps = proj.fromPointToLatLng(suspectPX)
        np_gps[i,] = [suspect_gps.lng, suspect_gps.lat]
    return np_gps


def labels2(labels, coords, pixels = PIXELperLABEL, zoom = ZOOM_LEVEL):
    suspect = np.zeros((0,2))
    for j in range(coords.shape[0]):
        new_gps = labels_GPS(labels=labels[j,].reshape(LABEL_SIZE,LABEL_SIZE), center_gps= coords[j,:].ravel(), pixels = pixels, zoom = zoom)
        if new_gps.shape[0]>0:
              suspect = np.concatenate([suspect, new_gps], axis=0)
    return suspect


def testrun(i=3):
    labels = np.genfromtxt('tmp_images/assemble_y.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    est1 = np.genfromtxt('tmp_images/assemble1.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    est4 = np.genfromtxt('tmp_images/assemble4.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    ids = np.genfromtxt('tmp_images/assemble_ids.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False)
    mypath='coord_lists/click/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyind = [int(f[2:4])-1 for f in onlyfiles]
    num_sites =  np.zeros((61, 4))
    for i in range(len(onlyfiles)):
         click = np.genfromtxt(mypath+onlyfiles[i], delimiter=',', skip_header= True, dtype='uint16')
         num_sites[i,0] = click.shape[0]

    tr=np.zeros((10,4))
    tr[:,0] = 1 - (.02*np.array(range(10)))
    for k in range(10):
      t=tr[k,0]
      for i in range(61):
        x1,x2 = labels_suspect(labels[i,], treshold=t)
        num_sites[i,1] = x1.shape[0]
        x1,x2 = labels_suspect(est1[i,], treshold=t)
        num_sites[i,2] = x1.shape[0]
        x1,x2 = labels_suspect(est4[i,], treshold=t)
        num_sites[i,3] = x1.shape[0]
      for j in range(1,4):
        tr[k,j] = np.sqrt(np.sum((num_sites[:,j] - num_sites[:,0])**2)/61)
    print(tr)


    suspect_gps = labels_GPS(labels=labels[i,:], center_gps= ids[i,1:3].ravel())  # gps array = [longitude is greenwitch, latitude is around 50]
    return(suspect_gps)


def snap2grid(obj_gps, zoom = ZOOM_LEVEL, img_size = IMAGE_SIZE, overlap=.1):
    img_size = img_size - img_size*overlap
    scale = 2.**zoom
    proj = MercatorProjection()
    objPx = proj.fromLatLngToPoint(G_LatLng(obj_gps[1], obj_gps[0]))
    def myround(num):
        return np.round(num*scale/img_size)*img_size/scale
    centerPx = G_Point( myround(objPx.x), myround(objPx.y))
    center_gps = proj.fromPointToLatLng(centerPx)
    return [np.round(center_gps.lng,6), np.round(center_gps.lat,6)]


def unique_rows(a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def snap_list(inlist, zoom=ZOOM_LEVEL, img_size = IMAGE_SIZE,  overlap=.1):
    outlist=inlist
    for i in range(len(inlist)):
        outlist[i,]=snap2grid(inlist[i,], zoom=zoom, img_size=img_size, overlap=overlap)
    return unique_rows(outlist)


def snap_square(long_range = [-3.3,-3], lat_range= [51,51.3], zoom = ZOOM_LEVEL, img_size = IMAGE_SIZE, overlap=.1):
    DEFAULT_LAT_DIFFERENCE_AT_19 = 0.0008#0.0010
    DEFAULT_LONG_DIFFERENCE_AT_19 = 0.0010#0.0015
    long_dense_grid = np.arange(long_range[0], long_range[1],DEFAULT_LONG_DIFFERENCE_AT_19)
    lat_dense_grid = np.arange(lat_range[0], lat_range[1],DEFAULT_LAT_DIFFERENCE_AT_19)
    inlist = np.array([[lng, lat] for lng in long_dense_grid for lat in lat_dense_grid])
    return snap_list(inlist, zoom=zoom, img_size=img_size, overlap=overlap)

#t=snap_square()

