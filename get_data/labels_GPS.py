"""
Description:    Get GPRs coordinates from labels (classification output)
Author:         Iva
Date:           11/2/2016
Python version: 2.7.10 (venv2)
"""

import numpy as np
from __future__ import division
import math
from map_coverage import MercatorProjection, G_Point, G_LatLng

MERCATOR_RANGE = 256
ZOOM_LEVEL = 17
PIXELperLABEL = 4
IMAGE_SIZE= 300

def labels_suspect(labels, pixels = PIXELperLABEL):
    n = np.sqrt(np.prod(labels.shape)).astype('int')
    ind = labels.reshape(n,n) > .992
    grid = np.round((np.array(range(n))-(n-1)/2.) * pixels)
    xc = np.array([grid for i in range(n)])
    yc = np.transpose(xc)
    return xc[ind].ravel(), yc[ind].ravel()

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

def testrun(i=3):
    y = np.genfromtxt('tmp_images/assemble_y.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    ids = np.genfromtxt('tmp_images/assemble_ids.csv', delimiter=',', skip_header= False).astype(dtype='float32')
    suspect_gps = labels_GPS(labels=y[i,].reshape(70,70), center_gps= ids[i,1:3].ravel())  # gps array = [longitude is greenwitch, latitude is around 50]
    return(suspect_gps)

def snap2grid(obj_gps, zoom = ZOOM_LEVEL, img_size = IMAGE_SIZE - IMAGE_SIZE/10):
    scale = 2.**zoom
    proj = MercatorProjection()
    objPx = proj.fromLatLngToPoint(G_LatLng(obj_gps[1], obj_gps[0]))
    def myround(num):
        return np.round(num*scale/img_size)*img_size/scale
    centerPx = G_Point( myround(objPx.x), myround(objPx.y))
    center_gps = proj.fromPointToLatLng(centerPx)
    return [np.round(center_gps.lng,6), np.round(center_gps.lat,6)]