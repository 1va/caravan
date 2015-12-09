"""
Description:    Get list of GPS coordinates of caravan parks from open street map (.xml)
Author:         Iva
Date:           09/12/2015
Python version: 2.7
"""

import numpy as np
from lxml import etree

from time import ctime
#it takes 15min to parse england
filelist=['cornwall','wales','england', 'scotland']
i=3

A=np.zeros([0,2],dtype='float32')
#for node in tree.getroot().getchildren():
print(ctime())
for event, node in etree.iterparse('/home/bigdata/Downloads/'+filelist[i]+'-latest.osm'):
    if (node.tag=='node'):
        for child in node.getchildren():
            if ((child.tag=='tag')and(child.attrib['k']=='tourism')and(child.attrib['v']=='caravan_site')):
              #try:
                newrow=np.array([[node.attrib['lat'], node.attrib['lon']]],dtype='float32',)
                A = np.concatenate((A,newrow))
    if (node.tag!='tag'):
        node.clear()
              #except :
              #   pass
print(A.shape)
np.savetxt('get_data/GPS_osm_'+filelist[i]+'.csv', A,  fmt='%.6f', delimiter=', ')
print(ctime())

for i in [1,2,3]:
    A = np.concatenate((A,np.genfromtxt('get_data/GPS_osm_'+filelist[i]+'.csv', delimiter=',', skip_header= False, dtype='float')),axis=0)



np.savetxt('get_data/GPS_osm_568caravans.csv', A,  fmt='%.6f', delimiter=', ')