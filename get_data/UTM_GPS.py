"""
Description:    Get list of GPS coordinates of caravan parks from other peoples' lists
Author:         Iva
Date:           07/12/2015
Python version: 2.7
"""

#from utm import to_latlon, from_latlon
from pyproj import transform, Proj
import numpy as np

# forst define the conversion from the stupid british eastling and northling
bng = Proj(init= 'epsg:27700')
wgs84 = Proj(init= 'epsg:4326')
#transform(bng, wgs84, 458528,99293)

neals_data = np.genfromtxt('coord_lists/Neals_List1_final.csv', delimiter=',', skip_header= True, dtype='float')[:,0:2]
x=np.array(transform(bng, wgs84, neals_data[:,0], neals_data[:,1]))
np.savetxt('coord_lists/Neals_List1_final_gps.csv',x.transpose(), fmt='%.6f', delimiter=', ')

neals_coord = np.vstack({tuple(row) for row in np.transpose([x[1],x[0]])})

m77_data = np.genfromtxt('coord_lists/77m_Sample_list.csv', delimiter=',', skip_header= True, dtype='float')[:,0:2]
x=np.array(transform(bng, wgs84, m77_data[:,0], m77_data[:,1]))
np.savetxt('coord_lists/77m_Sample_list_gps.csv',x.transpose(), fmt='%.6f', delimiter=', ')

zoopla_data = np.genfromtxt('get_data/park_homes_zoopla_3col.csv', delimiter=',', skip_header= True, dtype='float')
zoopla_caravans = np.vstack({tuple(row) for row in zoopla_data[zoopla_data[:,2]==1,0:2]})
zoopla_controls = np.vstack({tuple(row) for row in zoopla_data[zoopla_data[:,2]==0,0:2]})

zoopla_controls_subset = zoopla_controls[[5*i for i in range(200*9)],:]

np.savetxt('get_data/GPS_5000caravans.csv', neals_coord[0:5000,:],  fmt='%.6f', delimiter=', ')
np.savetxt('get_data/GPS_5000controls.csv', zoopla_controls[0:5000,:],  fmt='%.6f', delimiter=', ')


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

#random.seed(1234)
#random_areas = [[random.uniform(50.8484,52.0527), random.uniform(-2.75874, 0.4485376)]for i in range(20)]