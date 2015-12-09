import numpy as np
import pymongo

IMAGE_SIZE=300

db_train = pymongo.MongoClient("192.168.0.99:30000")["google"]["trainingset"]
n=db_train.count()
cursor=db_train.find().limit(n)
RGB_profile = np.zeros([n,256,3],dtype='uint16')
y = np.zeros([n], dtype='uint8')
coord = np.zeros([n,2], dtype='float32')
i=0
for one_image in cursor:
    img_array = np.fromstring(one_image["image"], dtype='uint8').reshape(IMAGE_SIZE, IMAGE_SIZE,3)
    for j in range(3):
         x = np.bincount(img_array[:,:,j].reshape([IMAGE_SIZE**2]))
         RGB_profile[i,:len(x),j] = x
    y[i] = int(one_image['class'])
    coord[i,:] = one_image['coordinates']
    i += 1
    if (i%100 ==0):
            print('.'),

data_matrix = np.concatenate([RGB_profile.reshape(n,256*3)], axis=1)
np.savetxt('RGB_profiles.csv', data_matrix,  fmt='%d', delimiter=', ')

data_matrix = np.concatenate([y.reshape([n,1]),coord], axis=1)
np.savetxt('RGB_coords.csv', data_matrix,  fmt='%.6f', delimiter=', ')
