"""
Description:
Author: Bence Komarniczky
Date:
Python version: 3.4
"""

import pymongo
# import download
from get_data.download_GMapQuery import download_box, save_to_disk, GMapQuery

# to get it into matrix form
# img = Image.open(path)
# np.asarray(img)

# download area around Southampton
db = pymongo.MongoClient("192.168.0.99:30000")["google"]["test"]

# download_box((50.7255, -1.8639), (51.0293, -0.9392), db, zoom_level=17)
save_to_disk("test/HarrowWoodFarm/", db, {"near": [-1.7255, 50.78036], "maxDistance": 0.4})

#save_to_disk("test/ONS/", db, {"near": [-1.2461391, 50.8629015], "maxDistance": 1})

# download_box([50.904, -1.423], [50.939, -1.3787], db, zoom_level=17)
#
# one_point = GMapQuery(50.9183, -1.4077)
# test = one_point.get_image()
#
# one_point.insert_mongo(mongo_db_database=db, image=one_point.get_image(), classification=False)


# one_point = GMapQuery(50.9183, -1.4077)
# one_point.download_image("test_class.png")
#
one_point = GMapQuery(50.9183, -1.4092)

one_point.download_image("test_class2.png")

new_point = GmapQuery(50.7677717,-0.8529036)
one_point.download_image("test_class.png")

# 50.894006, -1.386596
# 50.947139, -1.372863
# 50.946923, -1.454745
#
# coords = [[50.9, x] for x in np.arange(-1.4115, -1.3985, 0.0015)]
#
# download_list(coords, "test")

import pymongo
db_test = pymongo.MongoClient("192.168.0.99:30000")["google"]["test"]

download_box((50.7255, -1.8639), (51.0293, -0.9392), db_test, zoom_level=17)
save_to_disk("test/", db)