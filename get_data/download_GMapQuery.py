"""
Description:    A series of functions to download static google satellite images
                into mongodb and then retrieve them when needed.
Author:         Bence Komarniczky, modified by Iva
Date:           15/06/2015
Python version: 3.4 -> now 2.7
"""

import urllib
from os import path, makedirs
from time import sleep
import csv

import numpy as np
from PIL import Image
from pymongo.collection import Collection
from pymongo.errors import AutoReconnect
from bson.son import SON
from io import BytesIO


# define constant difference between grid points if using zoom_level=19
DEFAULT_LAT_DIFFERENCE_AT_19 = 0.0010
DEFAULT_LONG_DIFFERENCE_AT_19 = 0.0015


class GMapQuery(object):
    """
    A GMapQuery object is an instance of google map query. It can be used to
    fetch and download static images through the GoogleMaps API
    """

    # define some fixed variables for the API call
    beginning = "https://maps.googleapis.com/maps/api/staticmap?"
    key = 'key=AIzaSyADEXfeHpcdEXfsEWF_F7iX5bKzVSnlkk8'
 #   key = "key=AIzaSyAy6h3CDuz0FXXivMpTTZlWlyMXaZWNHPQ"  #invalid for unknown reason
    format = "format=png"
    map_type = "maptype=satellite"
    size = "size=600x600"

    def __init__(self, latitude, longitude, zoom_level=18):
        """
        Provide a float pair of latitude/longitude and a zoom level to create a new
        GMapQuery object. This object will then be used to access the GMaps API.
        :rtype : GMapQuery
        :param latitude:    latitude of the center point
        :param longitude:   longitude of the center point
        :param zoom_level:  level of zoom to generate image. See Google API documentation

        :type latitude:     float
        :type longitude:    float
        :type zoom_level:   int
        """

        # grab data from init
        self.longitude = round(longitude, 4)
        self.latitude = round(latitude, 4)
        self.center = "center=" + str(latitude) + "," + str(longitude)
        self.zoom = "zoom=%d" % zoom_level

        # construct API call
        self.google_query = "&".join([GMapQuery.beginning,
                                      self.center,
                                      self.zoom, GMapQuery.format,
                                      GMapQuery.map_type, GMapQuery.size, GMapQuery.key])

    def get_image(self):
        """
        Return a grey scale PIL.Image.Image object with resolution of 300x300.
        """
        file = None

        # query Google
        try:
            file = urllib.urlopen(self.google_query)
        except :
            # in case of error retry 5 times
            i = 0
            while i < 5:
                sleep(10)
                try:
                    file = urllib.urlopen(self.google_query)
                    break
                except URLError:
                    # increment attempt count
                    i += 1
                    continue

        # in case of fatal error return None
        if file is None:
            return None

        b = BytesIO(file.read())
        img = Image.open(b)

        # convert to grey scale and then resize
        img = img.convert('L')
        img.thumbnail((300, 300), Image.ANTIALIAS)

        # print stats
        print("fetched image for lat: %2.4f long: %2.4f" % (self.latitude, self.longitude))

        return img

    def download_image(self, file_path):
        """
        Save the corresponding satellite image to disk, after fetching from Google.

        :param file_path:   File path to save image to. Extension should be png.
        :return error code: 1 = download error

        :type file_path:    str
        :rtype              int
        """

        # download image
        image = self.get_image()

        # if download failed return error code 1
        if image is None:
            print("**** Error: image failed to download: lat: %2.4f long: %2.4f" % (self.latitude, self.longitude))
            return 1

        # if all went okay save image and
        image.save(file_path)

        return 0

    def insert_mongo(self, mongo_db_database, classification):
        """
        Given a mongodb collection insert a new document into the specified mongodb
        collection with the image as a binary numpy array.

        :param mongo_db_database:   Mongodb collection to insert image into
        :param classification:      boolean classification of whether image contains
                                    trailer parks or not
        :return error_code:         integer error code.
                                    1 = download error
                                    2 = image already in collection
                                    3 = insert failed

        :type mongo_db_database:    Collection
        :type classification:       bool
        :rtype                      int
        """

        assert type(mongo_db_database) == Collection, \
            "Must supply valid mongodb collection"

        # check if image is already in the database
        if mongo_db_database.find({"coordinates": [self.latitude, self.longitude]}).count() > 0:
            print("** Image is already in database lat: %2.4f long: %2.4f" % (self.latitude, self.longitude))
            return 2

        # fetch image
        image = self.get_image()

        # check if image is valid
        if image is None:
            print("Image was invalid! lat: %2.4f long: %2.4f" % (self.latitude, self.longitude))
            return 1

        # convert image to numpy array, resize and convert to binary
        image_array = np.asarray(image, dtype='uint8')

        image_byte = bson.BSON.encode({'i':image_array.tolist()})

        # construct document to be inserted
        doc = bson.BSON.encode({"coordinates": [self.longitude, self.latitude],
               "image": image_byte,
               "class": classification})

        # insert into mongodb
        try:
            mongo_db_database.insert_one(doc)
        except AutoReconnect:

            # attempt connection 5 more times
            i = 0
            while i < 5:
                # wait 5 seconds
                sleep(5)

                try:
                    mongo_db_database.insert_one(doc)
                    return 0
                except AutoReconnect:
                    i += 1
                    continue

            # if retries all failed return error code of 3
            return 3

        # return error code
        return 0


def download_list(coordinates, mongo_collection, zoom_level=17, classification=False):
    """
    Given a list of coordinates, download the corresponding satellite images and insert them into MongoDB.

    :param coordinates:         coordinates to be fetched
    :param mongo_collection:    collection to insert images into
    :param zoom_level:          level of zoom default: 17 (check GoogleAPI)
    :param classification:      whether the resulting images should be labelled as True/False
    :return:                    a tuple of errors/successes

    :type coordinates           list or tuple
    :type mongo_collection      Collection
    :type zoom_level            int
    :type classification        bool
    :rtype                      tuple
    """

    # keep track of errors
    error_stats = [0, 0, 0, 0]

    # loop through list of coordinates, create GMapQuery objects and insert into database
    for coordinate in coordinates:
        new_query = GMapQuery(coordinate[0], coordinate[1], zoom_level=zoom_level)
        new_error = new_query.insert_mongo(mongo_db_database=mongo_collection,
                                           classification=classification)

        # keep track of errors
        error_stats[new_error] += 1

    return tuple(error_stats)


def download_strip(left, right, mongo_collection, zoom_level=17):
    """
    Download a vertical strip of images from google maps.
    :param left:                Coordinate points of leftmost image.
    :param right:               Coordinate points of rightmost image.
    :param mongo_collection:    Mongo collection to insert data to.
    :param zoom_level:          Level of zoom for images.
    :return:                    Error tuple from download_list

    :type left                  list or tuple
    :type right                 list or tuple
    :type mongo_collection      Collection
    :type zoom_level            int
    :rtype                      tuple
    """

    # check validity of left/right coordinates: they must be level!
    assert left[0] == right[0], "left and right coordinate points must be level"

    # generate step size to ensure images don't miss anything. There will be
    # a bit of overlap though
    step_size_x = DEFAULT_LONG_DIFFERENCE_AT_19 * 1.8 ** (19 - zoom_level)

    # create list of points from step size info
    coordinate_points = [[left[0], x] for x in np.arange(left[1], right[1], step_size_x)]

    # download this list of points
    error_tuple = download_list(coordinate_points,
                                mongo_collection=mongo_collection,
                                zoom_level=zoom_level)

    return error_tuple


def download_box(lower_left, upper_right, mongo_collection, zoom_level=17):
    """
    Given the bottom left coordinate and the upper right coordinate of a rectangular shape
    download a complete set of images that cover the whole area. Insert these into Mongo.

    :param lower_left:          Coordinates of lower left point.
    :param upper_right:         Coordinates of upper right point.
    :param mongo_collection:    Mongo collection to store images in.
    :param zoom_level:          Level of zoom.
    :return:                    A tuple of error code counts.

    :type lower_left:           list or tuple
    :type upper_right:          list or tuple
    :type mongo_collection      Collection
    :type zoom_level            int
    :rtype                      tuple
    """

    # check for validity of bounding points. Their relative position must be as described
    assert lower_left[0] < upper_right[0] and lower_left[1] < upper_right[1], "Invalid coordinates supplied"

    # generate step size to tile rectangle efficiently
    step_size_y = DEFAULT_LAT_DIFFERENCE_AT_19 * 1.8 ** (19 - zoom_level)

    # create list of y_coordinates to as starting points for vertical strips
    y_coordinates = np.arange(lower_left[0], upper_right[0], step_size_y)

    # initialise errors as None
    collect_errors = None

    # loop through starting points and download one strip for each
    for y_coord in y_coordinates:

        # in first instance set collect_errors to the first error counts received
        if collect_errors is None:
            collect_errors = np.array(download_strip((y_coord, lower_left[1]),
                                                     (y_coord, upper_right[1]),
                                                     mongo_collection=mongo_collection,
                                                     zoom_level=zoom_level))

        # in all other instances just collect into collect_errors
        else:
            collect_errors += np.array(download_strip((y_coord, lower_left[1]),
                                                      (y_coord, upper_right[1]),
                                                      mongo_collection=mongo_collection,
                                                      zoom_level=zoom_level))

    # return immutable tuple of error counts
    return tuple(collect_errors)


def save_to_disk(file_path, mongo_collection, query=None, include_csv=True, limit=500):
    """
    Retrieve a set of images from a mongo db collection that match the query and
    save them to a folder with lat_long.png names.

    :param file_path:           folder to save images into. Need not to exist.
    :param mongo_collection:    mongo db collection to retrieve images from
    :param query:               mongo db query to specify which images need downloading
                                default=None: all images will be downloaded
                                Should be of form: {"near": [longitude, latitude], "maxDistance": km}
    :param include_csv:         Boolean flag specifying whether a csv file should be created along
                                with the images that can be used to mark interesting images.
    :param limit:               The number of images to be downloaded. The default is 500.

    :return         number of images downloaded

    :type file_path:            str
    :type mongo_collection:     Collection
    :type query:                None | dict[str, float]
    :type include_csv           bool
    :type limit                 int
    :rtype                      int
    """

    # specify earth radius for converting radians to km
    const_earth_radius = 6371

    # create folder if file_path doesn't exist
    if not path.exists(file_path):
        makedirs(file_path)

    # populate query with empty dict if default None was supplied
    if query is None:
        cursor = list[mongo_collection.find()]
        query_name = "all_valid"
    else:
        # check for valid query input
        assert "maxDistance" in query.keys(), "You must give a maxDistance parameter to the query parameter!"
        assert "near" in query.keys(), "A 'near' field must be specified in the query parameter."

        # grab the data
        mongo_database = mongo_collection.database
        cursor = mongo_database.command(SON([
            ('geoNear', mongo_collection.name),
            ('near', query["near"]),
            ('spherical', 'true'),
            ('limit', limit),
            ('distanceMultiplier', const_earth_radius),
            ('maxDistance', query["maxDistance"] / const_earth_radius)]))
        cursor = list(cursor["results"])

        query_name = "%2.4f_%2.4f_distance_%4.4f" % (query["near"][0], query["near"][1], float(query["maxDistance"]))

    total_images = 0

    # collect image coordinates
    image_coordinates = []

    # loop through found images
    for one_image in cursor:
        total_images += 1

        # convert from binary to numpy array and then to image
        image_array = np.fromstring(one_image["obj"]["image"], dtype='uint8')
        img = Image.fromarray(image_array.reshape(300, 300), 'L')

        # tell user what has been downloaded
        print("%2.4f_%2.4f" % tuple(one_image["obj"]["coordinates"]))
        image_coordinates.append([one_image["obj"]["coordinates"][0],
                                  one_image["obj"]["coordinates"][1],
                                  one_image["obj"]["_id"],
                                  0])

        # generate file name and save the image
        file_name = "%2.4f_%2.4f" % (one_image["obj"]["coordinates"][0], one_image["obj"]["coordinates"][1])
        img.save("%s/%s.png" % (file_path, file_name))

    if include_csv:
        file_name = "%s/%s.csv" % (file_path, query_name)
        with open(file_name, 'w', newline="\n") as out_file:
            out_csv = csv.writer(out_file, quoting=csv.QUOTE_NONNUMERIC)
            for row in image_coordinates:
                out_csv.writerow(row)

    return total_images
