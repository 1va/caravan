'''
Trying to figure out what the coverage of downloaded maps is:
England fits image 600x600 at zoom 7
Splitting the image into 25000 (that is daily limit) improves the zoom by 7

If we want zoom 19 it would take 734 days. Each day we would cover area of half of the New Forrest
At zoom 18 its quarter of the time and 4*the area per day.
'''


from __future__ import division
import math
MERCATOR_RANGE = 256

def  bound(value, opt_min, opt_max):
  if (opt_min != None):
    value = max(value, opt_min)
  if (opt_max != None):
    value = min(value, opt_max)
  return value

def  degreesToRadians(deg) :
  return deg * (math.pi / 180)

def  radiansToDegrees(rad) :
  return rad / (math.pi / 180)


class G_Point :
    def __init__(self,x=0, y=0):
        self.x = x
        self.y = y


class G_LatLng :
    def __init__(self,lt, ln):
        self.lat = lt
        self.lng = ln

class MercatorProjection :
    def __init__(self) :
      self.pixelOrigin_ =  G_Point( MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
      self.pixelsPerLonDegree_ = MERCATOR_RANGE / 360
      self.pixelsPerLonRadian_ = MERCATOR_RANGE / (2 * math.pi)


    def fromLatLngToPoint(self, latLng, opt_point=None) :
      point = opt_point if opt_point is not None else G_Point(0,0)
      origin = self.pixelOrigin_
      point.x = origin.x + latLng.lng * self.pixelsPerLonDegree_
      # NOTE(appleton): Truncating to 0.9999 effectively limits latitude to
      # 89.189.  This is about a third of a tile past the edge of the world tile.
      siny = bound(math.sin(degreesToRadians(latLng.lat)), -0.9999, 0.9999)
      point.y = origin.y + 0.5 * math.log((1 + siny) / (1 - siny)) * -     self.pixelsPerLonRadian_
      return point

    def fromPointToLatLng(self,point) :
      origin = self.pixelOrigin_
      lng = (point.x - origin.x) / self.pixelsPerLonDegree_
      latRadians = (point.y - origin.y) / -self.pixelsPerLonRadian_
      lat = radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
      return G_LatLng(lat, lng)

#pixelCoordinate = worldCoordinate * pow(2,zoomLevel)

def getCorners(center, zoom=10, mapWidth=600, mapHeight=600):
    scale = 2**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(center)
    SWPoint = G_Point(centerPx.x-(mapWidth/2)/scale, centerPx.y+(mapHeight/2)/scale)
    SWLatLon = proj.fromPointToLatLng(SWPoint)
    NEPoint = G_Point(centerPx.x+(mapWidth/2)/scale, centerPx.y-(mapHeight/2)/scale)
    NELatLon = proj.fromPointToLatLng(NEPoint)
    return {
        'N' : NELatLon.lat,
        'E' : NELatLon.lng,
        'S' : SWLatLon.lat,
        'W' : SWLatLon.lng,
    }

centerLat =50.7831533
centerLon = -0.9574026
centerPoint = G_LatLng(centerLat, centerLon)
corners = getCorners(centerPoint, zoom=18,mapWidth=600, mapHeight=600)
#corners = getCorners(centerPoint, zoom=17,mapWidth=300, mapHeight=300)
LAT_DIFFERENCE = corners['N']-corners['S'] #0.0008#0.0010
LONG_DIFFERENCE = corners['E']-corners['W'] #0.00010#0.0016
print ('N-S LAT_DIFFERENCE : %.6f, E-W LONG_DIFFERENCE : %.6f ' %(LAT_DIFFERENCE ,LONG_DIFFERENCE ))


England_LAT_DIFFERENCE = 5  # cca from 50.6 to 55.5
England_LONG_DIFFERENCE = 6  # cca from -4.4 to 1.4

day_limit= round(25000**.5)
corners = getCorners(centerPoint, zoom=19, mapWidth=600*day_limit, mapHeight=600*day_limit)
LAT_DIFFERENCE = corners['N']-corners['S'] #0.0008#0.0010
LONG_DIFFERENCE = corners['E']-corners['W'] #0.00010#0.0016
print ('N-S LAT_DIFFERENCE : %.6f, E-W LONG_DIFFERENCE : %.6f ' %(LAT_DIFFERENCE ,LONG_DIFFERENCE ))
time_required = England_LAT_DIFFERENCE*England_LONG_DIFFERENCE/LAT_DIFFERENCE/LONG_DIFFERENCE
print('Time required %.0f days.' % (time_required))


'''
http://maps.googleapis.com/maps/api/staticmap?center=50.7831533,-0.9574026&zoom=5&size=600x600&format=png&maptype=satellite

{'E': -65.710988,
'N': 74.11120692972199,
'S': 0.333879313530149,
'W': -178.210988}
'''