import json
import math
import time

import requests


class EsriClient:
    def __init__(self):
        self.__token = "0VzbnJ_GntgdEB4Mf8EcEi7rnMCJg8TvGfvjaaW9HTIF2jMqAncVfdRdnaMGJXtnakO66e4rX8RE7qZ4TWOjLaGtoXT3QhLGJfDxwhZ7Mkfl_Y_iG_T8pPgmlRcrHqCy8_0z98AOjb_1ynVr0HfzHOH-_RbV0IOETX_mK32eQfPooobh7eAS8u_DmnXUYpylHbxKSKGMzKYPuZ4W9RnO1Wz9EFUusLmnNH6lJJjliW_ZO42J_2X_6B4ojzVOLsvGzW0-jmQDc1z7IhF-aALC0gs6W1AvfnbbyKszFb70fxkId0uGEL9kwEiIR46A7sve"

    @staticmethod
    def get_bounding_box(lat, lon, d):
        """ d: offset in meters """

        r = 6378137  # Earth’s radius, sphere

        # Coordinate offsets in radians
        d_lat = (d / r)
        d_lon = d / (r * math.cos(math.pi * lat / 180))

        # Coordinate offsets in decimal degrees
        d_lat = d_lat * 180 / math.pi
        d_lon = d_lon * 180 / math.pi

        min_lat = lat - d_lat
        max_lat = lat + d_lat
        min_lon = lon - d_lon
        max_lon = lon + d_lon

        return min_lat, max_lat, min_lon, max_lon

    def get_traffic_counts_by_bounding_box(self, min_lat, max_lat, min_lon, max_lon):
        url = f"https://demographics5.arcgis.com/arcgis/rest/services/USA_Traffic_Counts/MapServer/0/query?" \
              f"f=json" \
              f"&outFields=CrossSt%2CCrossDir%2CCrossDist%2CCnt1year%2CCnttype1%2CStreet%2CTraffic1%2CTraffic2%2CTraffic3%2CTraffic4%2CTraffic5%2COBJECTID" \
              f"&outSR=102100" \
              f"&returnM=true" \
              f"&returnZ=true" \
              f"&spatialRel=esriSpatialRelIntersects" \
              f"&where=(Latitude%20BETWEEN%20{min_lat}%20AND%20{max_lat})%20AND%20(Longitude%20BETWEEN%20{min_lon}%20AND%20{max_lon})" \
              f"&token={self.__token}"
        response = requests.get(url)
        return response.json()


if __name__ == '__main__':
    esri_client = EsriClient()

    lat_0 = 40.6653
    lon_0 = -73.72904
    row82_point = (28.402451, -81.243217)

    # bounding_box = esri_client.get_bounding_box(*row82_point, 5000)
    # print(bounding_box)

    bounding_box = (28.357535235794025, 28.447366764205974, -81.29427921343128, -81.19215478656872)
    # bounding_box = (38.9369, 39.0077, -110.3378, -110.1774)
    json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box)
    print(json_response)

    with open('row82_5000m_response.json', 'w') as f:
        json.dump(json_response, f)

    print("done")

