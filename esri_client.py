import json
import math
from typing import Optional

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

    def get_traffic_counts_by_bounding_box(self, min_lat, max_lat, min_lon, max_lon,
                                           street_filter: Optional[str] = None):
        where_clause = f"(Latitude%20BETWEEN%20{min_lat}%20AND%20{max_lat})%20AND%20(Longitude%20BETWEEN%20{min_lon}%20AND%20{max_lon})"
        if street_filter is not None:
            street_filter = street_filter.lower()
            where_clause += f"%20AND%20((LOWER(Street)%20LIKE%20%27%25{street_filter}%25%27)OR(LOWER(CrossSt)%20LIKE%20%27%25{street_filter}%25%27))"
        # f"&outFields=CrossSt%2CCrossDir%2CCrossDist%2CCnt1year%2CCnttype1%2CStreet%2CTraffic1%2CTraffic2%2CTraffic3%2CTraffic4%2CTraffic5%2COBJECTID" \
        # f"&outSR=102100" \
        #               f"&returnM=true" \
        #               f"&returnZ=true" \
        # f"&spatialRel=esriSpatialRelIntersects" \
        url = f"https://demographics5.arcgis.com/arcgis/rest/services/USA_Traffic_Counts/MapServer/0/query?" \
              f"f=json" \
              f"&outFields=CrossSt%2CCrossDir%2CCrossDist%2CCnt1year%2CCnttype1%2CStreet%2CTraffic1%2CTraffic2%2CTraffic3%2CTraffic4%2CTraffic5%2COBJECTID" \
              f"&where={where_clause}" \
              f"&token={self.__token}"
        response = requests.get(url)
        return response.json()

    def run_sandbox(self):
        # row0_point = (40.6653, -73.72904)
        # row2_point = (38.8972, -104.83874)
        # row82_point = (28.402451, -81.243217)
        # row25_point = (33.97376, -86.44779)
        row15_point = (45.06065, -83.4525)

        box_size = 3000
        bounding_box = self.get_bounding_box(*row15_point, box_size)
        print(f"p {row15_point} size {box_size}m bounding_box: {bounding_box}")

        # bounding_box = (28.357535235794025, 28.447366764205974, -81.29427921343128, -81.19215478656872)
        # bounding_box = (38.9369, 39.0077, -110.3378, -110.1774)

        # TODO check if street name exists - otherwise use joined on all 'street' substrings keys?
        # json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box, "6th")
        json_response = self.get_traffic_counts_by_bounding_box(*bounding_box)

        with open(f"output/row15_{box_size}m_response_reprocessed_unfiltered.json", 'w') as f:
            json.dump(json_response, f)

        print("done")


if __name__ == '__main__':
    esri_client = EsriClient()

    esri_client.run_sandbox()

