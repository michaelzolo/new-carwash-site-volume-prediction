import math
from typing import Optional

import requests


class EsriClient:
    def __init__(self):
        # self.__token = "0VzbnJ_GntgdEB4Mf8EcEi7rnMCJg8TvGfvjaaW9HTIF2jMqAncVfdRdnaMGJXtnakO66e4rX8RE7qZ4TWOjLaGtoXT3QhLGJfDxwhZ7Mkfl_Y_iG_T8pPgmlRcrHqCy8_0z98AOjb_1ynVr0HfzHOH-_RbV0IOETX_mK32eQfPooobh7eAS8u_DmnXUYpylHbxKSKGMzKYPuZ4W9RnO1Wz9EFUusLmnNH6lJJjliW_ZO42J_2X_6B4ojzVOLsvGzW0-jmQDc1z7IhF-aALC0gs6W1AvfnbbyKszFb70fxkId0uGEL9kwEiIR46A7sve"
        self.__token = "N39WC51_x8vaggafDm6fb2QYFOvKDxrNsaO5CcBjraLdXNJ4kzK33nbPQwRDFAtG207d0MuOmv4xJjVK_1Tuq_zq0ZevSVXQHiG_y77nef4Mvpky2R14xVkWpkc0NYYB4zHoTx8O06TFjvSVAAUIxG7vMMdV9pCAHRyCDULhgq4Dm2lUQ2mTHASYxzAnPFFVU5lkV34GqXPmvQLeXghqz-1eiKk_PZBVVMEstzBpGt91qXyZB6YoUVC_wA_pnkaQayekJ9f4zC1UjgxIQAuYqUCOMJ_F5s9W_KBDdqIyJ1Vj7rfEkCZhx4Lu5lZbTZla"

    @staticmethod
    def get_bounding_box(lat, lon, d):
        """ d: offset (in all directions) in meters """

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
        out_fields = "CrossSt%2CCrossDir%2CCrossDist%2CCnt1year%2CCnttype1%2CStreet%2CTraffic1%2CTraffic2%2CTraffic3%2CTraffic4%2CTraffic5%2COBJECTID"
        where_clause = f"(Latitude%20BETWEEN%20{min_lat}%20AND%20{max_lat})%20AND%20(Longitude%20BETWEEN%20{min_lon}%20AND%20{max_lon})"
        if street_filter is not None:
            street_filter = street_filter.lower()
            where_clause += f"%20AND%20((LOWER(Street)%20LIKE%20%27%25{street_filter}%25%27)OR(LOWER(CrossSt)%20LIKE%20%27%25{street_filter}%25%27))"
        url = f"https://demographics5.arcgis.com/arcgis/rest/services/USA_Traffic_Counts/MapServer/0/query?" \
              f"f=json" \
              f"&outFields={out_fields}" \
              f"&where={where_clause}" \
              f"&token={self.__token}"
        response = requests.get(url)
        return response.json()


if __name__ == '__main__':
    pass
