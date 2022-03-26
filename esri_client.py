import json
import math
from pathlib import Path
from typing import Optional

import requests

from arcgis.gis import GIS

ERROR_CODE_INVALID_TOKEN = 498


class EsriClient:
    def __init__(self):
        self.__token_cache_path = "auth_cache/token.json"
        self.__token = self.get_token()

    def get_token(self):
        token = self.get_token_from_file()

        if token is None:
            print("Token cache not found. Generating new token...")
            token = self.generate_new_token()
            print("Token generated.")

        return token

    def generate_new_token(self):
        url = "https://www.arcgis.com/sharing/rest/generateToken?f=json"
        payload = {'username': 'ezcarwash',
                   'password': '3301Hallandale',
                   'referer': 'https://www.arcgis.com/'}
        errormessage=""
        response = None
        try:
            response = requests.post(url, data=payload)
            if response.ok is False:
                errormessage += "Response status code: {} Reason for failure: {}".format(response.status_code,
                                                                                         requests.status_codes._codes[
                                                                                             response.status_code][0])
        except requests.exceptions.RequestException as e:
            errormessage += f"RequestException: {str(e)}"
        if errormessage != "":
            raise requests.exceptions.RequestException("Token request error:" + str(errormessage))

        token_json = response.json()

        self.save_token_json_to_file(token_json)

        return token_json['token']

    def generate_new_token_v1(self):
        ezcarwash_username = "Ezcarwash"
        ezcarwash_password = "3301Hallandale"
        ezcarwash_url_simplified = "https://www.arcgis.com/sharing/rest"

        gis_ezcarwash = GIS(url=ezcarwash_url_simplified, client_id="busanalystonline_2", username=ezcarwash_username,
                            password=ezcarwash_password)

        token = gis_ezcarwash._con.token

        self.save_token_json_to_file(token)

        return token

    def get_token_from_file(self):
        output_path_obj = Path(self.__token_cache_path)

        if not output_path_obj.is_file():
            return None

        with open(output_path_obj, 'r') as f:
            # TODO check expiration
            data = json.load(f)
            return data['token']

    def save_token_json_to_file(self, token_json):
        output_path_obj = Path(self.__token_cache_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w') as f:
            json.dump(token_json, f)

    @staticmethod
    def get_bounding_box(lat, lon, d):
        # TODO unit test for this and similar functions
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
                                           street_filter: Optional[str] = None,
                                           num_tries: int = 3):
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
        res_json = response.json()
        if 'error' in res_json:
            if res_json['error']['code'] == ERROR_CODE_INVALID_TOKEN:
                print("Token error response.")
                if num_tries > 1:
                    print(f"(Retries left: {num_tries}). Generating new token...")
                    self.generate_new_token()
                    print("Token generated. Retying request...")
                    return self.get_traffic_counts_by_bounding_box(min_lat, max_lat, min_lon, max_lon,
                                                                   street_filter,
                                                                   num_tries - 1)
                else:
                    print(f"Exceeded number of retry attempts. Final response: {res_json}")
        return res_json


if __name__ == '__main__':
    pass
    # client = EsriClient()
    # print(client.generate_new_token())
