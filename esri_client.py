import math
from typing import Optional

import requests

from arcgis.gis import GIS

ERROR_CODE_INVALID_TOKEN = 498


class EsriClient:
    def __init__(self):
        # self.__token = "0VzbnJ_GntgdEB4Mf8EcEi7rnMCJg8TvGfvjaaW9HTIF2jMqAncVfdRdnaMGJXtnakO66e4rX8RE7qZ4TWOjLaGtoXT3QhLGJfDxwhZ7Mkfl_Y_iG_T8pPgmlRcrHqCy8_0z98AOjb_1ynVr0HfzHOH-_RbV0IOETX_mK32eQfPooobh7eAS8u_DmnXUYpylHbxKSKGMzKYPuZ4W9RnO1Wz9EFUusLmnNH6lJJjliW_ZO42J_2X_6B4ojzVOLsvGzW0-jmQDc1z7IhF-aALC0gs6W1AvfnbbyKszFb70fxkId0uGEL9kwEiIR46A7sve"
        # self.__token = "N39WC51_x8vaggafDm6fb2QYFOvKDxrNsaO5CcBjraLdXNJ4kzK33nbPQwRDFAtG207d0MuOmv4xJjVK_1Tuq_zq0ZevSVXQHiG_y77nef4Mvpky2R14xVkWpkc0NYYB4zHoTx8O06TFjvSVAAUIxG7vMMdV9pCAHRyCDULhgq4Dm2lUQ2mTHASYxzAnPFFVU5lkV34GqXPmvQLeXghqz-1eiKk_PZBVVMEstzBpGt91qXyZB6YoUVC_wA_pnkaQayekJ9f4zC1UjgxIQAuYqUCOMJ_F5s9W_KBDdqIyJ1Vj7rfEkCZhx4Lu5lZbTZla"
        # self.__token = "Sm0J--BsyfonpJoP5_V53sjgrhFHoMeLraOJbe34gWuHOueZ99EJ0GuKAuQF6AO7cZ2xEmwDLDxfmUH58IrOGT4E3LXG6qS5mfzyZkxas07KfpcgS_0uBt-8KcufF5770gjmnGoYDkkKhBFQWx4viK6WZI5ZEQO3L_qBHJN7wDy0bbX76lJT_Q2j_y_ri1KNv50N5oOlC-rZ89DyL8xcQfyrLuWtDgR7-mu34fZtFpP4E-XIeZgThIo8J715sgX3goUamLZU3GibBT3LRmTubGcK4ErCubPXvkMUnorZCHqVwrKlBagEmSJh5Y0dihGu"
        # self.__token = "1pqsnfm_ba86N0lohVGKUIqI9vgRWFz3wMNN7kIqNyZ-_qkZIJ0r2VBhwECELtDd5G7ptKBjXWkxMOy6S8KuRnkL6cQ65F4VLLzNIoSn-AyvPjAXGDNYmfpjlwLR338zNt6eEthWtXBW1qixBSMlP3K21I2qKQ2PQiD2xK0jf-MaJINBNr4OpPMx25WNJFbOG4oSbBfctApQFRuwqOjEo-QmjkJYyDjvn9rEz_ujKjboG8w7BMoIaFXyYkZAwt_Yus7rvHm5_ZRv56gR_9QmgJrk-vs0c99d_2NaWH3TZisk8airhGwj_0r5ZN4ljawr"
        self.__token = "3PNe3B9duNckNQpGzM-b1hihmFDcQ2hbfdHEs-9KKFgzcaqeeQJwVrNEjPs7khGebkVk_mzVfGWe5-h1UufQoIwc6hxnwiUKjQgqmHzsNWdNh5Fr0wJTksT2hynm1XeYqXchdevuoe_4BxKGMk9uIOkiJ-HK9ccmRO0wU1-tfwtswUiv2f92LG9lUtwx7JxM0mFEY07QjWKr20O6Maf5PDme1BWkyXUB3mufYXtAj12ESYOPsFLcVGNj_LLbt4x9DCf-BUGosO9M6UYhyBJD0g.."

    # def submit_request(request):
    #     """ Returns the response from an HTTP request in json format."""
    #     with contextlib.closing(urllib2.urlopen(request)) as response:
    #         job_info = json.load(response)
    #         return job_info
    #
    # def get_token(portal_url, username, password):
    #     """ Returns an authentication token for use in ArcGIS Online."""
    #
    #     # Set the username and password parameters before
    #     #  getting the token.
    #     #
    #     params = {"username": username,
    #               "password": password,
    #               "referer": "http://www.arcgis.com",
    #               "f": "json"}
    #
    #     token_url = "{}/generateToken".format(portal_url)
    #     request = urllib2.Request(token_url, urllib.urlencode(params))
    #     token_response = submit_request(request)
    #     if "token" in token_response:
    #         print("Getting token...")
    #         token = token_response.get("token")
    #         return token
    #     else:
    #         # Request for token must be made through HTTPS.
    #         #
    #         if "error" in token_response:
    #             error_mess = token_response.get("error", {}).get("message")
    #             if "This request needs to be made over https." in error_mess:
    #                 token_url = token_url.replace("http://", "https://")
    #                 token = get_token(token_url, username, password)
    #                 return token
    #             else:
    #                 raise Exception("Portal error: {} ".format(error_mess))

    def get_token(self):
        ezcarwash_username = "Ezcarwash"
        ezcarwash_password = "3301Hallandale"

        # ezcarwash_url = "https://www.arcgis.com/sharing/rest/oauth2/authorize?client_id=busanalystonline_2&redirect_uri=https%3A%2F%2Fbao.arcgis.com%2FesriBAO%2Fpostsignin%2Fsignin.html&response_type=token&expiration=20160&locale=en-us&force_login=false"
        ezcarwash_url_simplified = "https://www.arcgis.com/sharing/rest"

        gis_ezcarwash = GIS(url=ezcarwash_url_simplified, client_id="busanalystonline_2", username=ezcarwash_username,
                            password=ezcarwash_password)

        self.__token = gis_ezcarwash._con.token
        print(f"new_token:{self.__token}")
        return self.__token

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
                print("Token error. Generating new token...")
                self.get_token()
                print("Token generated.")
                if num_tries > 1:
                    print("Sending new request...")
                    return self.get_traffic_counts_by_bounding_box(min_lat, max_lat, min_lon, max_lon,
                                                                   street_filter,
                                                                   num_tries - 1)
                else:
                    print(f"Num tries exceeded. Returning final response: {res_json}")
        return res_json


if __name__ == '__main__':
    pass
    # client = EsriClient()
    #
    # # token = client.get_token()
    # # print(token)
    #
    # js = client.get_traffic_counts_by_bounding_box(39.0578534235463, 39.111626576453695, -108.62965808210265,
    #                                                -108.56038191789736)
    # print(js)
