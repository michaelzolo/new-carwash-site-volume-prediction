from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from scipy.spatial import distance
import json
import usaddress
import pandas as pd
import copy
from pyproj import Transformer

from esri_client import EsriClient


def save_df_to_csv(df: pd.DataFrame, output_path: str):
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path_obj)


def save_json_to_file(data, output_path):
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w') as f:
        json.dump(data, f)


class EsriTrafficCountAnalyzer:
    def __init__(self, limit_num_closest_same_street: int = 3, limit_num_closest_unfiltered: int = 5,
                 box_size_m: int = 3000, output_cache_dir_path):
        self.__limit_num_closest_same_street = limit_num_closest_same_street
        self.__limit_num_closest_unfiltered = limit_num_closest_unfiltered
        self.__box_size_m = box_size_m
        self.__output_cache_dir_path = output_cache_dir_path

    def compute_average_count_from_closest_feature_points(self, feature_points_response_file_path: str | Path,
                                                          input_xy: tuple,
                                                          input_address: str,
                                                          limit_num_closest_to_use: int):
        feature_points = EsriTrafficCountAnalyzer.get_feature_points_from_file(feature_points_response_file_path)

        points_dict = EsriTrafficCountAnalyzer.transform_feature_points_to_points_dict(feature_points)

        input_address_tags = usaddress.tag(input_address)[0]
        print(f"Analyzing input point {input_xy} on street {input_address_tags['StreetName']}.")
        input_street_tags = dict(filter(lambda item: 'street' in item[0].lower(), input_address_tags.items()))
        input_street = input_street_tags['StreetName']

        # Same street. Up to 5 years old. (The best alternative)
        output_path = self.__output_cache_dir_path + "/processed_input_points/x{}_y{}_{}/same-5yo.csv".format(*input_xy,
                                                                                                              input_street)
        df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                    filter_street_substring=input_street,
                                                                    filter_years_ago=5,
                                                                    output_path=output_path)
        if df_closest.empty:
            # Same street. Any year.
            output_path = self.__output_cache_dir_path + "/processed_input_points/x{}_y{}_{}/same-any.csv".format(
                *input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_street_substring=input_street,
                                                                        output_path=output_path)
        if df_closest.empty:
            # Cross street. Up to 5 years old.
            output_path = self.__output_cache_dir_path + "/processed_input_points/x{}_y{}_{}/cross-5yo.csv".format(
                *input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_cross_st_substring=input_street,
                                                                        filter_years_ago=5,
                                                                        output_path=output_path)
        if df_closest.empty:
            # Cross street. Any year.
            output_path = self.__output_cache_dir_path + "/processed_input_points/x{}_y{}_{}/cross-any.csv".format(
                *input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_cross_st_substring=input_street,
                                                                        output_path=output_path)
        if df_closest.empty:
            print(
                f"!! ATTENTION: haven't found any closest points in: {points_dict}. Consider sending a different request, but first check the closest point filters!")
            return None, None, None, None
        else:
            total_closest_found = df_closest.shape[0]
            df_closest = df_closest.iloc[:limit_num_closest_to_use]
            mean_count = df_closest['Traffic1'].mean()
            most_frequent_count_year = df_closest['Cnt1year'].mode().max()
            return mean_count, most_frequent_count_year, df_closest.shape[0], total_closest_found

    @staticmethod
    def get_closest_points_df(p_xy, dict_f_p, output_path, limit: Optional[int] = None,
                              filter_street_name_tag: Optional[str] = None,
                              filter_street_substring: Optional[str] = None,
                              filter_cross_st_substring: Optional[str] = None,
                              filter_years_ago: Optional[int] = None):
        """ Return the closest points to p_xy among the points from dict_f_p. """
        filter_street_name_tag = None
        filter_street_substring = None
        filter_cross_st_substring = None

        list_xy = list(dict_f_p.keys())

        dict_closest = {}

        num_xy_processed = 0
        if limit is None:
            limit = len(list_xy)

        while len(list_xy) > 0:
            closest_xy = EsriTrafficCountAnalyzer.closest_point(p_xy, list_xy)
            closest_p = copy.deepcopy(dict_f_p[closest_xy])

            # Parse street tags.
            try:
                closest_p_tags = usaddress.tag(closest_p['Street'])
            except:  # TODO narrow exceptions?
                closest_p_tags = closest_p['Street']
            if type(closest_p_tags) is tuple and 'StreetName' in closest_p_tags[0].keys():
                closest_p_street_name_tag = closest_p_tags[0]['StreetName']
            else:
                closest_p_street_name_tag = closest_p['Street']
                print(f"(Using raw street name \"{closest_p_street_name_tag}\" for point {closest_xy})")

            # Apply filters and get the actual point.
            if (
                    filter_street_name_tag is None or closest_p_street_name_tag.lower() == filter_street_name_tag.lower()) and (
                    filter_street_substring is None or filter_street_substring.lower() in closest_p[
                'Street'].lower()) and (
                    filter_cross_st_substring is None or filter_cross_st_substring.lower() in closest_p[
                'CrossSt'].lower()) and (
                    filter_years_ago is None or datetime.now().year - closest_p['Cnt1year'] <= filter_years_ago):
                closest_p['street_name_tag'] = closest_p_street_name_tag  # TODO REMOVE DEBUG
                closest_p['street_tags'] = closest_p_tags[
                    0]  # TODO make it ok with the case when it's a string. I guess just move the subscript to up.
                dict_closest[closest_xy] = closest_p

            list_xy.remove(closest_xy)

            num_xy_processed += 0
            if num_xy_processed >= limit:
                break

        # Save as a DataFrame to CSV file. (In the future this CSV can be used for debugging and caching).
        df_closest = EsriTrafficCountAnalyzer.dict_points_to_df(dict_closest)
        save_df_to_csv(df_closest, output_path)

        return df_closest

    @staticmethod
    def dict_points_to_df(dict_points):
        df_points = pd.DataFrame.from_dict(dict_points, orient='index')
        if not df_points.empty:
            df_points.index = df_points.index.map(lambda xy: EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(*xy))
            df_points.index.names = ['lat', 'lon']
        return df_points

    @staticmethod
    def save_df_to_csv(df, output_dir, output_file):
        output_path_obj = Path(output_dir)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path_obj / output_file)

    @staticmethod
    def closest_point(point, points):
        closest_index = distance.cdist([point], points).argmin()
        return points[closest_index]

    @staticmethod
    def get_feature_points_from_file(file_path):
        features = []
        with open(file_path, 'r') as file:
            data = json.load(file)
            features = data['features']
        return features

    @staticmethod
    def transform_feature_points_to_points_dict(f_ps):
        points_dict = {}
        for f_p in f_ps:
            points_dict[(f_p['geometry']['x'], f_p['geometry']['y'])] = f_p['attributes']
        return points_dict

    @staticmethod
    def convert_xy_to_lat_lon(x, y):
        transformer = Transformer.from_crs('epsg:3857', 'epsg:4326')
        return transformer.transform(x, y)

    @staticmethod
    def convert_lat_lon_to_xy(lon, lat):
        transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
        return transformer.transform(lon, lat)

    def analyze(self, input_lat_lot, response_cache_json_path, input_address: str = None,
                resend_request_if_no_unfiltered_data: bool = True):
        esri_client = EsriClient()

        bounding_box = esri_client.get_bounding_box(*input_lat_lot, self.__box_size_m)
        print(f"bounding_box: {bounding_box}")

        if not Path(response_cache_json_path).is_file():
            json_response = None
            while json_response is None:
                try:
                    print("Sending request...")
                    if input_address is not None:
                        street_filter = usaddress.tag(input_address)[0]['StreetName']
                    else:
                        street_filter = None
                    json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box, street_filter)
                except:
                    print("Failed request. Trying again...")
                    time.sleep(1)
            save_json_to_file(json_response, response_cache_json_path)

        mean_count, most_frequent_count_year, num_closest_used, total_closest_found = self.compute_average_count_from_closest_feature_points(
            response_cache_json_path,
            self.convert_lat_lon_to_xy(*input_lat_lot),
            input_address,
            limit_num_closest_to_use=self.__limit_num_closest_same_street
        )

        if mean_count is None:
            # Trying to find data without the street filter.
            response_cache_json_path = Path(os.path.splitext(response_cache_json_path)[0] + '_unfiltered.json')

            if response_cache_json_path.exists():
                # Look at previously saved responses of unfiltered data.
                print(f"Found unfiltered response data for point {input_lat_lot}")
                mean_count, most_frequent_count_year, num_closest_used, total_closest_found = self.compute_average_count_from_closest_feature_points(
                    response_cache_json_path,
                    self.convert_lat_lon_to_xy(*input_lat_lot),
                    input_address,
                    limit_num_closest_to_use=self.__limit_num_closest_unfiltered
                )
            elif mean_count is None and resend_request_if_no_unfiltered_data:
                # Send a new request.
                print(f"*** Sending (new) unfiltered request for point {input_lat_lot}")
                json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box)

                save_json_to_file(json_response, response_cache_json_path)

                mean_count, most_frequent_count_year, num_closest_used, total_closest_found = self.compute_average_count_from_closest_feature_points(
                    response_cache_json_path,
                    self.convert_lat_lon_to_xy(*input_lat_lot),
                    input_address,
                    limit_num_closest_to_use=self.__limit_num_closest_unfiltered
                )
        return mean_count, most_frequent_count_year, num_closest_used, total_closest_found

    def analyze_by_api(self, lat, lon, input_address: str = None, resend_request_if_no_unfiltered_data: bool = False):
        input_lat_lot = (lat, lon)
        response_cache_json_path = self.__output_cache_dir_path + f"/responses/lat{lat}_lon{lon}_{self.__box_size_m}m_response.json"

        analysis_result = self.analyze(input_lat_lot, response_cache_json_path, input_address,
                                       resend_request_if_no_unfiltered_data)
        (mean_count, most_frequent_count_year, num_closest_used, total_closest_found) = analysis_result
        print(analysis_result)
        return mean_count, most_frequent_count_year, num_closest_used, total_closest_found

    def analyze_by_csv(self, input_csv_path, resend_request_if_no_unfiltered_data: bool = False):
        input_csv_df = pd.read_csv(input_csv_path)

        traffic_count_list, traffic_year_list, traffic_num_closest_used, traffic_total_closest = [], [], [], []
        for row_num, row in input_csv_df.iterrows():
            print(f"----------------- row_num: {row_num} ----------------------")
            input_lat_lot = (row['Latitude'], row['Longitude'])
            input_address = row['address']
            response_cache_json_path = self.__output_cache_dir_path + f"/responses/row{row_num}_{self.__box_size_m}m_response.json"

            analysis_result = self.analyze(input_lat_lot, response_cache_json_path, input_address,
                                           resend_request_if_no_unfiltered_data)
            (mean_count, most_frequent_count_year, num_closest_used, total_closest_found) = analysis_result

            traffic_count_list.append(mean_count)
            traffic_year_list.append(most_frequent_count_year)
            traffic_num_closest_used.append(num_closest_used)
            traffic_total_closest.append(total_closest_found)

        num_rows = input_csv_df.shape[0]
        traffic_count_list += [None] * (num_rows - len(traffic_count_list))
        traffic_year_list += [None] * (num_rows - len(traffic_year_list))
        traffic_num_closest_used += [None] * (num_rows - len(traffic_num_closest_used))
        traffic_total_closest += [None] * (num_rows - len(traffic_total_closest))
        input_csv_df['traffic_count'] = traffic_count_list
        input_csv_df['traffic_year'] = traffic_year_list
        input_csv_df['traffic_num_closest_used'] = traffic_num_closest_used
        input_csv_df['traffic_total_closest'] = traffic_total_closest

        input_csv_df.to_csv(os.path.splitext(input_csv_path)[0] + '_with_counts.csv')


if __name__ == '__main__':
    analyzer = EsriTrafficCountAnalyzer(output_cache_dir_path=r"output/api")  # output_cache_dir_path without '/' on the edges
    lat = 40.6653
    lon = - 73.72904

    traffic_count, most_frequent_count_year, num_closest_used, total_closest_found = analyzer.analyze_by_api(lat, lon)