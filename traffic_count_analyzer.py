from __future__ import annotations

import logging
import math
import os
import re
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

from esri_client import EsriClient, LOGGER_NAME

# Defaults (later can move them into config file).
LIMIT_SAME_STREET_DEFAULT = 4
LIMIT_UNFILTERED_DEFAULT = 4
BOX_SIZE_DEFAULT = 3000
RESEND_IF_MISSING_CACHE_UNFILTERED_DEFAULT = True
FILTER_YEARS_AGO_PREFERENCE = 5

log = logging.getLogger(LOGGER_NAME)


class EsriTrafficCountAnalyzer:
    def __init__(self, output_cache_dir_path: str, limit_same_street: int = LIMIT_SAME_STREET_DEFAULT,
                 limit_unfiltered: int = LIMIT_UNFILTERED_DEFAULT,
                 box_size_m: int = BOX_SIZE_DEFAULT,
                 resend_if_missing_cache_unfiltered: bool = RESEND_IF_MISSING_CACHE_UNFILTERED_DEFAULT):
        self.__limit_same_street = limit_same_street
        self.__limit_unfiltered = limit_unfiltered
        self.__box_size_m = box_size_m
        self.__cache_dir = output_cache_dir_path
        self.__esri_client = EsriClient()
        self.__resend_if_missing_cache_unfiltered = resend_if_missing_cache_unfiltered

    def set_limit_same_street(self, limit_same_street):
        self.__limit_same_street = limit_same_street

    def set_limit_unfiltered(self, limit_unfiltered):
        self.__limit_unfiltered = limit_unfiltered

    @staticmethod
    def save_df_to_csv(df: pd.DataFrame, output_path: str):
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path_obj)

    @staticmethod
    def save_json_to_file(data, output_path):
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w') as f:
            json.dump(data, f)

    def analyze_closest(self, p_xy: tuple, dict_xy_fp: dict,
                        address_filter: Optional[str] = None):
        if address_filter is None:
            limit_closest = self.__limit_unfiltered
            street_filter = None
        else:
            limit_closest = self.__limit_same_street
            address_tags = usaddress.tag(address_filter)[0]
            log.info(f"Analyzing point {p_xy} on street {address_tags['StreetName']}.")
            street_tags = dict(filter(lambda item: 'street' in item[0].lower(), address_tags.items()))
            street_filter = street_tags['StreetName']

        # Same street. Up to 5 years old. (The best alternative)
        df_closest = self.get_closest_p_df(p_xy, dict_xy_fp,
                                           street_substring_filter=street_filter,
                                           years_ago_filter=FILTER_YEARS_AGO_PREFERENCE,
                                           cache_file_label=f"same-{FILTER_YEARS_AGO_PREFERENCE}yo")
        if df_closest.empty:
            # Same street. Any year.
            df_closest = self.get_closest_p_df(p_xy, dict_xy_fp,
                                               street_substring_filter=street_filter,
                                               cache_file_label="same-any")
        if df_closest.empty:
            # Cross street. Up to 5 years old.
            df_closest = self.get_closest_p_df(p_xy, dict_xy_fp,
                                               cross_st_substring_filter=street_filter,
                                               years_ago_filter=FILTER_YEARS_AGO_PREFERENCE,
                                               cache_file_label=f"cross-{FILTER_YEARS_AGO_PREFERENCE}yo")
        if df_closest.empty:
            # Cross street. Any year.
            df_closest = self.get_closest_p_df(p_xy, dict_xy_fp,
                                               cross_st_substring_filter=street_filter,
                                               cache_file_label="cross-any")
        if df_closest.empty:
            log.warning(
                f"!! ATTENTION: haven't found any closest points among nearby points: {dict_xy_fp},"
                f" point: {p_xy}, address_filter: {address_filter}."
                f" Consider sending a different request, but first check the closest point filters!")
            return None, None, None, None
        else:
            total_closest_found = df_closest.shape[0]
            df_closest = df_closest.iloc[:limit_closest]
            mean_count = df_closest['Traffic1'].mean()
            most_frequent_count_year = df_closest['Cnt1year'].mode().max()
            return mean_count, most_frequent_count_year, df_closest.shape[0], total_closest_found

    def get_closest_p_df(self, p_xy, dict_f_p,
                         limit: Optional[int] = None,
                         filter_street_name_tag: Optional[str] = None,
                         street_substring_filter: Optional[str] = None,
                         cross_st_substring_filter: Optional[str] = None,
                         years_ago_filter: Optional[int] = None,
                         cache_file_label: Optional[str] = None):
        """ Return the closest points to p_xy among the points from dict_f_p. """
        dict_closest = EsriTrafficCountAnalyzer.get_closest_points_dict(p_xy, dict_f_p, limit,
                                                                        filter_street_name_tag,
                                                                        street_substring_filter,
                                                                        cross_st_substring_filter,
                                                                        years_ago_filter)

        # Save as a DataFrame to CSV file. (In the future this CSV can be used for debugging and caching).
        df_closest = EsriTrafficCountAnalyzer.dict_points_to_df(dict_closest)

        cache_file_label = None  # TODO put into param
        if cache_file_label is not None:
            output_path = self.__cache_dir + "/processed_input_points/x{}_y{}/{}.csv".format(*p_xy, {cache_file_label})
            EsriTrafficCountAnalyzer.save_df_to_csv(df_closest, output_path)

        return df_closest

    @staticmethod
    def get_closest_points_dict(p_xy: tuple, dict_f_p: dict, limit: Optional[int] = None,
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
                # log.info(f"(Using raw street name \"{closest_p_street_name_tag}\" for point {closest_xy})")

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

            num_xy_processed += 1
            if num_xy_processed >= limit:
                break

        return dict_closest

    @staticmethod
    def dict_points_to_df(dict_points):
        df_points = pd.DataFrame.from_dict(dict_points, orient='index')
        if not df_points.empty:
            df_points.index = df_points.index.map(lambda xy: EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(*xy))
            df_points.index.names = ['lat', 'lon']
        return df_points

    @staticmethod
    def closest_point(point, points):
        closest_index = distance.cdist([point], points).argmin()
        return points[closest_index]

    @staticmethod
    def fp_data_from_file(file_path):
        # TODO use Path
        features = []
        with open(file_path, 'r') as file:
            data = json.load(file)
            features = data['features']
        return features

    @staticmethod
    def transform_f_points_to_dict(f_ps: dict) -> dict:
        if f_ps is None:
            return {}

        points_dict = {}
        for f_p in f_ps:
            points_dict[(f_p['geometry']['x'], f_p['geometry']['y'])] = f_p['attributes']
        return points_dict

    @staticmethod
    def convert_xy_to_lat_lon(x, y):
        transformer = Transformer.from_crs('epsg:3857', 'epsg:4326')
        return transformer.transform(x, y)

    @staticmethod
    def convert_lat_lon_to_xy(lat, lon):
        transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
        return transformer.transform(lat, lon)

    def get_points_dict(self, input_lat_lot: tuple, address_filter: Optional[str] = None):
        suffix = '_unfiltered' if address_filter is None else ''
        cache_path = self.__cache_dir + f"/responses/lat{input_lat_lot[0]}_lon{input_lat_lot[1]}_{self.__box_size_m}m_response{suffix}.json"

        feature_points = EsriTrafficCountAnalyzer.fp_data_from_file(cache_path)
        if not feature_points:
            return None

        return self.transform_f_points_to_dict(feature_points)

    def min_bb_size_by_point(self, input_lat_lot: tuple, address_filter: Optional[str] = None,
                             num_points_to_include: Optional[int] = None):
        input_xy = EsriTrafficCountAnalyzer.convert_lat_lon_to_xy(*input_lat_lot)

        points_dict = self.get_points_dict(input_lat_lot, address_filter)
        dict_closest = EsriTrafficCountAnalyzer.get_closest_points_dict(input_xy, points_dict, num_points_to_include)

        d = EsriTrafficCountAnalyzer.min_bb_size_by_dict(input_xy, dict_closest)
        return d

    @staticmethod
    def min_bb_size_by_dict(p_xy, dict_xy):
        x = p_xy[0]
        min_x = min(dict_xy, key=lambda xy: xy[0])[0]
        max_x = max(dict_xy, key=lambda xy: xy[0])[0]
        min_x_d = x - min_x
        max_x_d = max_x - x
        final_x_d = max(min_x_d, max_x_d)

        y = p_xy[1]
        min_y = min(dict_xy, key=lambda xy: xy[1])[1]
        max_y = max(dict_xy, key=lambda xy: xy[1])[1]
        min_y_d = y - min_y
        max_y_d = max_y - y
        final_y_d = max(min_y_d, max_y_d)

        rightmost_virtual = EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(x + final_x_d, y)
        topmost_virtual = EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(x, y + final_y_d)
        input_lat_lon = EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(*p_xy)

        d_lat = topmost_virtual[0] - input_lat_lon[0]
        d_lon = rightmost_virtual[1] - input_lat_lon[1]

        d_lat_meters = EsriTrafficCountAnalyzer.d_lat_to_meters(d_lat)
        d_lon_meters = EsriTrafficCountAnalyzer.d_lon_to_meters(d_lon, input_lat_lon[0])

        d = max(d_lat_meters, d_lon_meters)
        return d

    @staticmethod
    def d_lat_to_meters(d_lat):
        # Decimal degrees -> Radians.
        d_lat = d_lat * math.pi / 180

        # Earth’s radius, sphere.
        r = 6378137

        d = round(d_lat * r)
        return d

    @staticmethod
    def d_lon_to_meters(d_lon, lat_nearby):
        # Decimal degrees -> Radians.
        d_lon = d_lon * math.pi / 180

        # Earth’s radius, sphere.
        r = 6378137

        d = round(d_lon * (r * math.cos(math.pi * lat_nearby / 180)))
        return d

    def __cached_request(self, input_lat_lot: tuple, address_filter: Optional[str] = None) -> dict:
        suffix = '_unfiltered' if address_filter is None else ''
        cache_path = self.__cache_dir + f"/responses/lat{input_lat_lot[0]}_lon{input_lat_lot[1]}_{self.__box_size_m}m_response{suffix}.json"

        fp_data = []
        if Path(cache_path).is_file():
            if address_filter is None:
                log.info(f"Found unfiltered response data for point {input_lat_lot}")
            fp_data = self.fp_data_from_file(cache_path)
        elif address_filter is not None or self.__resend_if_missing_cache_unfiltered:
            json_response = None
            while json_response is None:
                try:
                    if address_filter is None:
                        street_filter = None
                    else:
                        street_filter = usaddress.tag(address_filter)[0]['StreetName']

                    bounding_box = self.__esri_client.get_bounding_box(*input_lat_lot, self.__box_size_m)
                    log.info(f"bounding_box: {bounding_box}")

                    log.info("Sending request...")
                    json_response = self.__esri_client.get_traffic_counts_by_bounding_box(*bounding_box, street_filter)
                except:
                    # TODO tries limit
                    log.warning(
                        f"Failed request for point {input_lat_lot}, box size: {self.__box_size_m}m. Trying again...")
                    time.sleep(1)
            EsriTrafficCountAnalyzer.save_json_to_file(json_response, cache_path)
            fp_data = json_response['features']

        return self.transform_f_points_to_dict(fp_data)

    def analyze(self, p_lat_lon: tuple, address_filter: Optional[str] = None):

        p_xy = self.convert_lat_lon_to_xy(*p_lat_lon)

        # Get data with given street filter.
        dict_closest_xy = self.__cached_request(p_lat_lon, address_filter)
        analysis_result = self.analyze_closest(p_xy, dict_closest_xy, address_filter)
        mean_count, most_frequent_year, num_closest_used, total_closest_found = analysis_result

        if mean_count is None and address_filter is not None:
            # Try getting the data without the address filtering.
            dict_closest_xy = self.__cached_request(p_lat_lon, address_filter=None)
            analysis_result = self.analyze_closest(p_xy, dict_closest_xy, address_filter)
            mean_count, most_frequent_year, num_closest_used, total_closest_found = analysis_result

        # TODO make sure bounding box is relevant - i.e. same street / unfiltered
        # minimal_bounding_box_size = self.min_bb_size_by_point(p_lat_lon, response_cache_json_pa,
        #                                                       limit_same_street)
        return mean_count, most_frequent_year, num_closest_used, total_closest_found

    def analyze_by_api(self, lat, lon, input_address: str = None):
        input_lat_lot = (float(lat), float(lon))
        log.info(f"Analyzing point: {input_lat_lot}, box size: {self.__box_size_m}m, address filter: {input_address}")

        analysis_result = self.analyze(input_lat_lot, input_address)
        mean_count, most_frequent_count_year, num_closest_used, total_closest_found = analysis_result

        return mean_count, most_frequent_count_year, num_closest_used, total_closest_found

    def analyze_by_csv(self, input_csv_path, num_closest_range,
                       resend_request_if_no_unfiltered_data: bool = False):
        input_csv_df = pd.read_csv(input_csv_path)

        for limit_num_closest_to_use in num_closest_range:
            self.append_traffic_counts_to_df(input_csv_df, limit_num_closest_to_use, limit_num_closest_to_use,
                                             resend_request_if_no_unfiltered_data)

        input_csv_df.to_csv(os.path.splitext(input_csv_path)[0] + '_with_counts.csv')

    # TODO use or remove
    # def analyze_csv_minimal_bounding_box(self, input_csv_path, num_closest_range,
    #                    resend_request_if_no_unfiltered_data: bool = False):
    #     input_csv_df = pd.read_csv(input_csv_path)
    #
    #     for limit_num_closest_to_use in num_closest_range:
    #         self.append_traffic_counts_to_df(input_csv_df, limit_num_closest_to_use, limit_num_closest_to_use,
    #                                          resend_request_if_no_unfiltered_data)
    #
    #     input_csv_df.to_csv(os.path.splitext(input_csv_path)[0] + '_with_counts.csv')

    def append_traffic_counts_to_df(self, input_csv_df, limit_num_closest_to_use, limit_num_closest_unfiltered,
                                    resend_request_if_no_unfiltered_data: bool = False):
        traffic_count_list, traffic_year_list, traffic_num_closest_used, traffic_total_closest, minimal_box_sizes = [], [], [], [], []

        for row_num, row in input_csv_df.iterrows():
            log.info(f"----------------- row_num: {row_num} ----------------------")
            input_lat_lot = (row['Latitude'], row['Longitude'])
            input_address = row['address']

            analysis_result = self.analyze(input_lat_lot, input_address,
                                           resend_request_if_no_unfiltered_data, limit_num_closest_to_use,
                                           limit_num_closest_unfiltered)
            (mean_count, most_frequent_count_year, num_closest_used, total_closest_found,
             minimal_bounding_box_size) = analysis_result

            traffic_count_list.append(mean_count)
            traffic_year_list.append(most_frequent_count_year)
            traffic_num_closest_used.append(num_closest_used)
            traffic_total_closest.append(total_closest_found)
            minimal_box_sizes.append(minimal_bounding_box_size)

        num_rows = input_csv_df.shape[0]
        traffic_count_list += [None] * (num_rows - len(traffic_count_list))
        traffic_year_list += [None] * (num_rows - len(traffic_year_list))
        traffic_num_closest_used += [None] * (num_rows - len(traffic_num_closest_used))
        traffic_total_closest += [None] * (num_rows - len(traffic_total_closest))
        minimal_box_sizes += [None] * (num_rows - len(minimal_box_sizes))
        input_csv_df[f'avg_{limit_num_closest_to_use}_traffic_count'] = traffic_count_list
        input_csv_df[f'avg_{limit_num_closest_to_use}_traffic_year'] = traffic_year_list
        input_csv_df[f'avg_{limit_num_closest_to_use}_traffic_num_closest_used'] = traffic_num_closest_used
        input_csv_df[f'avg_{limit_num_closest_to_use}_traffic_total_closest'] = traffic_total_closest
        input_csv_df[f'{limit_num_closest_to_use}_p_minimal_box_size'] = minimal_box_sizes

    def save_point_data_to_csv(self, lat, lon):
        points_dict = self.get_points_dict((lat, lon))
        points_df = self.dict_points_to_df(points_dict)
        audit_path = f"audit/lat{lat}_lon{lon}.csv"
        self.save_df_to_csv(points_df, audit_path)
        log.info(f"Audit data saved to {audit_path}")

    def test_request_and_bb(self):
        lat, lon = 39.08474, -108.59502
        box_size = 3000
        print(self.__esri_client.get_bounding_box(lat, lon, box_size))
        self.save_point_data_to_csv(lat, lon)

    @staticmethod
    def rename_cached_responses_by_csv(helper_csv_path, cache_dir):
        helper_df = pd.read_csv(helper_csv_path)

        cache_path_obj = Path(cache_dir)

        for json_path_obj in cache_path_obj.iterdir():
            json_name = json_path_obj.name
            prefix, base_name = json_name.split('_', 1)
            row_i = int(re.findall(r'\d+', prefix)[0])
            lat = helper_df.at[row_i, 'Latitude']
            lon = helper_df.at[row_i, 'Longitude']
            new_name = f"lat{lat}_lon{lon}_{base_name}"
            dir_path = json_path_obj.parents[0]
            json_path_obj.rename(dir_path / new_name)


if __name__ == '__main__':
    pass
    # analyzer = EsriTrafficCountAnalyzer(
    #     output_cache_dir_path=r"output/api")  # output_cache_dir_path without '/' on the edges

    # lat, lon = 33.551310, -84.255990
    # traffic_count, most_frequent_count_year, num_closest_used, total_closest_found = analyzer.analyze_by_api(lat, lon)
    # print(traffic_count, most_frequent_count_year, num_closest_used, total_closest_found)
    # print("done")
