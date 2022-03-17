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


class EsriTrafficCountAnalyzer:
    def __init__(self, input_csv_path, limit_num_closest_same_street, limit_num_closest_unfiltered):
        self.__input_csv_path = input_csv_path
        self.__limit_num_closest_same_street = limit_num_closest_same_street
        self.__limit_num_closest_unfiltered = limit_num_closest_unfiltered
        self.__output_dir_path = r"output"

    def tag_input_csv(self):
        input_df = self.get_input_csv_df()
        input_df['address_tags'] = input_df['address'].apply(lambda a: usaddress.tag(str(a)))
        input_df.to_csv(os.path.splitext(self.__input_csv_path)[0] + '_tagged.csv')

    def get_input_csv_df(self):
        return pd.read_csv(self.__input_csv_path)

    def compute_mean_count_from_closest_points(self, f_p_file_path, input_xy, input_address,
                                               limit_num_closest_to_use: int = None):
        if limit_num_closest_to_use is None:
            limit_num_closest_to_use = self.__limit_num_closest_same_street
        feature_points = EsriTrafficCountAnalyzer.get_feature_points_from_file(f_p_file_path)

        points_dict = EsriTrafficCountAnalyzer.transform_feature_points_to_points_dict(feature_points)

        input_address_tags = usaddress.tag(input_address)[0]
        print(f"Analyzing input point {input_xy} on street {input_address_tags['StreetName']}.")
        input_street_tags = dict(filter(lambda item: 'street' in item[0].lower(), input_address_tags.items()))
        input_street = input_street_tags['StreetName']

        filter_street_substring = input_street_tags['StreetName']

        # Same street. Up to 5 years old. (The best alternative)
        output_path = self.__output_dir_path + "/x{} y{} {}/same-5yo.csv".format(*input_xy, input_street)
        df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                    filter_street_substring=filter_street_substring,
                                                                    filter_years_ago=5,
                                                                    output_path=output_path)
        if df_closest.empty:
            # Same street. Any year.
            output_path = self.__output_dir_path + "/x{} y{} {}/same-any.csv".format(*input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_street_substring=filter_street_substring,
                                                                        output_path=output_path)
        if df_closest.empty:
            # Cross street. Up to 5 years old.
            output_path = self.__output_dir_path + "/x{} y{} {}/cross-5yo.csv".format(*input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_cross_st_substring=filter_street_substring,
                                                                        filter_years_ago=5,
                                                                        output_path=output_path)
        if df_closest.empty:
            # Cross street. Any year.
            output_path = self.__output_dir_path + "/x{} y{} {}/cross-any.csv".format(*input_xy, input_street)
            df_closest = EsriTrafficCountAnalyzer.get_closest_points_df(input_xy, points_dict,
                                                                        filter_cross_st_substring=filter_street_substring,
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
            # closest_to_mean = \  # TODO REMOVE. Since this is inaccurate. We need the dominant, not the closest in value.
            #     df_closest.iloc[(df_closest['Traffic1'] - mean_count).abs().argsort()[:1]]['Cnt1year'].values[0]
            return mean_count, most_frequent_count_year, df_closest.shape[0], total_closest_found

    @staticmethod
    def get_closest_points_df(p_xy, dict_f_p, output_path, limit: Optional[int] = None,
                              filter_street_name_tag: Optional[str] = None,
                              filter_street_substring: Optional[str] = None,
                              filter_cross_st_substring: Optional[str] = None,
                              filter_years_ago: Optional[int] = None):
        """ Return the closest feature points :dict_f_p """
        # TODO instead of this - use smarter filtering. For now all street filters are disabled.
        filter_street_name_tag = None
        filter_street_substring = None
        filter_cross_st_substring = None

        list_xy = list(dict_f_p.keys())

        dict_closest = {}

        num_xy_processed = 0
        if limit is None:
            limit = len(list_xy)

        while len(list_xy) > 0:
            # TODO !! check order -> to insure correct index usage -> and then optimize access with index access.

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

            # TODO filter only tags containing 'street' + make it more generic, make 'filter_tags' a list
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

    # Testing "analyzer.closest_point_index()" :: seems to work fine.
    # part_of_points = points_only[0:3]
    # for p in part_of_points:
    #     points_without_p = [x for x in part_of_points if x != p]
    #     closest_i = analyzer.closest_point_index(p, points_without_p)
    #     print(f"the closest point to {p} is {points_without_p[closest_i]}")

    @staticmethod
    def convert_xy_to_lat_lon(x, y):
        transformer = Transformer.from_crs('epsg:3857', 'epsg:4326')
        return transformer.transform(x, y)

    @staticmethod
    def convert_lat_lon_to_xy(lon, lat):
        transformer = Transformer.from_crs('epsg:4326', 'epsg:3857')
        return transformer.transform(lon, lat)

    def run(self, re_run_on_zero_same_street_closest: bool = False):
        input_csv_df = self.get_input_csv_df()

        esri_client = EsriClient()

        traffic_count_list, traffic_year_list, traffic_num_closest_used, traffic_total_closest = [], [], [], []
        # limit_last_input_row = 0  # TODO re-process lines list in config
        for row_num, row in input_csv_df.iterrows():
            print(f"----------------- row_num: {row_num} ----------------------")  # TODO logging
            # print(f"street only: {usaddress.tag(row['address'])[0]['StreetName']}")

            box_size = 3000

            response_json_path = self.__output_dir_path + f"/responses/row{row_num}_{box_size}m_response.json"

            bounding_box = esri_client.get_bounding_box(row['Latitude'], row['Longitude'], box_size)
            print(f"bounding_box: {bounding_box}")
            if not Path(response_json_path).is_file():
                json_response = None
                while json_response is None:
                    try:
                        json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box,
                                                                                       usaddress.tag(row['address'])[0][
                                                                                           'StreetName'])
                    except:
                        print("Failed request. trying again ...")
                        time.sleep(1)
                with open(response_json_path, 'w') as f:
                    json.dump(json_response, f)

            mean_count, most_frequent_count_year, num_closest_used, total_closest_found = analyzer.compute_mean_count_from_closest_points(
                response_json_path,
                analyzer.convert_lat_lon_to_xy(row['Latitude'], row['Longitude']),  # 28.402451, -81.243217
                row['address']
            )

            if mean_count is None:
                # Trying to find data without the street filter.
                response_json_path = Path(os.path.splitext(response_json_path)[0] + '_unfiltered.json')

                # First - look at previously saved responses. Otherwise - send a new request.
                if response_json_path.exists():
                    print(f"* Found unfiltered response data for point {(row['Latitude'], row['Longitude'])}")
                    mean_count, most_frequent_count_year, num_closest_used, total_closest_found = analyzer.compute_mean_count_from_closest_points(
                        response_json_path,
                        analyzer.convert_lat_lon_to_xy(row['Latitude'], row['Longitude']),  # 28.402451, -81.243217
                        row['address'],
                        limit_num_closest_to_use=5
                        # TODO make sure to use smart averaging, instead of the fixed parameter
                    )
                elif mean_count is None and re_run_on_zero_same_street_closest:
                    print(f"* Sending a new request for point {(row['Latitude'], row['Longitude'])}")
                    json_response = esri_client.get_traffic_counts_by_bounding_box(*bounding_box)

                    with open(response_json_path, 'w') as f:
                        json.dump(json_response, f)

                    mean_count, most_frequent_count_year, num_closest_used, total_closest_found = analyzer.compute_mean_count_from_closest_points(
                        response_json_path,
                        analyzer.convert_lat_lon_to_xy(row['Latitude'], row['Longitude']),  # 28.402451, -81.243217
                        row['address'],
                        limit_num_closest_to_use=5
                        # TODO make sure to use smart averaging, instead of the fixed parameter
                    )

            traffic_count_list.append(mean_count)
            traffic_year_list.append(most_frequent_count_year)
            traffic_num_closest_used.append(num_closest_used)
            traffic_total_closest.append(total_closest_found)

            # if limit_last_input_row <= 0:
            #     break
            # limit_last_input_row -= 1

        num_rows = input_csv_df.shape[0]
        traffic_count_list += [None] * (num_rows - len(traffic_count_list))
        traffic_year_list += [None] * (num_rows - len(traffic_year_list))
        traffic_num_closest_used += [None] * (num_rows - len(traffic_num_closest_used))
        traffic_total_closest += [None] * (num_rows - len(traffic_total_closest))
        input_csv_df['traffic_count'] = traffic_count_list
        input_csv_df['traffic_year'] = traffic_year_list
        input_csv_df['traffic_num_closest_used'] = traffic_num_closest_used
        input_csv_df['traffic_total_closest'] = traffic_total_closest
        print(input_csv_df)
        input_csv_df.to_csv(os.path.splitext(input_csv_path)[0] + '_with_counts.csv')

    def sandbox_run(self):
        # print(usaddress.tag("135 Ripley St. Alpena, MI  49707"))
        # print(EsriTrafficCountAnalyzer.convert_xy_to_lat_lon(-9289851.9568,5629166.3554000039))

        input_csv_path = 'input2_with_counts.csv'
        analyzer = EsriTrafficCountAnalyzer(input_csv_path, limit_num_closest_same_street=3,
                                            limit_num_closest_unfiltered=5)
        input_csv_df = analyzer.get_input_csv_df()
        mode = input_csv_df['traffic_year'].mode().max()
        print(mode)


if __name__ == '__main__':
    # analyzer = EsriTrafficCountAnalyzer(input_csv_path='input2.csv',
    #                                     limit_num_closest_same_street=3,
    #                                     limit_num_closest_unfiltered=5)

    analyzer.run()
    # sandbox_run()
