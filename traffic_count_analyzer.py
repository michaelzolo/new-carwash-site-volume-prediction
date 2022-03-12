from numpy import random
from scipy.spatial import distance
import json


class EsriTrafficCountAnalyzer:
    @staticmethod
    def closest_point_index(point, points):
        closest_index = distance.cdist([point], points).argmin()
        return closest_index

    @staticmethod
    def get_feature_points_from_file(file_path):
        features = []
        with open(file_path, 'r') as file:
            data = json.load(file)
            features = data['features']
        return features

    @staticmethod
    def transform_feature_points_to_points_dict(feature_points):
        points_dict = {}
        for fp in feature_points:
            points_dict[(fp['geometry']['x'], fp['geometry']['y'])] = fp['attributes']
        return points_dict


if __name__ == '__main__':
    analyzer = EsriTrafficCountAnalyzer()
    response_file_path = r"C:\Users\Michael\OneDrive - Fine ALGs 2019 Ltd\omniX\Results and runs\Traffic count\Boren sent first excel to add column\row0\d_5000m_response.json"
    feature_points = analyzer.get_feature_points_from_file(response_file_path)
    points_dict = analyzer.transform_feature_points_to_points_dict(feature_points)
    points_only = list(points_dict.keys())  # TODO !! check order - to insure correct index usage!

    row0_point = (-8207479.18948, 4963098.1353)  # 40.6653, -73.72904


    # points_only.remove(row0_point)

    closest_i = analyzer.closest_point_index(row0_point, points_only)
    closest_to_row0 = points_only[closest_i]
    print(closest_to_row0)
    print(points_dict[closest_to_row0])

    # Testing "analyzer.closest_point_index()" :: seems to work fine.
    # part_of_points = points_only[0:3]
    # for p in part_of_points:
    #     points_without_p = [x for x in part_of_points if x != p]
    #     closest_i = analyzer.closest_point_index(p, points_without_p)
    #     print(f"the closest point to {p} is {points_without_p[closest_i]}")
