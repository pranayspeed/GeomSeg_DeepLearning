from GeometrySegmentation import get_all_test_cases, get_test_case_data

import numpy as np
import open3d
import os

from tqdm import tqdm

def main():
    data_path = "../data/TRAIN-20s/"

    new_data_path = "../data/TRAIN-20s-normals/"
    test_cases = get_all_test_cases(data_path)

    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    with tqdm(range(len(test_cases)), desc="Converting dataset to normals") as pbar:

        for idx in pbar:
            test_data = get_test_case_data(test_cases[idx][0], test_cases[idx][1], test_cases[idx][2])
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(test_data['data'])
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
            normals = np.asarray(pcd.normals)
            
            full_path = new_data_path + f"{test_cases[idx][1]:03d}" + "/" +f"{test_cases[idx][2]:05d}" +".npz"
            if not os.path.exists(new_data_path + f"{test_cases[idx][1]:03d}" + "/"):
                os.makedirs(new_data_path + f"{test_cases[idx][1]:03d}" + "/")

            np.savez(full_path, data=test_data['data'], cls=test_data['cls'], normals=normals, ins=test_data['ins'], scan=test_data['scan'])


if __name__ == "__main__":
    main()