import numpy as np
import pylas
import os

def read_las(las_file):
    with pylas.open(las_file) as fh:
        print('Points from Header:', fh.header.point_count)

        
        las = fh.read()

        point_format = las.point_format
        print('las file', point_format)
        print('las file',list(point_format.dimension_names))
        #print(las)
        print('Points from data:', len(las.points))
        print('Points dimension:', len(las.points[0]))

        print('Classifications:', np.unique(las.classification, return_counts=True))
        ground_pts = las.classification == 2
        bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
        print('Ground Point Return Number distribution:')
        for r,c in zip(bins,counts):
            print('    {}:{}'.format(r,c))

        print('X count: ', len(las.X))
        

input_folder = "/home/geoseg/Work/blender_modeling/output_velodyne/"

indx = 5
dir_list_las_files = os.listdir(input_folder)
for las_file in dir_list_las_files:
    if ".las" in las_file:
        las_file_path = os.path.join(input_folder, las_file)
        read_las(las_file_path)
        indx-=1
        if indx==0:
            break