import os
import slam_dataset_sdk
from slam_dataset_sdk.datasets_iterator import dataset_itr, get_supported_datasets

import argparse

# data_root = "/run/user/1000/gvfs/smb-share:server=10.84.164.159,share=datasets" #os.environ.get("DATASETS")
# dataset_name = "paris_luco"
# dataset_name = "kitti"
# sequence = 0


import open3d
import numpy as np

parser = argparse.ArgumentParser(
                    prog = 'Dataset loader',
                    description = 'print dataset frames')

#parser.add_argument('--dataset_name', type=str, default='paris_luco', help='dataset name')
parser.add_argument('--dataset_name', type=str, default='kitti', help='dataset name')
parser.add_argument('--sequence', type=int, default=0, help='sequence number')
#parser.add_argument('--data_root', type=str, default='/run/user/1000/gvfs/smb-share:server=10.84.162.67,share=datasets')
parser.add_argument('--data_root', type=str, default='/home/pranayspeed/Work/git_repo/robotcar-dataset-sdk/data/')
parser.add_argument('--list_dataset', type=bool, default=False, help='supported dataset names list')

dataset_root_path_map = {
"kitti": '/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/datasets',#/run/user/1000/gvfs/smb-share:server=10.84.162.67,share=datasets',
"paris_luco": '/run/user/1000/gvfs/smb-share:server=10.84.162.67,share=datasets',
"mydataset": '/home/pranayspeed/Work/git_repo/robotcar-dataset-sdk/data/',
"oxford": '/home/pranayspeed/Work/git_repo/datasets/oxford_radar_robotcar_dataset_sample_medium/2019-01-10-14-36-48-radar-oxford-10k-partial/',
}


args = parser.parse_args()


args.data_root = dataset_root_path_map[args.dataset_name]

spported_datasets = get_supported_datasets() 
if args.dataset_name not in spported_datasets:
    print("Error : Unsupported dataset name")
if args.list_dataset or args.dataset_name not in spported_datasets:
    print("Supported datasets:")
    print(spported_datasets)
    print("Usage: python3 paris_luco_test.py --dataset_name <supported_dataset_name> --sequence <seq_number> --data_root <dataset_root_dir>")
    print("output for dataset_itr: dict{raw_frame, timestamp, gt_pose}")
    exit(0)


# for raw_frame, timestamp, gt_pose in dataset_itr(args.data_root, args.dataset_name, args.sequence):
#     print(raw_frame, timestamp, gt_pose) 

vis = None
title = "Visualizer"
final_poses = np.eye(4).flatten()


# from geometry_fitting.cone_fitting import detect_cones, get_sphere_mesh, get_cone_mesh
# import math

# def cone_pcd_geom(cones):
#     all_geom = []
#     for i in range(len(cones)):
#         cone_axis_new, cone_center_new, half_angle, error = cones[i]


#         print("Error: ", error, "\nHalf angle: ", math.degrees(half_angle), "\nCone axis: ", cone_axis_new, "\nCone center: ", cone_center_new)
#         ls1 = open3d.geometry.LineSet()
#         ls1.points = open3d.utility.Vector3dVector([cone_center_new, cone_center_new-cone_axis_new])
#         ls1.lines = open3d.utility.Vector2iVector([[0, 1]])
#         ls1.colors = open3d.utility.Vector3dVector([[1, 0, 0]])


#         apex_sphere = get_sphere_mesh(0.1, cone_center_new)
#         apex_sphere.paint_uniform_color([0, 0, 1])
#         cone_mesh = get_cone_mesh(half_angle, cone_axis_new, cone_center_new)

#         #all_geom.append(ls1)

#         all_geom.append(cone_mesh)
#         all_geom.append(apex_sphere)
#     return all_geom

for data in dataset_itr(args.data_root, args.dataset_name, args.sequence):
    #print(data['raw_frame'].shape, data['timestamp'], data['gt_pose'])

    scan = data['raw_frame'].transpose()    
    scan = np.dot(data['gt_pose'], np.vstack([scan, np.ones((1, scan.shape[1]))]))
    
    ptcld= np.asarray(scan[:3, :].astype(np.float64).copy())

    ptcld = ptcld[[1, 0, 2]].transpose().astype(np.float64)
    
    if vis is None:
        vis = open3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
        render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate        


        # cones_params, cones_points = detect_cones(ptcld, 0.009)

        # pcd_geoms = cone_pcd_geom(cones_params)

        pcd = open3d.geometry.PointCloud()
        # initialise the geometry pre loop
        
        pcd.points = open3d.utility.Vector3dVector(ptcld)#.astype(np.float64))
        #pcd.points = open3d.utility.Vector3dVector(ptcld[:3].transpose().astype(np.float64))
        pcd.paint_uniform_color([0.9, 0.1, 0.1])
        #pcd.colors = open3d.utility.Vector3dVector(np.tile(ptcld[3:].transpose(), (1, 3)).astype(np.float64))
        # Rotate pointcloud to align displayed coordinate frame colouring
        #pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
        vis.add_geometry(pcd)
        # render_option = vis.get_render_option()
        # render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
        # render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
        coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        vis.add_geometry(coordinate_frame)
        # while True:
        #     vis.poll_events()
        #     vis.update_renderer() 
            

    #pcd.points = open3d.utility.Vector3dVector(ptcld[:3].transpose().astype(np.float64))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(ptcld)
    pcd.paint_uniform_color([0.9, 0.1, 0.1])
    vis.add_geometry(pcd)
    #pcd.colors = open3d.utility.Vector3dVector(
    #    np.tile(ptcld[3:].transpose(), (1, 3)).astype(np.float64) / 40)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()    

vis.run()