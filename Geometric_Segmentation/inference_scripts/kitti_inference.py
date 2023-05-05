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

import torch


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




PLANE = [0, 125, 125]
CONE = [0, 0, 255]
CYLINDER = [0, 255, 0]
SPHERE = [255, 0, 0]

OTHER = [0,0,0]#[255, 255, 255]

color_map = {
    
    0: PLANE,
    1: SPHERE,
    2: CYLINDER,
    3: CONE,
    4: OTHER,
    -1: OTHER,
    }




# for raw_frame, timestamp, gt_pose in dataset_itr(args.data_root, args.dataset_name, args.sequence):
#     print(raw_frame, timestamp, gt_pose) 

vis = None
title = "Visualizer"
final_poses = np.eye(4).flatten()

device = torch.device('cpu')




from model import RandLANet

print('Loading model...')

#print(next(iter(loader)))
#d_in = next(iter(loader))[0].size(-1)

#print(d_in)
#exit(0)
d_in = 3 #6
num_classes = 5 #14
model = RandLANet(d_in, num_classes, 16, 4, device)
#Best model

model.load_state_dict(torch.load('runs/models_ten_v2_BS_6_CLS_5_3276_v2_epoch_100/checkpoint_100.pth', map_location=device)['model_state_dict'])


model.eval()



def terminate_window(vis):
    vis.destroy_window()
    exit(0)


next_inference=False
def next_inf_fn(vis):
    global next_inference
    next_inference=True


pcd_1 = None
pcd_2 = None
next_inference= True

vis = open3d.visualization.VisualizerWithKeyCallback()
escape_key=256
enter_key = 257
vis.register_key_callback(escape_key, terminate_window)
vis.register_key_callback(enter_key, next_inf_fn)
#vis.register_key_callback(ord())
#vis = o3d.visualization.Visualizer()

vis.create_window()





from geometry_fitting.cone_fitting import detect_cones, get_sphere_mesh, get_cone_mesh
import math

def cone_pcd_geom(cones):
    all_geom = []
    for i in range(len(cones)):
        cone_axis_new, cone_center_new, half_angle, error = cones[i]


        print("Error: ", error, "\nHalf angle: ", math.degrees(half_angle), "\nCone axis: ", cone_axis_new, "\nCone center: ", cone_center_new)
        ls1 = open3d.geometry.LineSet()
        ls1.points = open3d.utility.Vector3dVector([cone_center_new, cone_center_new-cone_axis_new])
        ls1.lines = open3d.utility.Vector2iVector([[0, 1]])
        ls1.colors = open3d.utility.Vector3dVector([[1, 0, 0]])


        apex_sphere = get_sphere_mesh(0.1, cone_center_new)
        apex_sphere.paint_uniform_color([0, 0, 1])
        cone_mesh = get_cone_mesh(half_angle, cone_axis_new, cone_center_new)

        #all_geom.append(ls1)

        all_geom.append(cone_mesh)
        all_geom.append(apex_sphere)
    return all_geom

pcd_geoms_old = None
for data in dataset_itr(args.data_root, args.dataset_name, args.sequence):
    #print(data['raw_frame'].shape, data['timestamp'], data['gt_pose'])
    while not next_inference:
        vis.poll_events()
        vis.update_renderer()
    with torch.no_grad():
        scan = data['raw_frame']

        cones_params, cones_points = detect_cones(scan, 0.0009)

        pcd_geoms = cone_pcd_geom(cones_params)



        points = torch.from_numpy(scan).unsqueeze(0).float()
        #scan = data['raw_frame'].transpose()  
        pcd = open3d.geometry.PointCloud()
        scan_points = points.view(-1,3).detach().numpy()
        pcd.points = open3d.utility.Vector3dVector(scan_points)

        print(scan.shape, points.size())

        points = points.to(device)  

        scores = model(points)
        predictions = torch.max(scores, dim=-2).indices

        predictions = predictions.cpu().numpy()
        cls = np.array(predictions.flatten())#.cpu().numpy()
        #print(cls.shape, scan_data.shape)
        color_values = np.array([color_map[x] for x in cls])

        pcd.colors = open3d.utility.Vector3dVector(color_values)
    #     scan = np.dot(data['gt_pose'], np.vstack([scan, np.ones((1, scan.shape[1]))]))
        if pcd_1:
            vis.remove_geometry(pcd_1)
            vis.remove_geometry(pcd_geoms_old)
        vis.add_geometry(pcd)
        for geom in pcd_geoms:
            vis.add_geometry(geom)
        #vis.add_geometry(pcd_geoms)
        pcd_1 = pcd
        pcd_geoms_old = pcd_geoms
        
    next_inference=False
vis.destroy_window()