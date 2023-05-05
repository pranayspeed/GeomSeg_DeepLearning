

import numpy as np
from pathlib import Path
import time
print("before torch")
import torch

import torch.nn as nn

import open3d as o3d

import os

from data import data_loaders
from model import RandLANet
from utils.ply import read_ply, write_ply

t0 = time.time()


#path = Path('datasets') / 's3dis' / 'subsampled' / 'test'
#path = Path('/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_new/test')

device = torch.device('cpu') #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
#loader, _ = data_loaders(path)



#Namespace(dataset=PosixPath('/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_new'), epochs=50, load='', adam_lr=0.01, batch_size=1, decimation=4, dataset_sampling='geomseg', neighbors=16, scheduler_gamma=0.95, test_dir='test', train_dir='train', val_dir='val', logs_dir=PosixPath('runs'), gpu=device(type='cuda', index=0), name='2023-04-06_15:10', num_workers=0, save_freq=10)

dataset_base = Path(os.path.expandvars('$NAS_DRIVE'))/'lidar-research/Geometric_Seg_Pranay/datasets'

print(dataset_base)
dataset = dataset_base / 'Geometry_street/processed_new'
#dataset = Path('/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_4cls')
#dataset = Path('/home/geoseg/Work/git_repo/datasets/kitti_test/kitti/processed')
print(dataset)
dataset_sampling = 'geomseg'

_, loader= data_loaders(
    dataset,
    dataset_sampling,
    batch_size=1, #args.batch_size,
    num_workers=1 , #args.num_workers,
    pin_memory=True
)


print('Loading model...')

d_in = 3 #6
num_classes = 5 #14
model = RandLANet(d_in, num_classes, 16, 4, device)

#model.load_state_dict(torch.load('runs/2020-04-11_17:03/checkpoint_10.pth')['model_state_dict'])

#model.load_state_dict(torch.load('runs/V2/checkpoint_10.pth')['model_state_dict'])
model.load_state_dict(torch.load('runs/V3_BS_4/checkpoint_50.pth')['model_state_dict'])
#model.load_state_dict(torch.load('runs/V4_BS_2_CLS_5/checkpoint_20.pth')['model_state_dict'])


#Following model with 4 classes is totally overfitting for ground plane class
#num_classes = 4 #14
#model = RandLANet(d_in, num_classes, 16, 4, device)
#model.load_state_dict(torch.load('runs/processed_4cls_BS_4_CLS_4/checkpoint_10.pth')['model_state_dict']) 

model.eval()


import open3d

PLANE = [0, 125, 125]
CONE = [0, 0, 255]
CYLINDER = [0, 255, 0]
SPHERE = [255, 0, 0]

OTHER = [0,0,0]#[255, 255, 255]
def visualize_test_case_cuda(scan_data_cuda, cls_cuda):
    color_map = {
    
    0: PLANE,
    1: SPHERE,
    2: CYLINDER,
    3: CONE,
    4: OTHER,
    }
    scan_data = np.asarray(scan_data_cuda.cpu().numpy())
    cls = np.array(cls_cuda.flatten())#.cpu().numpy()
    print(cls.shape, scan_data.shape)
    color_values = np.array([color_map[x] for x in cls])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scan_data)
    pcd.colors = open3d.utility.Vector3dVector(color_values)
    open3d.visualization.draw_geometries([pcd])

#print(np.unique(predictions, return_counts=True))
#visualize_test_case_cuda(cloud, predictions)

#print('Done. Time elapsed: {:.1f}s'.format(t1-t0))



color_map = {
    
    0: PLANE,
    1: SPHERE,
    2: CYLINDER,
    3: CONE,
    4: OTHER,
    }


def terminate_window(vis):
    vis.destroy_window()
    exit(0)


print(device)

def run_inference(loader, model):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    escape_key=256
    vis.register_key_callback(escape_key, terminate_window)
    #vis.register_key_callback(ord())
    #vis = o3d.visualization.Visualizer()

    vis.create_window()


    #ctr = vis.get_view_control()

    #ctr.set_zoom(50.2)

    pcd_old = None
    pcd_old_gt =None
    fps = 0
    for points, labels in loader:
        #pcd = o3d.io.read_point_cloud(os.path.join(path, pcd))
        #pcd.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))

        with torch.no_grad():
            scan_points = points.view(-1,3).detach().numpy()
            #print(scan_points.T.shape)
            #scan_data = np.asarray(points.cpu().numpy())
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(scan_points)
            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(scan_points)
            #NEED to check why rotate
            #rot_mat = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            rot_mat = [[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]]
            pcd.rotate(np.array(rot_mat), center=(0, 0, 0))
            pcd_gt.rotate(np.array(rot_mat), center=(0, 0, 0))

            extents = pcd_gt.get_axis_aligned_bounding_box().get_extent()
            req_trans = np.max(extents)*1.3
            #print(extents, req_trans)

            pcd_gt.translate([req_trans,0,0])
            start_time = time.time()
            points = points.to(device)
            labels = labels.to(device)
            scores = model(points)
            predictions = torch.max(scores, dim=-2).indices
            accuracy = (predictions == labels).float().mean() # TODO: compute mIoU usw.
            print('Accuracy:', accuracy.item())
            predictions = predictions.cpu().numpy()

            time_elapsed = time.time()-start_time
            fps = (1/(time_elapsed+1e-10))*points.shape[0]
            print('Done. Time elapsed: {:.1f}s   , FPS: {}'.format(time_elapsed, fps))

            actual_cls = labels.view(-1,1).cpu().numpy().flatten()
            cls = np.array(predictions.flatten())#.cpu().numpy()
            #print(cls.shape, scan_data.shape)
            color_values = np.array([color_map[x] for x in cls])

            pcd.colors = open3d.utility.Vector3dVector(color_values)

            color_values = np.array([color_map[x] for x in actual_cls])
            pcd_gt.colors = open3d.utility.Vector3dVector(color_values)


            if pcd_old:
                vis.remove_geometry(pcd_old)
                vis.remove_geometry(pcd_old_gt)
            vis.add_geometry(pcd)
            vis.add_geometry(pcd_gt)
            #time.sleep(0.25)
            vis.poll_events()
            
            vis.update_renderer()
            pcd_old = pcd
            pcd_old_gt = pcd_gt
            #vis.remove_geometry(pcd)
    vis.destroy_window()


run_inference(loader, model)



def visualize_once(loader, model):
    for points, labels in loader:
        #pcd = o3d.io.read_point_cloud(os.path.join(path, pcd))
        #pcd.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))

        with torch.no_grad():
            scan_points = points.view(-1,3).detach().numpy()

            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(scan_points)
            actual_cls = labels.view(-1,1).cpu().numpy().flatten()

            color_values = np.array([color_map[x] for x in actual_cls])
            pcd_gt.colors = open3d.utility.Vector3dVector(color_values)              

            pcd_gt.rotate(np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]]), center=(0, 0, 0))

            o3d.visualization.draw_geometries([pcd_gt])

#visualize_once(loader, model)