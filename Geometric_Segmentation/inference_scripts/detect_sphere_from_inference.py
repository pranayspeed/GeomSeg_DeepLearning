import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

import open3d as o3d


from data import data_loaders
from model import RandLANet

import argparse

## To be used for sphere fitting
#import pyransac3d as pyrsc

import open3d

PLANE = [0, 125, 125]
CONE = [0, 0, 255]
CYLINDER = [0, 255, 0]
SPHERE = [255, 0, 0]

OTHER = [0,0,0]#[255, 255, 255]
def visualize_test_case_cuda(scan_data_cuda, cls_cuda):
    color_map = {
    -1:OTHER,
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



color_map = {
    
    0: PLANE,
    1: SPHERE,
    2: CYLINDER,
    3: CONE,
    4: OTHER,
    -1: OTHER,
    }


def terminate_window(vis):
    vis.destroy_window()
    exit(0)


next_inference=False
def next_inf_fn(vis):
    global next_inference
    next_inference=True

pcd_geoms_old = None


#define 	GLFW_KEY_N   78
#define 	GLFW_KEY_ENTER 257
#define     GLFW_KEY_ESCAPE 256
def run_inference(loader, model, device):
    global next_inference
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    escape_key=256
    enter_key = 257
    vis.register_key_callback(escape_key, terminate_window)
    vis.register_key_callback(enter_key, next_inf_fn)
    #vis.register_key_callback(ord())
    #vis = o3d.visualization.Visualizer()

    vis.create_window()


    #ctr = vis.get_view_control()

    #ctr.set_zoom(50.2)

    pcd_old = None
    pcd_old_gt =None
    fps = 0

    next_inference= True


    SHAPE_INDEX = 3  # Cone is 3, need to figure out how to fit minimax cone

    for points, labels in loader:
        #pcd = o3d.io.read_point_cloud(os.path.join(path, pcd))
        #pcd.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
        while not next_inference:
            vis.poll_events()
            vis.update_renderer()
        with torch.no_grad():
            scan_points = points.view(-1,3).detach().numpy()
            #print(scan_points.T.shape)
            #scan_data = np.asarray(points.cpu().numpy())
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(scan_points)
            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(scan_points)

            #np.savetxt('scan_points.txt', scan_points)
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
            color_values = color_values[cls==SHAPE_INDEX]
            #pcd.colors = open3d.utility.Vector3dVector(color_values)
            
            scan_points_spheres = scan_points[cls==SHAPE_INDEX]
            pcd.points = open3d.utility.Vector3dVector(scan_points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

            rot_mat = [[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]]
            pcd.rotate(np.array(rot_mat), center=(0, 0, 0))            
            #fit_shapes(scan_points_spheres)
            pcd_geoms = fit_cones(scan_points_spheres)
            color_values = np.array([color_map[x] for x in actual_cls])
            
            pcd_gt.colors = open3d.utility.Vector3dVector(color_values)


            if pcd_old:
                vis.remove_geometry(pcd_old)
                #vis.remove_geometry(pcd_old_gt)
                for geom in pcd_geoms_old:
                    vis.remove_geometry(geom)
            vis.add_geometry(pcd)
            #vis.add_geometry(pcd_gt)
            for geom in pcd_geoms:
                vis.add_geometry(geom)

            #time.sleep(0.25)
            vis.poll_events()
            
            vis.update_renderer()
            pcd_old = pcd
            pcd_old_gt = pcd_gt
            pcd_geoms_old = pcd_geoms
            #vis.remove_geometry(pcd)
        next_inference=False
    vis.destroy_window()



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

        #all_geom.append(cone_mesh)
        all_geom.append(apex_sphere)
    return all_geom

def fit_cones(points):
    cones_params, cones_points = detect_cones(points, 0.009)

    pcd_geoms = cone_pcd_geom(cones_params) 

    return pcd_geoms   

# def fit_shapes(points):
#     sph1 = pyrsc.Sphere()

#     np.savetxt('cone_points.txt', points)
#     center, radius, inliers = sph1.fit(points, thresh=0.4)
#     print('Sphere: ', center, radius, inliers)

def main():


    

    device = torch.device('cpu') #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Loading data...')

    #dataset = Path('/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/datasets/new_street/processed')
    #dataset = Path('/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/datasets/models_ten_v2/processed')
    dataset = Path('/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/datasets/Old_not_used_dataset/registration_test/processed')

    print(dataset)
    #dataset_sampling = 'active_learning' #'geomseg'
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

    #Best model
    #model.load_state_dict(torch.load('runs/models_ten_v2_BS_6_CLS_5_3276/checkpoint_30.pth', map_location=device)['model_state_dict']) 
    model.load_state_dict(torch.load('runs/models_ten_v2_BS_6_CLS_5_3276_v2_epoch_100/checkpoint_100.pth', map_location=device)['model_state_dict'])


    model.eval()


    print(device)
    run_inference(loader, model, device)




if __name__ == '__main__':
    main()