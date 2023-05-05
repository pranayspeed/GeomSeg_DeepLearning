import time
# import pandas as pd
import pandas as pd

from save_keypoints import get_options, KeyPointExtractors
import argparse

import open3d as o3d
import numpy as np
import copy

import os

import matplotlib.pyplot as plt


def getUnixTime():
    return int(time.time())

class PoseGraphResultSaver:
    def __init__(self, init_pose, save_gap, num_frames, seq_idx, save_dir):
        self.pose_list = np.reshape(init_pose, (-1, 16))
        self.gt_pose_list = np.reshape(init_pose, (-1, 16))
        self.save_gap = save_gap
        self.num_frames = num_frames

        self.seq_idx = seq_idx
        self.save_dir = save_dir
        
        self.errors = []

    def saveUnoptimizedPoseGraphResult(self, cur_pose, curr_gt_pose, cur_node_idx):
        # save 
        self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 16))))
        
        self.gt_pose_list= np.vstack((self.gt_pose_list, np.reshape(curr_gt_pose, (-1, 16))))

    def save_kitti_poses(self, output_dir, file_name):
        
        est_poses = self.pose_list[:,:12]

        np.savetxt(os.path.join(output_dir, file_name+"_est.txt"),est_poses, fmt="%6f")
        gt_poses = self.gt_pose_list[:,:12]

        np.savetxt(os.path.join(output_dir, file_name+"_gt.txt"),gt_poses, fmt="%6f")        


    def compute_ate(self):
        curr_pred = np.array( [self.pose_list[-1][3], self.pose_list[-1][7], self.pose_list[-1][11]])
                        
        curr_gt = np.array( [self.gt_pose_list[-1][3], self.gt_pose_list[-1][7], self.gt_pose_list[-1][11]])
        ate = self.compute_ATE(curr_gt, curr_pred)
        rte = self.compute_RTE(curr_gt, curr_pred)
        
        return ate
    
                    
    def vizCurrentTrajectory(self, fig_idx, title, legends={}):
        x = self.pose_list[:,3]
        y = self.pose_list[:,7]
        z = self.pose_list[:,11]

        #print("pred" , x[-1],y[-1],z[-1])
        # if "nosample" in title:
        #     curr_pred = np.array([-self.pose_list[-1][7], -self.pose_list[-1][11], self.pose_list[-1][3]])
        # else:
        #     curr_pred = np.array( [self.pose_list[-1][3], self.pose_list[-1][7], self.pose_list[-1][11]])
        curr_pred = np.array( [self.pose_list[-1][3], self.pose_list[-1][7], self.pose_list[-1][11]])

                        
        curr_gt = np.array( [self.gt_pose_list[-1][3], self.gt_pose_list[-1][7], self.gt_pose_list[-1][11]])
        ate = self.compute_ATE(curr_gt, curr_pred)
        rte = self.compute_RTE(curr_gt, curr_pred)
        # Create empty plot with blank marker containing the extra label


        fig = plt.figure(fig_idx)
        
        plt.clf()
        if "kitti" in title and "nosample" in title:
            plt.plot(-y, x, color='blue')
        elif "synth" in title:
            plt.plot(x,y, color='blue')
        else:
            plt.plot(x,z, color='blue')


        #plt.plot(x,z, color='blue') # kitti camera coord for clarity
        #-y, x
        plt.axis('equal')
        plt.xlabel('x', labelpad=10)
        plt.ylabel('y', labelpad=10)
        
        #print(title)
        #print(self.gt_pose_list[-1])
        
        if 'kitti' in title or 'oxford' in title or 'luco' in title:
            x = self.gt_pose_list[:,3]
            y = self.gt_pose_list[:,7]
            z = self.gt_pose_list[:,11] 
            plt.plot(x,z, color='green')    
            #plt.plot(-y,-x, color='green')
            #print("gt" ,x[-1],y[-1],z[-1])
        else:
            x = self.gt_pose_list[:,3]
            y = self.gt_pose_list[:,7]
            z = self.gt_pose_list[:,11] 
            plt.plot(x,y, color='green')    
            #plt.plot(-y,-x, color='green')
            #print("gt" ,x[-1],y[-1],z[-1])            
        
        legends ["ATE"] = ate
        #legends ["RTE"] = rte
        
        for legnd in legends:
            plt.plot([], [], ' ', label=legnd +" : "+str(float(f'{legends[legnd]:.4f}')))
        
        plt.legend()
        plt.title(title)
        
        plt.draw()
        plt.pause(0.01) #is necessary for the plot to update for some reason

        #return ate, rte

    def compute_ATE(self, cur_gt, cur_pred):


        gt_xyz = cur_gt #[:3, 3] 

        pred_xyz = cur_pred #[:3, 3]

        align_err = gt_xyz - pred_xyz

        # print('i: ', i)
        # print("gt: ", gt_xyz)
        # print("pred: ", pred_xyz)
        # input("debug")
        self.errors.append(np.sqrt(np.sum(align_err ** 2)))
        
        ate = np.sqrt(np.mean(np.asarray(self.errors) ** 2)) 
        return ate
    def compute_RTE(self, cur_gt, cur_pred):


        gt_xyz = cur_gt #[:3, 3] 

        pred_xyz = cur_pred #[:3, 3]

        align_err = gt_xyz - pred_xyz

        # print('i: ', i)
        # print("gt: ", gt_xyz)
        # print("pred: ", pred_xyz)
        # input("debug")
        error = np.sqrt(np.sum(align_err ** 2))
        return error
   
    
    
    

def get_repeatiblity_options():
    # params
    parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

    parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time

    parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
    parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
    parser.add_argument('--num_candidates', type=int, default=10) # must be int
    parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

    parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

    parser.add_argument('--data_base_dir', type=str, 
                        default='/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/sequences')

    parser.add_argument('--ground_truth_dir', type=str, 
                        default='/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/poses')
    parser.add_argument('--sequence_idx', type=str, default='00')

    parser.add_argument('--save_gap', type=int, default=300)

    parser.add_argument('--mesh_path', type=str, 
                        default='mesh.ply')

    parser.add_argument('--mesh_map', action='store_true')

    parser.add_argument('--show_map', action='store_true')

    parser.add_argument('--show_vis', action='store_true')
    
    parser.add_argument('--method',  type=str, 
                        default="usip")

    parser.add_argument('--sequence',  type=int, 
                        default=0)

    #parser.add_argument('--dataset',  type=str, 
    #                    default="kitti")


    opt_detector = get_options(parser)
    #args = opt_detector# .opt# parser.parse_args()
    return opt_detector


from scipy.spatial.transform import Rotation as R

import random

# The rotation seems to be too large in magnitude, resulting is  bad detection
def get_random_transform():
    point_cloud = o3d.geometry.PointCloud()
    euler_angles = R.random().as_euler('zxy', degrees=False) #random radian rotation


    list_Size = 3
    # random float from 1 to 99.9
    integer_list = random.sample(range(1, 20), list_Size)
    float_list = [x/10 for x in integer_list]
              
    T = np.eye(4)
    T[:3, :3] = point_cloud.get_rotation_matrix_from_xyz(euler_angles)
    T[0, 3] = float_list[0]
    T[1, 3] = float_list[1]
    T[2, 3] = float_list[2]
    return T

def get_delta_translation_transform():
    point_cloud = o3d.geometry.PointCloud()
    euler_angles = R.random().as_euler('zxy', degrees=False) #random radian rotation


    list_Size = 3
    # random float from 1 to 99.9
    integer_list = random.sample(range(1, 20), list_Size)
    float_list = [x/100 for x in integer_list]
              
    T = np.eye(4)
    #T[:3, :3] = point_cloud.get_rotation_matrix_from_xyz(euler_angles)
    T[0, 3] = float_list[0]
    T[1, 3] = float_list[1]
    T[2, 3] = float_list[2]
    return T

def get_delta_rotation_transform():
    point_cloud = o3d.geometry.PointCloud()
    
    one_degree = 0.0174533
    
    euler_angles = R.random().as_euler('xyz', degrees=False) #random radian rotation

    
    list_Size = 3
    # random float from 1 to 99.9
    integer_list = random.sample(range(1, 5), list_Size)
    float_list = [x*one_degree for x in integer_list]
    
    
    euler_angles = float_list
              
    T = np.eye(4)
    T[:3, :3] = point_cloud.get_rotation_matrix_from_xyz(euler_angles)
    # T[0, 3] = float_list[0]
    # T[1, 3] = float_list[1]
    # T[2, 3] = float_list[2]
    return T

def generate_k_random_normalized(delta, k=3, low=0.05):
    if delta ==0:
        return np.zeros(k)
    a = np.random.rand(k)
    a = (a/a.sum()*(delta-low*k))
    weights = a+low

    # checking that the sum is 1
    assert np.isclose(weights.sum(), delta)
    return weights

def get_delta_transform(delta_trans=1.0, delta_rot=1.0):
    point_cloud = o3d.geometry.PointCloud()
    
    one_degree = 0.0174533
    
    euler_angles = R.random().as_euler('xyz', degrees=False) #random radian rotation


    euler_angles = generate_k_random_normalized(delta_rot, 3,0.05)*one_degree
    

    # random float from 1 to 99.9
    #integer_list = random.sample(range(1, 20), list_Size)
    float_list = generate_k_random_normalized(delta_trans, 3, 0.05)
            
              
    T = np.eye(4)
    T[:3, :3] = point_cloud.get_rotation_matrix_from_xyz(euler_angles)
    T[0, 3] = float_list[0]
    T[1, 3] = float_list[1]
    T[2, 3] = float_list[2]
    return T



def get_transformed_point_cloud(point_cloud, transform):
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(curr_pts) 

    point_cloud_transformed = copy.deepcopy(point_cloud).transform(transform)
    
    return point_cloud , point_cloud_transformed
    

def extract_keypoints_ISS(input_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector) :
    tic = time.time()
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(input_pcd)
    toc = 1000 * (time.time() - tic)
    #print("ISS Computation took {:.0f} [ms]".format(toc))     
    
    return keypoints, toc 


import PCLKeypoint

def extract_keypoints_SIFT(input_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector) :
    tic = time.time()
    # method = 'sift'
    min_scale = 0.5
    n_octaves = 4
    n_scales_per_octave = 8
    min_contrast = 0.1    
    keypoints = PCLKeypoint.keypointSift(input_pcd.points,
                                                             min_scale,
                                                             n_octaves,
                                                             n_scales_per_octave,
                                                             min_contrast)  # Mx3
    
    toc = 1000 * (time.time() - tic)
    #print("SIFT Computation took {:.0f} [ms]".format(toc), " count :", len(keypoints))     
    point_cloud_kps = o3d.geometry.PointCloud()
    point_cloud_kps.points = o3d.utility.Vector3dVector(keypoints) 
    
        
    return point_cloud_kps, toc     


def extract_keypoints_Harris(input_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector) :
    tic = time.time()
    # method = 'harris'
    radius = 1
    nms_threshold = 0.001
    threads = 0   
    keypoints = PCLKeypoint.keypointHarris3D(input_pcd.points,
                                                               radius,
                                                               nms_threshold,
                                                               threads,
                                                               True)  # Mx3
    
    toc = 1000 * (time.time() - tic)
    #print("Harris Computation took {:.0f} [ms]".format(toc), " count :", len(keypoints))     
    point_cloud_kps = o3d.geometry.PointCloud()
    point_cloud_kps.points = o3d.utility.Vector3dVector(keypoints) 
    
        
    return point_cloud_kps, toc 
    
def compute_inverse_transform_pcd(input_point_cloud, transfrm):
    inv_transfm = np.linalg.inv(transfrm)
    point_cloud_transformed = copy.deepcopy(input_point_cloud).transform(inv_transfm)
    return point_cloud_transformed, inv_transfm


from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

def compute_repeatablity(pc1, pc2, search_rad=0.5):
    matches = get_matches_pcd(pc1, pc2, search_rad)
    matches = np.unique(matches)
    if len(pc1.points) ==0:
        return 1.0
    #print("Matches : ", len(matches), "/", len(pc1.points))
    repeatiblity_res = len(matches)/ len(pc1.points)
    #print("Repeatabilty: ", repeatiblity_res)
    return repeatiblity_res, len(pc1.points), len(matches)

def get_matches_pcd(pc1, pc2, search_rad=0.5):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pc2.points)
    distances, indices = neigh.kneighbors(pc1.points, return_distance=True)
    matches = indices.flatten()  
    distances = distances.flatten()
    final_matches = [matches[i] for i in range(len(matches)) if distances[i]<search_rad]  
    return final_matches  

def print_matches_pcd(pc1, pc2, search_rad=0.0001):

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pc2.points)
    distances, indices = neigh.kneighbors(pc1.points, return_distance=True)
    matches = indices.flatten()  
    distances = distances.flatten()
    final_matches = [matches[i] for i in range(len(matches)) if distances[i]<search_rad]  
    print(final_matches)
    
    
###################################################################################    
    
    
    
    
from models.keypoint_detector import ModelDetector
import torch

from evaluation.kitti_test_loader import FarthestSampler

def model_state_dict_parallel_convert(state_dict, mode):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict


def model_state_dict_convert_auto(state_dict, gpu_ids):
    for k, v in state_dict.items():
        if (k[0:7] == 'module.' and len(gpu_ids) >= 2) or (k[0:7] != 'module.' and len(gpu_ids) == 1):
            return state_dict
        elif k[0:7] == 'module.' and len(gpu_ids) == 1:
            return model_state_dict_parallel_convert(state_dict, mode='to_single')
        elif k[0:7] != 'module.' and len(gpu_ids) >= 2:
            return model_state_dict_parallel_convert(state_dict, mode='to_parallel')
        else:
            raise Exception('Error in model_state_dict_convert_auto')


def nms(keypoints_np, sigmas_np, NMS_radius):
    '''

    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        return keypoints_np, sigmas_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sigmas_np = np.zeros(sigmas_np.shape, dtype=sigmas_np.dtype)

    while keypoints_np.shape[0] > 0:
        # print(sigmas_np.shape)
        # print(sigmas_np)

        min_idx = np.argmin(sigmas_np, axis=0)
        # print(min_idx)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[min_idx, :]
        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]

        # increase counter
        valid_keypoint_counter += 1

    return valid_keypoints_np[0:valid_keypoint_counter, :], \
           valid_sigmas_np[0:valid_keypoint_counter]

def extract_keypoints_usip(input_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector) :

    detector_model_path = '/home/pranayspeed/Work/git_repos/TSF_datasets/oxford-checkpoints/oxford/16384-256-k1k32-2d/best.pth'
    # build detector
    model_detector = ModelDetector(opt_detector)
    model_detector.detector.load_state_dict(
        model_state_dict_convert_auto(
            torch.load(
                detector_model_path,
                map_location='cpu'), opt_detector.gpu_ids))
    model_detector.freeze_model()      


    pc_np = np.asarray(input_pcd.points)
    normals_np = np.asarray(input_pcd.normals)
    
    anc_sn_new = anc_sn.detach().numpy().transpose()
    
    sn_np = np.append(normals_np, anc_sn_new[:,3],1)

    desired_keypoint_num = 128
    NMS_radius = 0
    noise_sigma = 0
    downsample_rate = 1    

    
    # get nodes, perform random sampling to reduce computation cost
    node_np = farthest_sampler.sample(
        pc_np[np.random.choice(pc_np.shape[0], int(opt_detector.input_pc_num / 4), replace=False)],
        opt_detector.node_num)


    # convert to torch tensor
    anc_pc1 = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
    anc_sn1 = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
    anc_node1 = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM


    # anc_pc_cuda = anc_pc.to(opt_detector.device)
    # anc_sn_cuda = anc_sn.to(opt_detector.device)
    # anc_node_cuda = anc_node.to(opt_detector.device)

    anc_pc_cuda = torch.unsqueeze(anc_pc1, dim=0).to(opt_detector.device)
    anc_sn_cuda = torch.unsqueeze(anc_sn1, dim=0).to(opt_detector.device)
    anc_node_cuda = torch.unsqueeze(anc_node1, dim=0).to(opt_detector.device)


    #print(anc_pc_cuda.size(), anc_sn_cuda.size(), anc_node_cuda.size())
    tic = time.time()
    anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
    anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
    anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
        
    
    toc = 1000 * (time.time() - tic)

        
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3

        frame_keypoint_np = anc_keypoints_np[b]
        frame_sigma_np = anc_sigmas_np[b]

        # nms
        keypoints, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

    
    
    #print("SIFT Computation took {:.0f} [ms]".format(toc), " count :", len(keypoints))     
    point_cloud_kps = o3d.geometry.PointCloud()
    point_cloud_kps.points = o3d.utility.Vector3dVector(keypoints) 
    
        
    return point_cloud_kps, toc  



def extract_keypoints_usip_odom(input_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector) :
    NMS_radius=0
    detector_model_path = '/home/pranayspeed/Work/git_repos/TSF_datasets/oxford-checkpoints/oxford/16384-256-k1k32-2d/best.pth'
    # build detector
    model_detector = ModelDetector(opt_detector)
    model_detector.detector.load_state_dict(
        model_state_dict_convert_auto(
            torch.load(
                detector_model_path,
                map_location='cpu'), opt_detector.gpu_ids))
    model_detector.freeze_model()      


    anc_pc_cuda = anc_pc.to(opt_detector.device)
    anc_sn_cuda = anc_sn.to(opt_detector.device)
    anc_node_cuda = anc_node.to(opt_detector.device)

    #print(anc_pc_cuda.size(), anc_sn_cuda.size(), anc_node_cuda.size())
    tic = time.time()
    anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
    anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
    anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
        
    
    toc = 1000 * (time.time() - tic)

        
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3

        frame_keypoint_np = anc_keypoints_np[b]
        frame_sigma_np = anc_sigmas_np[b]

        # nms
        keypoints, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

    
    
    #print("SIFT Computation took {:.0f} [ms]".format(toc), " count :", len(keypoints))     
    point_cloud_kps = o3d.geometry.PointCloud()
    point_cloud_kps.points = o3d.utility.Vector3dVector(keypoints) 
    
        
    return point_cloud_kps, toc 



##############################################################################


from evaluation.kitti_test_loader import  KittiTestLoader_Odometry, KittiTestLoader_Odometry_Original
import torch

def get_kitti_sample(opt_detector, start_index=0):
    numpy_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_numpy'    
    
    test_txt_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_bin'

    testset = KittiTestLoader_Odometry(test_txt_folder, numpy_folder, opt_detector)

    # test_txt_folder = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/sequences'
    # testset = KittiTestLoader_Odometry_Original(test_txt_folder, numpy_folder, opt_detector)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    
    i, data = next(enumerate(testloader, start_index))
    anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
        
        frame_normals_np = np.transpose(anc_sn[b].detach().numpy())  # Nx4
        
        pcd.normals = o3d.utility.Vector3dVector(frame_normals_np[:,:3]) 
        #print("seq frame id : ", i)
        return pcd , (anc_pc, anc_sn, anc_node, gt_pose)


def get_kitti_sample_itr(opt_detector, start_index=0):
    numpy_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_numpy'    
    
    test_txt_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_bin'

    testset = KittiTestLoader_Odometry(test_txt_folder, numpy_folder, opt_detector)

    # test_txt_folder = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/sequences'
    # testset = KittiTestLoader_Odometry_Original(test_txt_folder, numpy_folder, opt_detector)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    
    return enumerate(testloader, start_index)
    anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
        
        frame_normals_np = np.transpose(anc_sn[b].detach().numpy())  # Nx4
        
        pcd.normals = o3d.utility.Vector3dVector(frame_normals_np[:,:3]) 
        #print("seq frame id : ", i)
        return pcd , (anc_pc, anc_sn, anc_node, gt_pose)


from evaluation.oxford_test_loader import OxfordTestLoader   
def get_oxford_sample(opt_detector, start_index=0):
    root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/oxford'
    # testset = OxfordDataLoader_Odometry(root_folder, self.opt_detector)
    
    testset = OxfordTestLoader(root_folder, opt_detector)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    
    i, data = next(enumerate(testloader, start_index))
    anc_pc, anc_sn, anc_node, seq_id= data
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
        # print("seq frame id : ", i)
        # return pcd

        frame_normals_np = np.transpose(anc_sn[b].detach().numpy())  # Nx4
        
        pcd.normals = o3d.utility.Vector3dVector(frame_normals_np[:,:3]) 
        #print("seq frame id : ", i)
        gt_pose = np.eye(4) # Don't have ground truth for Oxford samples for now
        return pcd , (anc_pc, anc_sn, anc_node, gt_pose)
    

from evaluation.kitti_test_loader import SyntheticDataLoader   
def get_synth_sample(opt_detector, start_index=0):
    root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_data'
    testset = SyntheticDataLoader(root_folder, opt_detector)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    
    i, data = next(enumerate(testloader, start_index))
    anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
        # print("seq frame id : ", i)
        # return pcd

        frame_normals_np = np.transpose(anc_sn[b].detach().numpy())  # Nx4
        
        pcd.normals = o3d.utility.Vector3dVector(frame_normals_np[:,:3]) 
        #print("seq frame id : ", i)
        return pcd , (anc_pc, anc_sn, anc_node, gt_pose)




from evaluation.kitti_test_loader import ParisLucoDataLoader   
def get_luco_sample(opt_detector, start_index=0):
    root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_data'
    testset = ParisLucoDataLoader(root_folder, opt_detector)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    
    i, data = next(enumerate(testloader, start_index))
    anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
    for b in range(anc_pc.size(0)):
        frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
        # print("seq frame id : ", i)
        # return pcd

        frame_normals_np = np.transpose(anc_sn[b].detach().numpy())  # Nx4
        
        pcd.normals = o3d.utility.Vector3dVector(frame_normals_np[:,:3]) 
        #print("seq frame id : ", i)
        return pcd , (anc_pc, anc_sn, anc_node, gt_pose)



##################################################################################



def compute_transform_invariant_repeatiblity(input_orig_pcd, keypoint_extraction_fn, transf=None, target_pcd=None, search_radius=0.5):
    
    input_orig_pcd_np, (anc_pc, anc_sn, anc_node, gt_pose)=input_orig_pcd
    
    test_transf = transf
    if test_transf is None:
        test_transf = get_delta_transform()
    

        
    orig_pcd , pcd_transf = get_transformed_point_cloud(input_orig_pcd_np, test_transf)
    
    farthest_sampler = FarthestSampler()
    
    orig_pcd_kps, orig_extract_time = keypoint_extraction_fn(orig_pcd, anc_pc, anc_sn, anc_node, farthest_sampler)
    
    if target_pcd is not None:
        pcd_transf, (anc_pc, anc_sn, anc_node)=target_pcd
    
    pcd_transf_kps, pcd_transf_time = keypoint_extraction_fn(pcd_transf, anc_pc, anc_sn, anc_node, farthest_sampler) 
    
    if  len(orig_pcd_kps.points) ==0 or len(pcd_transf_kps.points)==0:
        print("Keypoint Extraction failed")
        return 0.0
    pcd_transf_kps_inv, _ = compute_inverse_transform_pcd(pcd_transf_kps, test_transf)
    
    
    
    repeatiblity_val, points_cnt, matches = compute_repeatablity(orig_pcd_kps, pcd_transf_kps_inv, search_radius)
    return repeatiblity_val, orig_pcd, orig_pcd_kps, pcd_transf_kps_inv, points_cnt, matches
      
      
#############################################################################

# from walle.core import RotationMatrix


# def gen_data(N=100, frac=0.1):
#   # create a random rigid transform
#   transform = np.eye(4)
#   transform[:3, :3] = RotationMatrix.random()
#   transform[:3, 3] = 2 * np.random.randn(3) + 1

#   # create a random source point cloud
#   src_pc = 5 * np.random.randn(N, 3) + 2
#   dst_pc = Procrustes.transform_xyz(src_pc, transform)

#   # corrupt
#   rand_corrupt = np.random.choice(np.arange(len(src_pc)), replace=False, size=int(frac*N))
#   dst_pc[rand_corrupt] += np.random.uniform(-10, 10, (int(frac*N), 3))

#   return src_pc, dst_pc, transform, rand_corrupt


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def compute_noise_invariant_repeatiblity(input_orig_pcd, keypoint_extraction_fn, noise_mag=(0.0, 0.01), target_pcd=None, search_radius=0.5):
    
    input_orig_pcd_np, (anc_pc, anc_sn, anc_node, gt_pose)=input_orig_pcd
    
    
    mu, sigma = noise_mag # 0, 0.1  # mean and standard deviation
    pcd_transf = apply_noise(input_orig_pcd_np, mu, sigma)

    orig_pcd = copy.deepcopy(input_orig_pcd_np)    
    #orig_pcd , pcd_transf = get_transformed_point_cloud(input_orig_pcd_np, test_transf)
    
    farthest_sampler = FarthestSampler()
    
    orig_pcd_kps, orig_extract_time = keypoint_extraction_fn(orig_pcd, anc_pc, anc_sn, anc_node, farthest_sampler)
    
    if target_pcd is not None:
        pcd_transf, (anc_pc, anc_sn, anc_node)=target_pcd
    
    pcd_transf_kps, pcd_transf_time = keypoint_extraction_fn(pcd_transf, anc_pc, anc_sn, anc_node, farthest_sampler) 
    
    if  len(orig_pcd_kps.points) ==0 or len(pcd_transf_kps.points)==0:
        print("Keypoint Extraction failed")
        return 0.0
    #pcd_transf_kps_inv, _ = compute_inverse_transform_pcd(pcd_transf_kps, test_transf)
    
    pcd_transf_kps_inv = copy.deepcopy(pcd_transf_kps) 
    
    repeatiblity_val, points_cnt, matches = compute_repeatablity(orig_pcd_kps, pcd_transf_kps_inv, search_radius)
    return repeatiblity_val, orig_pcd, orig_pcd_kps, pcd_transf_kps_inv, points_cnt, matches


##########################################################################

def apply_downsampling(pcd, voxel_size):
    noisy_pcd = copy.deepcopy(pcd)
    noisy_pcd.voxel_down_sample(voxel_size)
    return noisy_pcd

def compute_resolution_invariant_repeatiblity(input_orig_pcd, keypoint_extraction_fn, voxel_size=0.5, target_pcd=None, search_radius=0.5):
    
    input_orig_pcd_np, (anc_pc, anc_sn, anc_node, gt_pose)=input_orig_pcd
    
    pcd_transf = apply_downsampling(input_orig_pcd_np, voxel_size)

    orig_pcd = copy.deepcopy(input_orig_pcd_np)    
    #orig_pcd , pcd_transf = get_transformed_point_cloud(input_orig_pcd_np, test_transf)
    
    farthest_sampler = FarthestSampler()
    
    orig_pcd_kps, orig_extract_time = keypoint_extraction_fn(orig_pcd, anc_pc, anc_sn, anc_node, farthest_sampler)
    
    if target_pcd is not None:
        pcd_transf, (anc_pc, anc_sn, anc_node)=target_pcd
    
    pcd_transf_kps, pcd_transf_time = keypoint_extraction_fn(pcd_transf, anc_pc, anc_sn, anc_node, farthest_sampler) 
    
    if  len(orig_pcd_kps.points) ==0 or len(pcd_transf_kps.points)==0:
        print("Keypoint Extraction failed")
        return 0.0
    #pcd_transf_kps_inv, _ = compute_inverse_transform_pcd(pcd_transf_kps, test_transf)
    
    pcd_transf_kps_inv = copy.deepcopy(pcd_transf_kps) 
    
    repeatiblity_val, points_cnt, matches = compute_repeatablity(orig_pcd_kps, pcd_transf_kps_inv, search_radius)
    return repeatiblity_val, orig_pcd, orig_pcd_kps, pcd_transf_kps_inv, points_cnt, matches



def get_datasets_and_keypoints():
    datasets = {
    "kitti": get_kitti_sample,
    "oxford": get_oxford_sample,
    "luco": get_luco_sample
    }


    keypoint_detectors = {
    "Harris": extract_keypoints_Harris,
    "ISS": extract_keypoints_ISS,
    "SIFT": extract_keypoints_SIFT,
    "USIP": extract_keypoints_usip
    }
    return datasets, keypoint_detectors



###############################################################

def run_transformation_invariant_repeatiblity(keypoint_detectors, datasets, opt_detector):
    results = {}
    for keypoint_detect in keypoint_detectors:
        curr_detector_fn = keypoint_detectors[keypoint_detect]
        
        for dataset in datasets:
            repeat_list = []
            matches_list =[]
            points_count_list =[]
            curr_dataset_fn = datasets[dataset]
            tst_transf = get_delta_transform() # Generate new random transform
            for i in range(20):
                rep_val, orig_pcd_data, orig_kps_data, transf_kps_data, points_count, matches = compute_transform_invariant_repeatiblity(curr_dataset_fn(opt_detector, i),curr_detector_fn, tst_transf)
                repeat_list.append(rep_val)
                #print(points_count, matches)
                matches_list.append(matches)
                points_count_list.append(points_count)
            repeatablity_val = np.array(repeat_list).mean()
            avg_keypoints_detected = np.array(points_count_list).mean()
            avg_keypoints_matched = np.array(matches_list).mean()
            print(keypoint_detect, dataset, repeatablity_val, avg_keypoints_detected, avg_keypoints_matched)
            if keypoint_detect not in results:
                results[keypoint_detect] = {}
            if dataset not in results[keypoint_detect]:
                results[keypoint_detect][dataset]={}
                
            results[keypoint_detect][dataset]["repeatability"]= repeatablity_val
            results[keypoint_detect][dataset]["matched"]= avg_keypoints_matched
            results[keypoint_detect][dataset]["detected"]= avg_keypoints_detected
    return results   
        
###################################################################################


def run_noise_invariant_repeatiblity(keypoint_detectors, datasets, opt_detector, sigma_list=[0.000001,0.00001, 0.0001, 0.001, 0.01, 0.1]):
    results = {}
    for keypoint_detect in keypoint_detectors:
        curr_detector_fn = keypoint_detectors[keypoint_detect]
        
        for dataset in datasets:
            #sigma_list = [0.000001,0.00001, 0.0001, 0.001, 0.01, 0.1]
            
            repeat_list = {}
            matches_list = {}
            points_count_list = {}
            for sigma in sigma_list:
                repeat_list[sigma]=[]
                matches_list[sigma]=[]
                points_count_list[sigma]=[]
                
            curr_dataset_fn = datasets[dataset]
            
            for i in range(20):
                #mu, sigma =  0.0, 0.00001 
                mu=0.0 
                curr_data = curr_dataset_fn(opt_detector, i)
                for sigma in sigma_list:
                    rep_val, orig_pcd_data, orig_kps_data, transf_kps_data, points_count, matches = compute_noise_invariant_repeatiblity(curr_data,curr_detector_fn, (mu, sigma))
                    repeat_list[sigma].append(rep_val)
                    matches_list[sigma].append(matches)
                    points_count_list[sigma].append(points_count)

            if keypoint_detect not in results:
                results[keypoint_detect] = {}
            if dataset not in results[keypoint_detect]:
                results[keypoint_detect][dataset]={}
                
            for sigma in sigma_list:
                results[keypoint_detect][dataset][sigma] = {}
                
                repeatablity_val = np.array(repeat_list[sigma]).mean()
                avg_keypoints_detected = np.array(points_count_list[sigma]).mean()
                avg_keypoints_matched = np.array(matches_list[sigma]).mean()
                print(keypoint_detect, dataset, sigma, "->",repeatablity_val, avg_keypoints_detected, avg_keypoints_matched)

                
                results[keypoint_detect][dataset][sigma]["repeatability"]= repeatablity_val
                results[keypoint_detect][dataset][sigma]["matched"]= avg_keypoints_matched
                results[keypoint_detect][dataset][sigma]["detected"]= avg_keypoints_detected
    return results   



###################################################################

def run_resolution_invariant_repeatiblity(keypoint_detectors, datasets, opt_detector, voxel_sizes = [0.05, 0.1, 0.5, 1.0, 2.0]):
    results = {}
    for keypoint_detect in keypoint_detectors:
        curr_detector_fn = keypoint_detectors[keypoint_detect]
        
        for dataset in datasets:
            #voxel_sizes = [0.05, 0.1, 0.5, 1.0, 2.0]
            
            
            repeat_list = {}
            matches_list = {}
            points_count_list = {}
            for voxel_size in voxel_sizes:
                repeat_list[voxel_size]=[]
                matches_list[voxel_size]=[]
                points_count_list[voxel_size]=[]
                
            curr_dataset_fn = datasets[dataset]
            
            for i in range(20):
                
                curr_data = curr_dataset_fn(opt_detector, i)
                for voxel_size in voxel_sizes:
                    rep_val, orig_pcd_data, orig_kps_data, transf_kps_data, points_count, matches = compute_resolution_invariant_repeatiblity(curr_data,curr_detector_fn, voxel_size)
                    repeat_list[voxel_size].append(rep_val)
                    matches_list[voxel_size].append(matches)
                    points_count_list[voxel_size].append(points_count)

            if keypoint_detect not in results:
                results[keypoint_detect] = {}
            if dataset not in results[keypoint_detect]:
                results[keypoint_detect][dataset]={}
                
            for voxel_size in voxel_sizes:
                results[keypoint_detect][dataset][voxel_size] = {}
                
                repeatablity_val = np.array(repeat_list[voxel_size]).mean()
                avg_keypoints_detected = np.array(points_count_list[voxel_size]).mean()
                avg_keypoints_matched = np.array(matches_list[voxel_size]).mean()
                print(keypoint_detect, dataset, voxel_size, "->",repeatablity_val, avg_keypoints_detected, avg_keypoints_matched)

                
                results[keypoint_detect][dataset][voxel_size]["repeatability"]= repeatablity_val
                results[keypoint_detect][dataset][voxel_size]["matched"]= avg_keypoints_matched
                results[keypoint_detect][dataset][voxel_size]["detected"]= avg_keypoints_detected
    return results  



################################################

import utils.ICP as ICP
def compute_transform(source_kps, traget_kps, initial_transf):
    min_len = min(len(traget_kps), len(source_kps))
    #odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
    #odom_transform = ICP_register_global(traget_kps[:min_len], source_kps[:min_len],initial_transf)
    #odom_transform, _, _ = ICP.icp_ransac(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
    odom_transform, _, _ = ICP.icp_robust(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)


    return odom_transform