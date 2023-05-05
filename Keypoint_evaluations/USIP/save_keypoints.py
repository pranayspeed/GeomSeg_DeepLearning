import matplotlib
matplotlib.use('TkAgg')
is_plot = False
is_timing = False

# ============= dataset ===============
dataset_type = 'kitti'
#test_txt_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti-reg-test'
test_txt_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_bin'
numpy_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_numpy'
output_folder = './keypoints_knn_ball/kitti'

# dataset_type = 'oxford'
# root_folder = '/ssd/jiaxin/TSF_datasets/oxford'
# output_folder = '/ssd/jiaxin/keypoints_lambda/oxford'

# dataset_type = 'redwood'
# npy_folder = '/ssd/jiaxin/TSF_datasets/redwood/numpy_gt_normal'
# output_folder = '/ssd/jiaxin/keypoints_lambda/redwood'

# dataset_type = '3dmatch_eval'
# npy_folder = '/ssd/jiaxin/TSF_datasets/3DMatch_eval_npy'
# output_folder = '/ssd/jiaxin/keypoints_noisy/3dmatch'

# dataset_type = 'modelnet'
# root = '/ssd/jiaxin/TSF_datasets/modelnet40-test_rotated_numpy'
# output_folder = '/ssd/jiaxin/keypoints_nosigma/modelnet'


gpu_id = 0

is_ensure_keypoint_num = False
desired_keypoint_num = 128
NMS_radius = 0
noise_sigma = 0
downsample_rate = 1
# =============== method ================
method = 'usip'
#detector_model_path = '/data/tsf/oxford/checkpoints/save/detector/BALL-16384-512-r2k64-k16/best.pth'
detector_model_path = '/home/pranayspeed/Work/git_repos/TSF_datasets/oxford-checkpoints/oxford/16384-256-k1k32-2d/best.pth'
# detector_model_path = '/data/tsf/scenenn/checkpoints/save/detector/5000-512-k1k32-full-3d/best.pth'
# detector_model_path = '/data/tsf/match3d/checkpoints/save/detector/10240-512-k1k32-full-lambda100/best.pth'
# detector_model_path = '/data/tsf/modelnet/checkpoints/save/no_sigma/lambda_1/5000-512k32-3d/best.pth'


#method = 'iss'
iss_salient_radius = 2
iss_non_max_radius = 2
iss_gamma_21 = 0.975
iss_gamma_32 = 0.975
iss_min_neighbors = 5
threads = 0

# method = 'harris'
radius = 1
nms_threshold = 0.001
threads = 0

# method = 'sift'
min_scale = 0.5
n_octaves = 4
n_scales_per_octave = 8
min_contrast = 0.1

# method = 'random'



import os
import time
import copy
import numpy as np
import math
import shutil

import torch
torch.cuda.empty_cache()

# if dataset_type == 'kitti':
#     import kitti.options_detector

#     opt_detector_instance = kitti.options_detector.Options()
#     opt_detector_instance.parse_without_process()
#     opt_detector_instance.opt.gpu_ids = [gpu_id]
#     opt_detector_instance.opt.batch_size = 1
#     opt_detector_instance.process_opts()
#     opt_detector = opt_detector_instance.opt
# elif dataset_type == 'oxford':
#     import oxford.options_detector

#     opt_detector_instance = oxford.options_detector.Options()
#     opt_detector_instance.parse_without_process()
#     opt_detector_instance.opt.gpu_ids = [gpu_id]
#     opt_detector_instance.opt.batch_size = 1
#     opt_detector_instance.process_opts()
#     opt_detector = opt_detector_instance.opt
# elif dataset_type == 'redwood':
#     import scenenn.options_detector

#     opt_detector_instance = scenenn.options_detector.Options()
#     opt_detector_instance.parse_without_process()
#     opt_detector_instance.opt.gpu_ids = [gpu_id]
#     opt_detector_instance.opt.batch_size = 1
#     opt_detector_instance.process_opts()
#     opt_detector = opt_detector_instance.opt
# elif dataset_type == '3dmatch_eval':
#     import match3d.options_detector

#     opt_detector_instance = match3d.options_detector.Options()
#     opt_detector_instance.parse_without_process()
#     opt_detector_instance.opt.gpu_ids = [gpu_id]
#     opt_detector_instance.opt.batch_size = 1
#     opt_detector_instance.process_opts()
#     opt_detector = opt_detector_instance.opt
# elif dataset_type == 'modelnet':
#     import modelnet.options_detector
#     opt_detector = modelnet.options_detector.Options().parse()
# else:
#     assert False

# opt_detector.input_pc_num = int(opt_detector.input_pc_num / downsample_rate)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import string

#from models.keypoint_detector import ModelDetector
#from models.keypoint_descriptor import ModelDescriptor
from util.visualizer import Visualizer
from util import vis_tools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from evaluation.kitti_test_loader import KittiTestLoader, KittiTestLoader_Odometry, KittiTestLoader_Odometry_Original, SyntheticDataLoader, OxfordDataLoader_Odometry, ParisLucoDataLoader
from evaluation.oxford_test_loader import OxfordTestLoader
from evaluation.redwood_loader import RedwoodLoader
from data.match3d_eval_loader import Match3DEvalLoader
from data.modelnet_rotated_loader import ModelNet_Rotated_Loader

#import PCLKeypoint
import pcl

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


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


def ensure_keypoint_number(frame_keypoint_np, frame_pc_np, keypoint_num):
    if frame_keypoint_np.shape[0] == keypoint_num:
        return frame_keypoint_np
    elif frame_keypoint_np.shape[0] > keypoint_num:
        return frame_keypoint_np[np.random.choice(frame_keypoint_np.shape[0], keypoint_num, replace=False), :]
    else:
        additional_frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], keypoint_num-frame_keypoint_np.shape[0], replace=False), :]
        frame_keypoint_np = np.concatenate((frame_keypoint_np, additional_frame_keypoint_np), axis=0)
        return frame_keypoint_np




def get_options(arg_parser, dataset_type='kitti'):
    if dataset_type == 'kitti':
        import kitti.options_detector

        opt_detector_instance = kitti.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == 'synth':
        import kitti.options_detector

        opt_detector_instance = kitti.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == 'luco':
        import kitti.options_detector

        opt_detector_instance = kitti.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == 'oxford':
        import oxford.options_detector

        opt_detector_instance = oxford.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == 'redwood':
        import scenenn.options_detector

        opt_detector_instance = scenenn.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == '3dmatch_eval':
        import match3d.options_detector

        opt_detector_instance = match3d.options_detector.Options(arg_parser)
        opt_detector_instance.parse_without_process()
        opt_detector_instance.opt.gpu_ids = [gpu_id]
        opt_detector_instance.opt.batch_size = 1
        opt_detector_instance.process_opts()
        opt_detector = opt_detector_instance.opt
    elif dataset_type == 'modelnet':
        import modelnet.options_detector
        opt_detector = modelnet.options_detector.Options(arg_parser).parse()
    else:
        assert False

    opt_detector.input_pc_num = int(opt_detector.input_pc_num / downsample_rate)
    return opt_detector


import open3d as o3d

import utils.ICP as ICP
class KeyPointExtractors():
    def __init__(self, opt_detector, method= 'usip',dataset_type='kitti', desired_keypoint_num=5000):
        self.opt_detector = opt_detector
        self.method = method
        self.dataset_type = dataset_type
        self.desired_keypoint_num = desired_keypoint_num

        if 'kitti' == self.dataset_type:
            
            if self.method == "nosample":
                test_txt_folder = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/sequences'
                testset = KittiTestLoader_Odometry_Original(test_txt_folder, numpy_folder, opt_detector)
            else:
                test_txt_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/kitti_odometry_bin'
                testset = KittiTestLoader_Odometry(test_txt_folder, numpy_folder, opt_detector)
        elif 'synth' == self.dataset_type:
            root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_data'
            testset = SyntheticDataLoader(root_folder, self.opt_detector)
        elif 'luco' == self.dataset_type:
            root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_data'
            testset = ParisLucoDataLoader(root_folder, self.opt_detector)                  
        elif 'oxford' == self.dataset_type:
            root_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/oxford'
            # testset = OxfordDataLoader_Odometry(root_folder, self.opt_detector)
            
            testset = OxfordTestLoader(root_folder, self.opt_detector)
        elif 'redwood' == self.dataset_type:
            npy_folder = '/home/pranayspeed/Work/git_repos/TSF_datasets/redwood/numpy_gt_normal'
            testset = RedwoodLoader(npy_folder, self.opt_detector)
        elif '3dmatch_eval' == self.dataset_type:
            testset = Match3DEvalLoader(npy_folder, self.opt_detector)
        elif 'modelnet' == self.dataset_type:
            testset = ModelNet_Rotated_Loader(root, self.opt_detector)
        else:
            assert False

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                                shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)

        if self.method == 'usip':
            # build detector
            self.model_detector = ModelDetector(self.opt_detector)
            self.model_detector.detector.load_state_dict(
                model_state_dict_convert_auto(
                    torch.load(
                        detector_model_path,
                        map_location='cpu'), self.opt_detector.gpu_ids))
            self.model_detector.freeze_model()        
        
        self.dataset_iterator = enumerate(self.testloader)
    def get_keypoints(self, seq_name='kitti', is_ensure_keypoint_num=False):
        i, data = next(self.dataset_iterator)
        
        if i is None:
            return None
        gt_pose = np.eye(4)
        
        if 'kitti' == self.dataset_type:
            anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
        elif 'synth' == self.dataset_type:
            anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
        elif 'luco' == self.dataset_type:
            anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose = data
        elif 'oxford' == self.dataset_type:
            anc_pc, anc_sn, anc_node, seq_id= data
            #anc_pc, anc_sn, anc_node, seq_id, anc_idx, gt_pose = data
        elif 'redwood' == self.dataset_type or '3dmatch_eval' == self.dataset_type:
            anc_pc, anc_sn, anc_node, scene_idx, frame_idx = data
        elif 'modelnet' == self.dataset_type:
            anc_pc, anc_sn, anc_node, anc_idx, anc_type = data
        else:
            assert False

        # add noise on anc_pc
        #anc_pc = anc_pc + torch.randn(anc_pc.size()) * noise_sigma

        
        # timing
        begin_t = time.time()
        
        method = self.method
        desired_keypoint_num = self.desired_keypoint_num
        if method == 'usip':
            anc_pc_cuda = anc_pc.to(self.opt_detector.device)
            anc_sn_cuda = anc_sn.to(self.opt_detector.device)
            anc_node_cuda = anc_node.to(self.opt_detector.device)

            # run detection
            # Bx3xM, BxM
            #print(seq, anc_idx)
            
            anc_keypoints, anc_sigmas = self.model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
            anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
            anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
        elif method == 'iss':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
                                
                keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
                
                frame_keypoint_np = np.asarray(keypoints.points)
                
                # frame_keypoint_np = PCLKeypoint.keypointIss(frame_pc_np,
                #                                             iss_salient_radius,
                #                                             iss_non_max_radius,
                #                                             iss_gamma_21,
                #                                             iss_gamma_32,
                #                                             iss_min_neighbors,
                #                                             threads)  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'harris':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = PCLKeypoint.keypointHarris3D(frame_pc_np,
                                                               radius,
                                                               nms_threshold,
                                                               threads,
                                                               True
                                                               )  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'sift':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = PCLKeypoint.keypointSift(frame_pc_np,
                                                             min_scale,
                                                             n_octaves,
                                                             n_scales_per_octave,
                                                             min_contrast)  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'random':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'uniform':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                
                               
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
                total_len = len(frame_pc_np)

                req_downsample_k = math.ceil(total_len/desired_keypoint_num)
                pcd = pcd.uniform_down_sample(req_downsample_k)
                
                #print(len(pcd.points), req_downsample_k, end='\r')
                frame_keypoint_np = np.asarray(pcd.points)
    
                #print(total_len, desired_keypoint_num, req_downsample_k, len(pcd.points))
        
                #frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
                anc_keypoints_list.append(frame_keypoint_np)            
        elif method == 'nosample':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = frame_pc_np.copy()#[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
                anc_keypoints_list.append(frame_keypoint_np)
                
        elapsed_time = time.time()-begin_t
        if is_timing:
            print("time consumed per %d frame: %f" % (anc_pc.size(0), elapsed_time))

        # if method=='nosample':
        #     arrange_indx = [1,2,0]
        #     frame_keypoint_np = anc_keypoints_list[b]
        #         # assure at least one keypoint, by randomly selecting a point
        #     if frame_keypoint_np.shape[0] == 0:
        #         frame_keypoint_np = frame_pc_np[0:1, :]
        #     frame_keypoint_np = frame_keypoint_np[:, arrange_indx]
        #     frame_pc_orig = np.transpose(anc_pc[b].detach().numpy())
        #     frame_pc_orig = frame_pc_orig[: ,arrange_indx ]
        #     return (elapsed_time,frame_pc_orig,frame_keypoint_np) 
        
        for b in range(anc_pc.size(0)):
            frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
            if method == 'usip':
                frame_keypoint_np = anc_keypoints_np[b]
                frame_sigma_np = anc_sigmas_np[b]

                # nms
                frame_keypoint_np, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

                # remove small sigma
                if is_ensure_keypoint_num:
                    sorted_sigma_idx = np.argsort(frame_sigma_np)
                    if desired_keypoint_num > frame_keypoint_np.shape[0]:
                        desired_keypoint_num = frame_keypoint_np.shape[0]
                    sorted_sigma_idx = sorted_sigma_idx[0:desired_keypoint_num]
                    frame_keypoint_np = frame_keypoint_np[sorted_sigma_idx, ...]  # Mx3
            else:
                frame_keypoint_np = anc_keypoints_list[b]
                # assure at least one keypoint, by randomly selecting a point
                if frame_keypoint_np.shape[0] == 0:
                    frame_keypoint_np = frame_pc_np[0:1, :]
            
            if is_plot:
                print(frame_keypoint_np.shape)
                ax = vis_tools.plot_pc(frame_pc_np, size=3)
                ax = vis_tools.plot_pc(frame_keypoint_np, ax=ax, size=30, color=np.asarray([1, 0, 0]).reshape((1, 3)))

                plt.savefig(seq_name + ".png")                
                exit(0)
                
            return (elapsed_time, np.transpose(anc_pc[b].detach().numpy()),frame_keypoint_np, gt_pose)        
        
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        #odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        #odom_transform = ICP_register_global(traget_kps[:min_len], source_kps[:min_len],initial_transf)
        #odom_transform, _, _ = ICP.icp_ransac(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        odom_transform, _, _ = ICP.icp_robust(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)


        return odom_transform



#from open3d import t.pipelines.registration as treg
# Overriding visualization function, according to best camera view for colored-icp sample data.
def ICP_register(source, target, init_source_to_target):
    # source_temp = source.copy()
    # target_temp = target.copy()
    source_pcd = o3d.geometry.PointCloud()
    # initialise the geometry pre loop
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd = o3d.geometry.PointCloud()
    # initialise the geometry pre loop
    target_pcd.points = o3d.utility.Vector3dVector(target)
                    
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 0.07

    # # Initial alignment or source to target transform.
    # init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                                     [-0.139, 0.967, -0.215, 0.7],
    #                                     [0.487, 0.255, 0.835, -1.4],
    #                                     [0.0, 0.0, 0.0, 1.0]])

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # Convergence-Criteria for Vanilla ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                        relative_rmse=0.0001,
                                        max_iteration=20)
    # Down-sampling voxel-size.
    voxel_size = 0.025

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True  
    registration_icp = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, max_correspondence_distance,
                            init_source_to_target, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #estimation, criteria,
    #                        voxel_size)

    
    # #print("Time taken by ICP: ", icp_time)
    # print("Inlier Fitness: ", registration_icp.fitness)
    # print("Inlier RMSE: ", registration_icp.inlier_rmse)

    return registration_icp.transformation


def ICP_register_global(source, target, init_source_to_target):
    # source_temp = source.copy()
    # target_temp = target.copy()
    source_pcd = o3d.geometry.PointCloud()
    # initialise the geometry pre loop
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd = o3d.geometry.PointCloud()
    # initialise the geometry pre loop
    target_pcd.points = o3d.utility.Vector3dVector(target)
    voxel_size=0.5
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
                 
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)

    return result_ransac.transformation


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):

    
    distance_threshold = voxel_size * 0.5
    #print(":: Apply fast global registration with distance threshold %.3f" \
    #        % distance_threshold)
    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=distance_threshold))
    
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down =  copy.deepcopy(pcd) #  pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
      
# def get_keypoints(opt_detector, method= 'usip',dataset_type='kitti', desired_keypoint_num=5000):


#     if 'kitti' == dataset_type:
#         testset = KittiTestLoader_Odometry(test_txt_folder, numpy_folder, opt_detector)
#     elif 'oxford' == dataset_type:
#         testset = OxfordTestLoader(root_folder, opt_detector)
#     elif 'redwood' == dataset_type:
#         testset = RedwoodLoader(npy_folder, opt_detector)
#     elif '3dmatch_eval' == dataset_type:
#         testset = Match3DEvalLoader(npy_folder, opt_detector)
#     elif 'modelnet' == dataset_type:
#         testset = ModelNet_Rotated_Loader(root, opt_detector)
#     else:
#         assert False

#     testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
#                                              shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)

#     if method == 'usip':
#         # build detector
#         model_detector = ModelDetector(opt_detector)
#         model_detector.detector.load_state_dict(
#             model_state_dict_convert_auto(
#                 torch.load(
#                     detector_model_path,
#                     map_location='cpu'), opt_detector.gpu_ids))
#         model_detector.freeze_model()


#     keypoint_num_list = []
#     for i, data in enumerate(testloader):
#         if 'kitti' == dataset_type:
#             anc_pc, anc_sn, anc_node, seq, anc_idx = data
#         elif 'oxford' == dataset_type:
#             anc_pc, anc_sn, anc_node, anc_idx = data
#         elif 'redwood' == dataset_type or '3dmatch_eval' == dataset_type:
#             anc_pc, anc_sn, anc_node, scene_idx, frame_idx = data
#         elif 'modelnet' == dataset_type:
#             anc_pc, anc_sn, anc_node, anc_idx, anc_type = data
#         else:
#             assert False

#         # add noise on anc_pc
#         #anc_pc = anc_pc + torch.randn(anc_pc.size()) * noise_sigma

#         # timing
#         begin_t = time.time()
        
        
#         if method == 'usip':
#             anc_pc_cuda = anc_pc.to(opt_detector.device)
#             anc_sn_cuda = anc_sn.to(opt_detector.device)
#             anc_node_cuda = anc_node.to(opt_detector.device)

#             # run detection
#             # Bx3xM, BxM
#             print(seq, anc_idx)
            
#             anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
#             anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
#             anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
#         elif method == 'iss':
#             anc_keypoints_list = []
#             for b in range(anc_pc.size(0)):
#                 frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
#                 frame_keypoint_np = PCLKeypoint.keypointIss(frame_pc_np,
#                                                             iss_salient_radius,
#                                                             iss_non_max_radius,
#                                                             iss_gamma_21,
#                                                             iss_gamma_32,
#                                                             iss_min_neighbors,
#                                                             threads)  # Mx3
#                 if is_ensure_keypoint_num:
#                     frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
#                 anc_keypoints_list.append(frame_keypoint_np)
#         elif method == 'harris':
#             anc_keypoints_list = []
#             for b in range(anc_pc.size(0)):
#                 frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
#                 frame_keypoint_np = PCLKeypoint.keypointHarris3D(frame_pc_np,
#                                                                radius,
#                                                                nms_threshold,
#                                                                threads)  # Mx3
#                 if is_ensure_keypoint_num:
#                     frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
#                 anc_keypoints_list.append(frame_keypoint_np)
#         elif method == 'sift':
#             anc_keypoints_list = []
#             for b in range(anc_pc.size(0)):
#                 frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
#                 frame_keypoint_np = PCLKeypoint.keypointSift(frame_pc_np,
#                                                              min_scale,
#                                                              n_octaves,
#                                                              n_scales_per_octave,
#                                                              min_contrast)  # Mx3
#                 if is_ensure_keypoint_num:
#                     frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
#                 anc_keypoints_list.append(frame_keypoint_np)
#         elif method == 'random':
#             anc_keypoints_list = []
#             for b in range(anc_pc.size(0)):
#                 frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
#                 frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
#                 anc_keypoints_list.append(frame_keypoint_np)
#         elif method == 'uniform':
#             anc_keypoints_list = []
#             for b in range(anc_pc.size(0)):
#                 frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                

                
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
#                 total_len = len(frame_pc_np)

#                 req_downsample_k = math.ceil(total_len/desired_keypoint_num)
#                 pcd = pcd.uniform_down_sample(req_downsample_k)
                
#                 #print(len(pcd.points), req_downsample_k, end='\r')
#                 frame_keypoint_np = np.asarray(pcd.points)
    
#                 print(total_len, desired_keypoint_num, req_downsample_k, len(pcd.points))
        
#                 #frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
#                 anc_keypoints_list.append(frame_keypoint_np)            

#         if is_timing:
#             print("time consumed per %d frame: %f" % (anc_pc.size(0), time.time()-begin_t))

#         for b in range(anc_pc.size(0)):
#             frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
#             if method == 'usip':
#                 frame_keypoint_np = anc_keypoints_np[b]
#                 frame_sigma_np = anc_sigmas_np[b]

#                 # nms
#                 frame_keypoint_np, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

#                 # remove small sigma
#                 if is_ensure_keypoint_num:
#                     sorted_sigma_idx = np.argsort(frame_sigma_np)
#                     if desired_keypoint_num > frame_keypoint_np.shape[0]:
#                         desired_keypoint_num = frame_keypoint_np.shape[0]
#                     sorted_sigma_idx = sorted_sigma_idx[0:desired_keypoint_num]
#                     frame_keypoint_np = frame_keypoint_np[sorted_sigma_idx, ...]  # Mx3
#             else:
#                 frame_keypoint_np = anc_keypoints_list[b]
#                 # assure at least one keypoint, by randomly selecting a point
#                 if frame_keypoint_np.shape[0] == 0:
#                     frame_keypoint_np = frame_pc_np[0:1, :]
#             yield (np.transpose(anc_pc[b].detach().numpy()),frame_keypoint_np)



if __name__ == '__main__':
    output_folder_real = output_folder + '/' + method
    # augment output_folder with random characters to avoid multi-process inference.
    output_folder = output_folder_real + "_" + random_string_generator(8)

    if 'kitti' == dataset_type:
        testset = KittiTestLoader(test_txt_folder, numpy_folder, opt_detector)
    elif 'oxford' == dataset_type:
        testset = OxfordTestLoader(root_folder, opt_detector)
    elif 'redwood' == dataset_type:
        testset = RedwoodLoader(npy_folder, opt_detector)
    elif '3dmatch_eval' == dataset_type:
        testset = Match3DEvalLoader(npy_folder, opt_detector)
    elif 'modelnet' == dataset_type:
        testset = ModelNet_Rotated_Loader(root, opt_detector)
    else:
        assert False

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)

    if method == 'usip':
        # build detector
        model_detector = ModelDetector(opt_detector)
        model_detector.detector.load_state_dict(
            model_state_dict_convert_auto(
                torch.load(
                    detector_model_path,
                    map_location='cpu'), opt_detector.gpu_ids))
        model_detector.freeze_model()


    keypoint_num_list = []
    for i, data in enumerate(testloader):
        if 'kitti' == dataset_type:
            anc_pc, anc_sn, anc_node, seq, anc_idx = data
        elif 'oxford' == dataset_type:
            anc_pc, anc_sn, anc_node, anc_idx = data
        elif 'redwood' == dataset_type or '3dmatch_eval' == dataset_type:
            anc_pc, anc_sn, anc_node, scene_idx, frame_idx = data
        elif 'modelnet' == dataset_type:
            anc_pc, anc_sn, anc_node, anc_idx, anc_type = data
        else:
            assert False

        # add noise on anc_pc
        anc_pc = anc_pc + torch.randn(anc_pc.size()) * noise_sigma

        # timing
        begin_t = time.time()
        print(method)
        if method == 'usip':
            anc_pc_cuda = anc_pc.to(opt_detector.device)
            anc_sn_cuda = anc_sn.to(opt_detector.device)
            anc_node_cuda = anc_node.to(opt_detector.device)

            # run detection
            # Bx3xM, BxM
            print(seq, anc_idx)
            anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
            anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
            anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
        elif method == 'iss':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = PCLKeypoint.keypointIss(frame_pc_np,
                                                            iss_salient_radius,
                                                            iss_non_max_radius,
                                                            iss_gamma_21,
                                                            iss_gamma_32,
                                                            iss_min_neighbors,
                                                            threads)  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'harris':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = PCLKeypoint.keypointHarris(frame_pc_np,
                                                               radius,
                                                               nms_threshold,
                                                               threads)  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'sift':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = PCLKeypoint.keypointSift(frame_pc_np,
                                                             min_scale,
                                                             n_octaves,
                                                             n_scales_per_octave,
                                                             min_contrast)  # Mx3
                if is_ensure_keypoint_num:
                    frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
                anc_keypoints_list.append(frame_keypoint_np)
        elif method == 'random':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
                anc_keypoints_list.append(frame_keypoint_np)

        if is_timing:
            print("time consumed per %d frame: %f" % (anc_pc.size(0), time.time()-begin_t))

        for b in range(anc_pc.size(0)):
            frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
            if method == 'usip':
                frame_keypoint_np = anc_keypoints_np[b]
                frame_sigma_np = anc_sigmas_np[b]

                # nms
                frame_keypoint_np, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

                # remove small sigma
                if is_ensure_keypoint_num:
                    sorted_sigma_idx = np.argsort(frame_sigma_np)
                    if desired_keypoint_num > frame_keypoint_np.shape[0]:
                        desired_keypoint_num = frame_keypoint_np.shape[0]
                    sorted_sigma_idx = sorted_sigma_idx[0:desired_keypoint_num]
                    frame_keypoint_np = frame_keypoint_np[sorted_sigma_idx, ...]  # Mx3
            else:
                frame_keypoint_np = anc_keypoints_list[b]
                # assure at least one keypoint, by randomly selecting a point
                if frame_keypoint_np.shape[0] == 0:
                    frame_keypoint_np = frame_pc_np[0:1, :]

            # plot
            if is_plot:
                print(frame_keypoint_np.shape)
                ax = vis_tools.plot_pc(frame_pc_np, size=3)
                ax = vis_tools.plot_pc(frame_keypoint_np, ax=ax, size=30, color=np.asarray([1, 0, 0]).reshape((1, 3)))

                plt.show()
                break

            # write to file
            if 'kitti' == dataset_type:
                if not os.path.isdir(os.path.join(output_folder, '%02d' % seq[b].item())):
                    os.makedirs(os.path.join(output_folder, '%02d' % seq[b].item()))
                output_file = os.path.join(output_folder, '%02d' % seq[b].item(), '%06d.bin' % anc_idx[b].item())
            elif 'oxford' == dataset_type:
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                output_file = os.path.join(output_folder, '%d.bin' % anc_idx[b].item())
            elif 'redwood' == dataset_type or '3dmatch_eval' == dataset_type:
                if not os.path.isdir(os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()])):
                    os.makedirs(os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()]))
                output_file = os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()], '%d.bin' % frame_idx[b].item())
            elif 'modelnet' == dataset_type:
                if anc_type[b].item() == 0:
                    output_folder_new = os.path.join(output_folder, 'original')
                elif anc_type[b].item() == 1:
                    output_folder_new = os.path.join(output_folder, 'rotated')
                else:
                    assert False
                if not os.path.isdir(output_folder_new):
                    os.makedirs(output_folder_new)
                output_file = os.path.join(output_folder_new, '%d.bin' % anc_idx[b].item())
            else:
                assert False
            output = frame_keypoint_np
            output = output.astype(np.float32)
            output.tofile(output_file)

            # print info
            print(output_file + ': %d' % output.shape[0])
            keypoint_num_list.append(output.shape[0])

    keypoint_num_np = np.asarray(keypoint_num_list)
    print('keypoint number max: %d, min: %d, mean: %d' % (np.max(keypoint_num_np),
                                                          np.min(keypoint_num_np),
                                                          round(np.mean(keypoint_num_np))))
    output_folder_real = output_folder_real + '_%d' % (round(np.mean(keypoint_num_np)))
    # output_folder_real = output_folder_real + '_%d_sigma_%.3f' % (round(np.mean(keypoint_num_np)), noise_sigma)
    # output_folder_real = output_folder_real + '_%d_lambda_%.3f' % (round(np.mean(keypoint_num_np)), opt_detector.keypoint_on_pc_alpha)
    # output_folder_real = output_folder_real + '_%d_downsample_%d' % (round(np.mean(keypoint_num_np)), downsample_rate)
    print(output_folder_real)
    try:
        assert False
        shutil.copytree(output_folder, output_folder_real)
        shutil.rmtree(output_folder)
    except Exception as e:
        print(e)
        print("tmp folder is: %s" % output_folder)

