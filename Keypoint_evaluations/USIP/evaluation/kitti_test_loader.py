import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py

from data.augmentation import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from util import vis_tools


def load_kitti_test_gt_txt(txt_root, seq):
    '''

    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]
    '''
    dataset = []
    print(os.path.join(txt_root, '%02d'%seq, 'groundtruths.txt'))
    with open(os.path.join(txt_root, '%02d'%seq, 'groundtruths.txt'), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):
            if i == 0:
                # skip the header line
                continue
            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])

            # search for existence
            anc_idx_is_exist = False
            pos_idx_is_exist = False
            for tmp_data in dataset:
                if tmp_data['anc_idx'] == anc_idx:
                    anc_idx_is_exist = True
                if tmp_data['anc_idx'] == pos_idx:
                    pos_idx_is_exist = True

            if anc_idx_is_exist is False:
                data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
                dataset.append(data)
            if pos_idx_is_exist is False:
                data = {'seq': seq, 'anc_idx': pos_idx, 'pos_idx': anc_idx}
                dataset.append(data)

    return dataset


def make_kitti_test_dataset(txt_root):
    folder_list = os.listdir(txt_root)
    folder_list.sort()
    folder_int_list = [int(x) for x in folder_list]

    dataset = []
    for seq in folder_int_list:
        dataset += (load_kitti_test_gt_txt(txt_root, seq))
    # print(dataset)
    # print(len(dataset))
    return dataset


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class KittiTestLoader(data.Dataset):
    def __init__(self, txt_root, numpy_root, opt):
        super(KittiTestLoader, self).__init__()
        self.txt_root = txt_root
        self.numpy_root = numpy_root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.dataset = make_kitti_test_dataset(txt_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = self.dataset[index]['seq']
        anc_idx = self.dataset[index]['anc_idx']

        # ===== load numpy array =====
        pc_np_file = os.path.join(self.numpy_root, '%02d' % seq, 'np_0.20_20480_r90_sn', '%06d.npy' % anc_idx)

        # random choice
        assert self.opt.surface_normal_len == 4
        pc_np = np.load(pc_np_file)  # Nx8, x, y, z, sn_x, sn_y, sn_z, curvature, reflectance
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)
        # ===== load numpy array =====

        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, seq, anc_idx



class KittiTestLoader_Odometry(data.Dataset):
    def __init__(self, txt_root, numpy_root, opt):
        super(KittiTestLoader_Odometry, self).__init__()
        self.txt_root = txt_root
        self.numpy_root = numpy_root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.sequence_idx = '%02d' % opt.sequence #'00'
        
        self.seq = opt.sequence
        scan_dir = os.path.join(txt_root, self.sequence_idx)
        
        self.scan_names = os.listdir(scan_dir)
        self.scan_names.sort() 
        
        self.dataset=self.scan_names
        
        ##
        self.gt_dir = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/poses'
        self.num_scans = len(self.scan_names)
        
        self.gt_fullpath = os.path.join(self.gt_dir, str(self.sequence_idx)+ '.txt')
        
        self.gt_poses=readKittigroundtruth(self.gt_fullpath, self.num_scans)
        
        
    def __len__(self):
        return len(self.dataset)

    def get_gt_pose(self, indx):
        curr_pose = self.gt_poses[indx]
        curr_pose = curr_pose.reshape((-1,4))
        pose_out = np.eye(4)
        pose_out[:3,:] = curr_pose
        #pose_out = pose_out.flatten()
        return pose_out
    
    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = self.seq #self.dataset[index]['seq']
        anc_idx = index #self.dataset[index]['anc_idx']

        # ===== load numpy array =====
        pc_np_file = os.path.join(self.numpy_root, '%02d' % seq, 'np_0.20_20480_r90_sn', '%06d.npy' % anc_idx)

        # random choice
        assert self.opt.surface_normal_len == 4
        pc_np = np.load(pc_np_file)  # Nx8, x, y, z, sn_x, sn_y, sn_z, curvature, reflectance
        
        
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)
        # ===== load numpy array =====

        
        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        #print(anc_pc.size(), anc_sn.size(), anc_node.size())
        
        
        return anc_pc, anc_sn, anc_node, seq, anc_idx, self.get_gt_pose(index)



def readScan(bin_path, dataset='KITTI'):
    if(dataset == 'KITTI'):
        return readKittiScan(bin_path)


def readKittiScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]
    return ptcloud_xyz

def readKittigroundtruth(gt_path, num_poses):
    with open(gt_path) as file:
        gt_poses = np.array([[float(val) for val in line.split()] for line in file])
    #gt_poses = np.fromfile(gt_path, dtype=np.float32)  
    #print(num_poses)
    #print(gt_poses)
    #gt_poses = gt_poses.reshape((-1, 12)) 
    #print(gt_poses.shape)
    
    #gt_poses.append([0.0, 0.0, 0.0, 1.0]) 
    return gt_poses
class KittiTestLoader_Odometry_Original(data.Dataset):
    def __init__(self, txt_root, numpy_root, opt):
        super(KittiTestLoader_Odometry_Original, self).__init__()
        self.txt_root = txt_root
        self.numpy_root = numpy_root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        sequence_idx = '00'
        #scan_dir = os.path.join(txt_root, '00')
        
        #scan_names = os.listdir(scan_dir)
        #scan_names.sort() 
        self.scan_dir = os.path.join(txt_root, sequence_idx, 'velodyne')
        
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names]
        
        self.dataset=self.scan_fullpaths
        
        ##
        self.gt_dir = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/poses'
        self.num_scans = len(self.scan_names)
        
        self.gt_fullpath = os.path.join(self.gt_dir, str(sequence_idx)+ '.txt')
        
        self.gt_poses=readKittigroundtruth(self.gt_fullpath, self.num_scans)
        
    def __len__(self):
        return len(self.dataset)

    def get_gt_pose(self, indx):
        curr_pose = self.gt_poses[indx]
        curr_pose = curr_pose.reshape((-1,4))
        pose_out = np.eye(4)
        pose_out[:3,:] = curr_pose
        #pose_out = pose_out.flatten()
        return pose_out

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = 0 #self.dataset[index]['seq']
        anc_idx = index #self.dataset[index]['anc_idx']

        pc_np = readKittiScan(self.scan_fullpaths[index])


        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = [] #torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node =  [] #torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, seq, anc_idx, self.get_gt_pose(index)

import open3d as o3d

# def readSynthScan(ply_path):
#     #print(ply_path)
#     #ply_path = ply_path.replace("._", "")
    
#     pcd = o3d.io.read_point_cloud(ply_path)
#     return np.asarray(pcd.points)
def readSynthScan(pc_np_file):
    #print(ply_path)
    return np.load(pc_np_file) 

def readSynth_gt_poses(vo_path_npy):
    #print(vo_path)
    
    pc_np = np.load(vo_path_npy)
    #print(pc_np.shape)
    pc_np = pc_np[:,:3]
    return pc_np



def pca_compute(data, sort=True):
    """
	SVD decomposition
    """
    average_data = np.mean(data, axis=0) 
    decentration_matrix = data - average_data
    H = np.dot(decentration_matrix.T, decentration_matrix)
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
    return eigenvalues

def caculate_surface_curvature(cloud, radius=0.003):
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
    curvature = []  
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
        neighbors = points[idx, :]
        w = pca_compute(neighbors)
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float32)
    return curvature

class SyntheticDataLoader(data.Dataset):
    def __init__(self, txt_root, opt):
        super(SyntheticDataLoader, self).__init__()
        #self.txt_root = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_data'
        self.txt_root = '/home/pranayspeed/Work/git_repos/TSF_datasets/mai_sim_crop_128_pranay/'

        self.opt = opt
        sequence_idx = 'frames'
        self.scan_dir = os.path.join(self.txt_root, sequence_idx)
        
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names]
        
        self.dataset=self.scan_fullpaths
        
        self.gt_dir = self.txt_root
        self.gt_fullpath = os.path.join(self.gt_dir,  'poses_exact.npy')
                
        self.gt_poses = readSynth_gt_poses(self.gt_fullpath)

        # farthest point sample
        self.farthest_sampler = FarthestSampler()


    def __len__(self):
        return len(self.dataset)

    def get_gt_pose(self, indx):
        curr_pose = self.gt_poses[indx]        
        #print(curr_pose.shape)
        pose_out = np.eye(4)
        pose_out[:3,3] = curr_pose

        return pose_out

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = 0 #self.dataset[index]['seq']
        anc_idx = index #self.dataset[index]['anc_idx']
        #print(self.scan_fullpaths[index])
        #print(self.scan_dir, index)
        pc_np = readSynthScan(self.scan_fullpaths[index])


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np) 

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))        
        
        curvatures = caculate_surface_curvature(pcd)
        
        curvatures = curvatures.reshape(-1,1)
        #curvatures = curvatures.transpose()
        print("curvature ",curvatures.shape)
        sn_np = np.asarray(pcd.normals)
        sn_np = np.append(sn_np, curvatures, 1)
        print(sn_np.shape)
        
        
        
        #sn_np = torch.unsqueeze(sn_np, dim=0)
        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)


        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn =  torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node =  torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        print(anc_pc.size(), anc_sn.size(), anc_node.size())
        
        return anc_pc, anc_sn, anc_node, seq, anc_idx, self.get_gt_pose(index)

##########################################

def readParisLucogroundtruth(gt_path):
    with open(gt_path) as file:
        gt_poses = np.array([[float(val) for val in line.split()] for line in file])
    return gt_poses



def readParisLucoScan(ply_path):
    #print(ply_path)
    #ply_path = ply_path.replace("._", "")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

class ParisLucoDataLoader(data.Dataset):
    def __init__(self, txt_root, opt):
        super(ParisLucoDataLoader, self).__init__()
        self.txt_root = '/home/pranayspeed/Downloads/ParisLuco/00/'

        self.opt = opt

        self.scan_dir = os.path.join(self.txt_root, 'frames')
        
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names]
        
        self.dataset=self.scan_fullpaths

        self.gt_fullpath = os.path.join(self.txt_root, "gt_traj_lidar.txt")
        
        self.gt_poses=readParisLucogroundtruth(self.gt_fullpath)

        # farthest point sample
        self.farthest_sampler = FarthestSampler()
        

    def __len__(self):
        return len(self.dataset)

    def get_gt_pose(self, indx):
        curr_pose = self.gt_poses[indx]
        
        #print(curr_pose.shape)
        pose_out = np.eye(4)
        pose_out[:3,3] = curr_pose
        #pose_out = pose_out.flatten()
        #print(pose_out.flatten())
        return pose_out

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = 0 #self.dataset[index]['seq']
        anc_idx = index #self.dataset[index]['anc_idx']
        #print(self.scan_fullpaths[index])
        #print(self.scan_dir, index)
        pc_np = readParisLucoScan(self.scan_fullpaths[index])



        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np) 

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))        
        
        curvatures = caculate_surface_curvature(pcd)
        
        curvatures = curvatures.reshape(-1,1)
        #curvatures = curvatures.transpose()
        #print("curvature ",curvatures.shape)
        sn_np = np.asarray(pcd.normals)
        sn_np = np.append(sn_np, curvatures, 1)
        #print(sn_np.shape)
        
        
        
        #sn_np = torch.unsqueeze(sn_np, dim=0)
        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)



        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node =  torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, seq, anc_idx, self.get_gt_pose(index)









######################################







import pandas as pd

def readOxfordScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]
    return ptcloud_xyz


def load_velodyne_binary(velodyne_bin_path):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    #ptcld = data.reshape((4, -1))
    
    scan = data.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]

    return ptcloud_xyz

def readOxford_gt_poses(vo_path_npy):
    #print(vo_path)
    
    pc_np = np.load(vo_path_npy)
    print(pc_np.shape)
    return pc_np
    
    poses_df = pd.read_csv(vo_path)    
    
    abs_poses = [matlib.identity(4)]
    poses_rel = poses_df.to_numpy()
    for row in poses_rel:
        
        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        
        abs_pose = abs_poses[-1] * rel_pose
        abs_poses.append(abs_pose)
    
    
    return np.asarray(abs_poses)


import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4
def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.
    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix
    Raises:
        ValueError: if `len(rpy) != 3`.
    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx

def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.
    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix
    Raises:
        ValueError: if `len(xyzrpy) != 6`
    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 =  matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

import numpy.matlib as ml

class OxfordDataLoader_Odometry(data.Dataset):
    def __init__(self, txt_root, opt):
        super(OxfordDataLoader_Odometry, self).__init__()
        self.txt_root = '/home/pranayspeed/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial_Velodyne_HDL-32E_Left_Pointcloud'
        

        self.opt = opt

        self.sequence = "2019-01-10-14-36-48-radar-oxford-10k-partial"
        
        lidar = 'velodyne_left'
        self.scan_dir = os.path.join(self.txt_root,self.sequence, lidar)
        
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names if '.bin' in name]
        
        self.dataset=self.scan_fullpaths
        #print(len(self.scan_fullpaths))
        #self.dataset = make_kitti_test_dataset(txt_root)

        self.gt_path = os.path.join(self.txt_root,self.sequence, 'vo','poses.npy')
        
        self.gt_poses = readOxford_gt_poses(self.gt_path)
        self.abs_poses = [ml.identity(4)]
        
        
        extrinsics_dir = os.path.join(self.txt_root,self.sequence)
        with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
        self.G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])     
       
       
       ################
        velodyne_dir = self.scan_dir
        timestamps_path = velodyne_dir + '.timestamps'
        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file: {}".format(timestamps_path))

        extension = ".bin"
        velodyne_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
        self.dataset=[]
        for velodyne_timestamp in velodyne_timestamps:

            self.dataset.append(os.path.join(self.scan_dir, str(velodyne_timestamp) + extension))
    
     
    def __len__(self):
        return len(self.dataset)


    def get_gt_pose(self, indx):
        row = self.gt_poses[indx]
        #return row
        return np.dot(row, self.G_posesource_laser)

        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)
        
        abs_pose = self.abs_poses[-1] * rel_pose
        self.abs_poses.append(abs_pose)
        # curr_pose = self.gt_poses[indx]
        # curr_pose = curr_pose.reshape((-1,4))
        # pose_out = np.eye(4)
        # pose_out[:3,:] = curr_pose
        # #pose_out = pose_out.flatten()
        return abs_pose

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = 0 #self.dataset[index]['seq']
        anc_idx = index #self.dataset[index]['anc_idx']
        #print(self.scan_fullpaths[index])
        #print(self.scan_dir, index)
        pc_np = load_velodyne_binary(self.scan_fullpaths[index])

        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = [] #torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node =  [] #torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, seq, anc_idx, self.get_gt_pose(index)



if __name__ == '__main__':
    import kitti.options_detector
    import kitti.options_descriptor

    opt_detector = kitti.options_detector.Options().parse()

    kitti_testset = KittiTestLoader('/ssd/dataset/kitti-reg-test', '/ssd/dataset/odometry/data_odometry_velodyne/numpy', opt_detector)
    print(len(kitti_testset))
    anc_pc, anc_sn, anc_node, seq, anc_idx = kitti_testset[537]
    print(seq)
    print(anc_idx)

    testloader = torch.utils.data.DataLoader(kitti_testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    print(len(testloader))
    loader_iter = iter(testloader)
    a = loader_iter.next()
    b = loader_iter.next()
    print(a)
    print(b)


