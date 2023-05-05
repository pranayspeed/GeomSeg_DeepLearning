import os 
import random
import numpy as np

def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points

    points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
    down_points = orig_points[points_down_idx, :]

    return down_points

def readScan(bin_path, dataset='KITTI'):
    if(dataset == 'KITTI'):
        return readKittiScan(bin_path)


def readKittiScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]
    return ptcloud_xyz
    

# def readgroundtruth(gt_path, dataset='KITTI'):
#     if(dataset == 'KITTI'):
#         return readKittigroundtruth(gt_path)

def readKittigroundtruth(gt_path, num_poses):
    with open(gt_path) as file:
        gt_poses = np.array([[float(val) for val in line.split()] for line in file])
    #gt_poses = np.fromfile(gt_path, dtype=np.float32)  
    #print(num_poses)
    #print(gt_poses)
    #gt_poses = gt_poses.reshape((-1, 12)) 
    print(gt_poses.shape)
    
    #gt_poses.append([0.0, 0.0, 0.0, 1.0]) 
    return gt_poses
class KittiScanDirManager:
    def __init__(self, scan_dir, gt_dir, sequence_idx):

        self.scan_dir = os.path.join(scan_dir, sequence_idx, 'velodyne')
        
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names]
  
        self.num_scans = len(self.scan_names)
        
        self.gt_fullpath = os.path.join(gt_dir, str(sequence_idx)+ '.txt')
        
        self.gt_poses=readKittigroundtruth(self.gt_fullpath, self.num_scans)


    def __repr__(self):
        return ' ' + str(self.num_scans) + ' scans in the sequence (' + self.scan_dir + '/)'

    def get_gt_pose(self, indx):
        curr_pose = self.gt_poses[indx]
        curr_pose = curr_pose.reshape((-1,4))
        pose_out = np.eye(4)
        pose_out[:3,:] = curr_pose
        #pose_out = pose_out.flatten()
        return pose_out
        
    def getScanNames(self):
        return self.scan_names
    def getScanFullPaths(self):
        return self.scan_fullpaths
    def printScanFullPaths(self):
        return print("\n".join(self.scan_fullpaths))

