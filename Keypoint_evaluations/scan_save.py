
import numpy as np

import utils.UtilsPointcloud as Ptutils
data_base_dir = '/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/sequences'
ground_truth_dir='/home/pranayspeed/Work/git_repos/data_odometry_velodyne_part/poses'
sequence_idx='00'
kitti_scan_bin = '/home/pranayspeed/Work/git_repo/datasets/kitti_numpy/00/000000.bin'


def readKittiScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]
    return ptcloud_xyz

# dataset 
# sequence_manager = Ptutils.KittiScanDirManager(data_base_dir, ground_truth_dir, sequence_idx)
# scan_paths = sequence_manager.scan_fullpaths
# num_frames = len(scan_paths)
# gt_fullpath = sequence_manager.gt_fullpath


curr_scan_pts = readKittiScan(kitti_scan_bin)
print(curr_scan_pts.shape)
np.savetxt("kitti_scan_00_0.txt", curr_scan_pts)