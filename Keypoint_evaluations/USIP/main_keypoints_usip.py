
import open3d as o3d
import numpy as np

# for sift and harris 3d 
#import PCLKeypoint
import pcl


import os
import sys
import csv
import copy
import time
import random
import argparse

import numpy as np
np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

#from minisam import *
from utils.ScanContextManager import *
#from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP


from KeyPoints import *

from My3dVis import My3DVis


import time
# import pandas as pd
import pandas as pd

from save_keypoints import get_options, KeyPointExtractors




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


parser.add_argument('--method',  type=str, 
                    default="usip")

parser.add_argument('--sequence',  type=int, 
                    default=0)

# parser.add_argument('--dataset',  type=str, 
#                     default="kitti")

opt_detector = get_options(parser)


args = opt_detector# .opt# parser.parse_args()



######################################
# Select keypoint extractor
#Extractor


keypoint_extractor_name =  args.method #keypoint_extractor.__class__.__name__

print(keypoint_extractor_name)
#keypoint_extractor = NoSampleExtractor()
#keypoint_extractor = RandomSampleExtractor(args.num_icp_points)
#keypoint_extractor = UniformSampleExtractor(args.num_icp_points)
#keypoint_extractor = Harris3DExtractor()
#keypoint_extractor = SIFT3DExtractor()
#keypoint_extractor = ISSExtractor()
#######################################

#Sequence figure name 

# seq_name = args.sequence_idx + "_" + keypoint_extractor_name
# if args.extractor ==1 or args.extractor==2:
#     seq_name+= "_" + str(args.num_icp_points)
    
    
seq_name = args.dataset+"_"+args.sequence_idx + "_" + keypoint_extractor_name
#if args.method =='ramdom' or args.method =='uniform' :
seq_name+= "_" + str(args.num_icp_points)
###########################################


# dataset 
sequence_manager = Ptutils.KittiScanDirManager(args.data_base_dir, args.ground_truth_dir, args.sequence_idx)
scan_paths = sequence_manager.scan_fullpaths
num_frames = len(scan_paths)
gt_fullpath = sequence_manager.gt_fullpath

# # Pose Graph Manager (for back-end optimization) initialization
# PGM = PoseGraphManager()
# PGM.addPriorFactor()

curr_transform = np.identity(4)


# Result saver
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose=curr_transform, 
                             save_gap=args.save_gap,
                             num_frames=num_frames,
                             seq_idx=args.sequence_idx,
                             save_dir=save_dir)

# Scan Context Manager (for loop detection) initialization
SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                        num_candidates=args.num_candidates, 
                                        threshold=args.loop_threshold)

# for save the results as a video
fig_idx = 1
fig = plt.figure(fig_idx)
writer = FFMpegWriter(fps=15)
video_name = args.sequence_idx + "_" + keypoint_extractor_name + "_" + str(args.num_icp_points) + ".mp4"
num_frames_to_skip_to_show = 5
num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)

if args.show_map:
    myvis = My3DVis(args.mesh_map)

    myvis.vis.get_render_option().mesh_show_back_face= True

odom_transform = np.identity(4)
icp_initial = np.eye(4) 



gt_poses = []

frame_limit = 600

keypoint_extraction_times = []
matching_times = []
keypoint_counts = [] 
absoulte_trajectory_errors = []


keypoint_extractor = KeyPointExtractors(opt_detector, method=args.method, dataset_type=args.dataset, desired_keypoint_num=args.num_icp_points)

try:
    #with writer.saving(fig, video_name, num_frames_to_save): # this video saving part is optional
    for_idx=-1
    # @@@ MAIN @@@: data stream
    #for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):
    
    print(args.method)
    next_data = keypoint_extractor.get_keypoints(seq_name)
    while next_data is not None:
        
    #for next_data in keypoint_extractor.get_keypoints():
        (elapsed_time, curr_scan_pts, curr_keypts, gt_pose_curr) = next_data
        next_data = keypoint_extractor.get_keypoints()
        for_idx+=1
        if for_idx ==frame_limit:
            break
        
        print(args.method +"-> Input Points count: "+str(curr_scan_pts.shape))
        
        # get current information     
        #curr_scan_pts = Ptutils.readScan(scan_path) 
        
        
        curr_gt_transf = gt_pose_curr # sequence_manager.get_gt_pose(for_idx)
        #print(gt_pose)
        #gt_pose = Ptutils.readgroundtruth(gt_fullpath)

        # get the start time
        start_time = time.time()

        #curr_keypts = keypoint_extractor.extract(curr_scan_pts)   
        
        # get the end time
        end_time = time.time()
        
        
        # get the execution time
        #elapsed_time = end_time - start_time
        keypoint_extraction_times.append(elapsed_time)
        
        keypoint_counts.append(len(curr_keypts))
        #print(args.show_map)
        if args.show_map: 
            # tmp_kpts = curr_keypts.transpose()
            # scan = np.dot(curr_transform, np.vstack([tmp_kpts, np.ones((1, tmp_kpts.shape[1]))]))
            # #print(curr_keypts.transpose().shape)
            
            # curr_keypts = scan[[1,0,2]].transpose()
            myvis.set_point_cloud(curr_keypts)   
                
            #Not using curr_transform as not doing optimization
            myvis.set_transform(curr_transform)
            #myvis.set_transform(np.eye(4))
            myvis.refresh()    

        if(for_idx == 0):
            prev_scan_pts = copy.deepcopy(curr_scan_pts)               
            prev_keypts = copy.deepcopy(curr_keypts)              
            icp_initial = np.eye(4)   
            
            #Setting first matching time to zero, as no matching required
            matching_times.append(0)             
            continue


        # get the start time
        start_time = time.time()

        
        # np.savetxt("prev_keypts.txt", prev_keypts)
        # np.savetxt("curr_keypts.txt", curr_keypts)
         
        # calc odometry        
        odom_transform = keypoint_extractor.compute_transform(prev_keypts, curr_keypts, icp_initial)
        #odom_transform = keypoint_extractor.compute_transform(prev_scan_pts, curr_scan_pts, icp_initial)

        
        # get the end time
        end_time = time.time()
        
        # get the execution time
        elapsed_time = end_time - start_time
        matching_times.append(elapsed_time)

        # update the current (moved) pose 
        curr_transform = np.matmul(curr_transform, odom_transform)

        icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)


        # renewal the prev information 
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
        prev_keypts = copy.deepcopy(curr_keypts)
        
        
        # save the ICP odometry pose result (no loop closure)
        ResultSaver.saveUnoptimizedPoseGraphResult(curr_transform, curr_gt_transf, for_idx) 
        if(for_idx % num_frames_to_skip_to_show == 0): 

            legends={}
            legends['KP_Extract_Time'] = keypoint_extraction_times[-1]
            legends['KP_Matching_Time'] = matching_times[-1]
            legends['KP_count'] = keypoint_counts[-1]
            
            result_dir = "results_1/"
            
            curr_ate = ResultSaver.vizCurrentTrajectory(fig_idx, seq_name, legends)
            plt.savefig(result_dir+seq_name + ".png")
            
            absoulte_trajectory_errors.append(curr_ate)
            
            if args.show_map:
                if myvis.generate_mesh:
                    o3d.io.write_triangle_mesh(result_dir+seq_name+"_mesh.ply", myvis.mesh)
                else:
                    total_points= o3d.geometry.PointCloud()
                    for ptcld in myvis.pcd_all:
                        total_points += ptcld
                    o3d.io.write_point_cloud(result_dir+seq_name+"_mesh.ply", total_points)
            #writer.grab_frame()

except KeyboardInterrupt:
    pass
finally:
    result_dir = "results_1/"
    df = pd.DataFrame(list(zip(keypoint_extraction_times,matching_times,keypoint_counts, absoulte_trajectory_errors)),
               columns =['KP_Extract', 'Matching', 'KP_count', 'ATE'])
    df.to_csv(result_dir+seq_name+'.csv')
if args.show_map:
    myvis.destroy()