
import open3d as o3d
import numpy as np

# for sift and harris 3d 
import PCLKeypoint

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

from minisam import *
from utils.ScanContextManager import *
from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP


from KeyPoints import RandomSampleExtractor

from My3dVis import My3DVis


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
parser.add_argument('--sequence_idx', type=str, default='00')

parser.add_argument('--save_gap', type=int, default=300)

parser.add_argument('--mesh_path', type=str, 
                    default='mesh.ply')

parser.add_argument('--show_map', action='store_true')


args = parser.parse_args()


# dataset 
sequence_dir = os.path.join(args.data_base_dir, args.sequence_idx, 'velodyne')
sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
scan_paths = sequence_manager.scan_fullpaths
num_frames = len(scan_paths)

# Pose Graph Manager (for back-end optimization) initialization
PGM = PoseGraphManager()
PGM.addPriorFactor()

# Result saver
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3, 
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
video_name = args.sequence_idx + "_" + str(args.num_icp_points) + ".mp4"
num_frames_to_skip_to_show = 5
num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)

if args.show_map:
    myvis = My3DVis()

    myvis.vis.get_render_option().mesh_show_back_face= True

odom_transform = np.identity(4)
icp_initial = np.eye(4) 
#with writer.saving(fig, video_name, num_frames_to_save): # this video saving part is optional


######################################
#Extractor

keypoint_extractor = RandomSampleExtractor(args.num_icp_points)


#######################################


# @@@ MAIN @@@: data stream
for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):

    # get current information     
    curr_scan_pts = Ptutils.readScan(scan_path) 

    curr_keypts = keypoint_extractor.extract(curr_scan_pts)  
    
    if args.show_map:  
        myvis.set_point_cloud(curr_keypts)   
            
        #Not using PGM.curr_se3 as not doing optimization
        myvis.set_transform(PGM.curr_se3)
        myvis.refresh()    

    # curr_scan_kps3d = PCLKeypoint.keypointHarris3D(curr_scan_pts)
    # myvis.set_point_cloud(curr_scan_kps3d)
    # print("Current Keypoints 3D count ", len(curr_scan_kps3d), end='\r')
        

    # save current node
    PGM.curr_node_idx = for_idx # make start with 0
    
    SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_keypts)
    if(PGM.curr_node_idx == 0):
        PGM.prev_node_idx = PGM.curr_node_idx
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
               
        prev_keypts = copy.deepcopy(curr_keypts)
        
              
        icp_initial = np.eye(4)  
                
        continue

        
    odom_transform = keypoint_extractor.compute_transform(prev_keypts, curr_keypts, icp_initial)


    # update the current (moved) pose 
    PGM.curr_se3 = np.matmul(PGM.curr_se3, odom_transform)
    icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)


    # add the odometry factor to the graph 
    PGM.addOdometryFactor(odom_transform)

    # renewal the prev information 
    PGM.prev_node_idx = PGM.curr_node_idx
    prev_scan_pts = copy.deepcopy(curr_scan_pts)
    prev_keypts = copy.deepcopy(curr_keypts)
    
    current_transf = PGM.curr_se3
    # loop detection and optimize the graph 
    if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0): 
        # 1/ loop detection 
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if(loop_idx == None): # NOT FOUND
            pass
        else:
            print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
            # 2-1/ add the loop factor 
            loop_scan_down_pts = SCM.getPtcloud(loop_idx)
            loop_transform, _, _ = ICP.icp(curr_keypts, loop_scan_down_pts, init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
            PGM.addLoopFactor(loop_transform, loop_idx)

            # 2-2/ graph optimization 
            PGM.optimizePoseGraph()

            # 2-2/ save optimized poses
            ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)
            
            for curr_node_idx in range(PGM.curr_node_idx):
                current_transf = get_optimized_transform(PGM.curr_node_idx, PGM.graph_optimized)
                if args.show_map:
                    myvis.refresh_transform(curr_node_idx, current_transf)

    
    # save the ICP odometry pose result (no loop closure)
    ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
    if(for_idx % num_frames_to_skip_to_show == 0): 
        ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
        
        if args.show_map:
            if myvis.generate_mesh:
                o3d.io.write_triangle_mesh(args.mesh_path, myvis.mesh)
            else:
                total_points= o3d.geometry.PointCloud()
                for ptcld in myvis.pcd_all:
                    total_points += ptcld
                o3d.io.write_point_cloud(args.mesh_path, total_points)
        #writer.grab_frame()
myvis.destroy()