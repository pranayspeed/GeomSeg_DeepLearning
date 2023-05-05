from repeatability_utils import *


opt_detector = get_repeatiblity_options()
args = opt_detector# .opt# parser.parse_args()




keypoint_extractor_name =  args.method
    
seq_name = args.dataset+"_"+args.sequence_idx + "_" + keypoint_extractor_name
#if args.method =='ramdom' or args.method =='uniform' :
#seq_name+= "_" + str(args.num_icp_points)
###########################################
for_idx=-1
frame_limit=-1


for_idx=-1

curr_transform = np.identity(4)

num_frames = 600
num_frames_to_skip_to_show = 1
# Result saver
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose=curr_transform, 
                             save_gap=args.save_gap,
                             num_frames=num_frames,
                             seq_idx=args.sequence_idx,
                             save_dir=save_dir)

# for save the results as a video
fig_idx = 1
fig = plt.figure(fig_idx)




def execute_odometry(keypoint_extraction_fn, dataset_fn_itr, opt_detector, ResultSaver, stop_frame_id=600, num_frames_to_skip_to_show=5):
    
    
    farthest_sampler = FarthestSampler()
    keypoint_extraction_times=[]
    keypoint_counts = []
    matching_times = []
    absoulte_trajectory_errors=[]
    print(opt_detector.method)  
    fig_idx=1  
    
    try:
        for  i, data in dataset_fn_itr(opt_detector, 0):
            if stop_frame_id !=-1:            
                if i > stop_frame_id:
                    break
                
            anc_pc, anc_sn, anc_node, seq, anc_idx, gt_pose_curr = data
            b=0 # assume single batch
            
            frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
            
            curr_pcd = o3d.geometry.PointCloud()
            curr_pcd.points = o3d.utility.Vector3dVector(frame_pc_np) 
            #for i in range(stop_frame_id):
            #curr_pcd, (anc_pc, anc_sn, anc_node, gt_pose_curr)=dataset_fn(opt_detector, i)

            
            curr_pcd_kps, curr_extract_time = keypoint_extraction_fn(curr_pcd, anc_pc, anc_sn, anc_node, farthest_sampler, opt_detector)
            
            curr_gt_transf = gt_pose_curr 

            keypoint_extraction_times.append(curr_extract_time)
            
            keypoint_counts.append(len(curr_pcd_kps.points))        
            
            if i ==0:
                prev_pcd_kps = copy.deepcopy(curr_pcd_kps)
                prev_pcd = copy.deepcopy(curr_pcd)
                icp_initial = np.eye(4)
                curr_transform = np.eye(4)
                
                #Setting first matching time to zero, as no matching required
                matching_times.append(0)             
                continue            

            # get the start time
            start_time = time.time()

            compute_transform
                
            # calc odometry                
            odom_transform = compute_transform(np.asarray(prev_pcd_kps.points), np.asarray(curr_pcd_kps.points), icp_initial)

            
            # get the end time
            end_time = time.time()        

            # get the execution time
            elapsed_time = 1000 * (end_time - start_time)
            matching_times.append(elapsed_time)

            # update the current (moved) pose 
            curr_transform = np.matmul(curr_transform, odom_transform)

            icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)


            # renewal the prev information 
            prev_pcd_kps = copy.deepcopy(curr_pcd_kps)
            prev_pcd = copy.deepcopy(curr_pcd)
            
            
            # save the ICP odometry pose result (no loop closure)
            for_idx = i
            ResultSaver.saveUnoptimizedPoseGraphResult(curr_transform, curr_gt_transf, for_idx) 
            curr_ate = ResultSaver.compute_ate()
            absoulte_trajectory_errors.append(curr_ate) 
            if(for_idx % num_frames_to_skip_to_show == 0):
                if opt_detector.show_vis: 
                
                    legends={}
                    legends['KP_Extract_Time'] = keypoint_extraction_times[-1]
                    legends['KP_Matching_Time'] = matching_times[-1]
                    legends['KP_count'] = keypoint_counts[-1]
                    
                    result_dir = "results_1/"
                    
                    ResultSaver.vizCurrentTrajectory(fig_idx, seq_name, legends)
                    plt.savefig(result_dir+seq_name + ".png")
                
                #ResultSaver.save_kitti_poses("results_1/", seq_name)   
                
                

            print("Current Frame : ", i, end="\r")
    except KeyboardInterrupt:
        print("Current Frame (Keyboard Interrupt): ", i, )
        pass 
    return keypoint_extraction_times, keypoint_counts, matching_times, absoulte_trajectory_errors        

keypoint_detectors = {
    "harris": extract_keypoints_Harris,
    "iss": extract_keypoints_ISS,
    "sift": extract_keypoints_SIFT,
    "usip": extract_keypoints_usip_odom
    }

print(args.method)
keypoint_extraction_fn = keypoint_detectors[args.method] #extract_keypoints_ISS
dataset_fn_itr = get_kitti_sample_itr
keypoint_extraction_times, keypoint_counts, matching_times, absoulte_trajectory_errors  = execute_odometry(keypoint_extraction_fn, dataset_fn_itr, opt_detector, ResultSaver, stop_frame_id=frame_limit, num_frames_to_skip_to_show=5)


result_dir = "results_1/"
df = pd.DataFrame(list(zip(keypoint_extraction_times,matching_times,keypoint_counts, absoulte_trajectory_errors)),
            columns =['KP_Extract', 'Matching', 'KP_count', 'ATE'])
df.to_csv(result_dir+seq_name+'.csv')

ResultSaver.save_kitti_poses(result_dir, seq_name) 