
# python Odometry.py --method sift
# python Odometry.py --method harris
# python Odometry.py --method iss
# python Odometry.py --method usip


evo_ape kitti gt.txt sift.txt -va --plot --plot_mode xz   --save_plot results_ape/sift_ape.png --save_results results_ape/sift.zip
evo_ape kitti gt.txt harris.txt -va --plot --plot_mode xz --save_plot results_ape/harris_ape.png --save_results results_ape/harris.zip
evo_ape kitti gt.txt iss.txt -va --plot --plot_mode xz    --save_plot results_ape/iss_ape.png --save_results results_ape/iss.zip
evo_ape kitti gt.txt usip.txt -va --plot --plot_mode xz   --save_plot results_ape/usip_ape.png --save_results results_ape/usip.zip

evo_traj kitti sift.txt harris.txt iss.txt usip.txt --ref=gt.txt -p --plot_mode=xz --save_plot all_traj.png

evo_rpe kitti gt.txt sift.txt -va --plot --plot_mode xz   --save_plot results_rpe/sift_rpe.png  --save_results results_rpe/sift.zip
evo_rpe kitti gt.txt harris.txt -va --plot --plot_mode xz --save_plot results_rpe/harris_rpe.png  --save_results results_rpe/harris.zip
evo_rpe kitti gt.txt iss.txt -va --plot --plot_mode xz    --save_plot results_rpe/iss_rpe.png  --save_results results_rpe/iss.zip
evo_rpe kitti gt.txt usip.txt -va --plot --plot_mode xz   --save_plot results_rpe/usip_rpe.png  --save_results results_rpe/usip.zip


evo_res results_rpe/*.zip -p --save_table results_rpe/table.csv --save_plot results_rpe/summary_rpe.png 