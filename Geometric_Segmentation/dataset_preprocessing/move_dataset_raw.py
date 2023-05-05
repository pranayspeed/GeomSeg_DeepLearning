
import os
import numpy as np


dataset_dir = "/home/geoseg/Work/git_repo/datasets/programmed_models/raw"
#save_dir = "/home/yashturkar/Work/git_repos/dataset/Geometry_corrected/processed"

seq_list = ['00','01','02','03','04','05','06','07']

in_dir = os.path.join(dataset_dir, "new_street")

list_files = os.listdir(in_dir)


pcd_files =[pcd_file for pcd_file in list_files if "noisy" in pcd_file]

indexs = [int(pcd_file.split("_")[0]) for pcd_file in pcd_files]
#indexs.sort()

sort_indexs = np.argsort(indexs)
print(sort_indexs)

pcd_files = np.array(pcd_files)[sort_indexs]


print(pcd_files)
print(len(pcd_files))

first_split = np.array_split(pcd_files, 8, axis=0)
print(first_split)
print(len(first_split))

for i in range(len(seq_list)):
    seq = seq_list[i]
    files_to_process = first_split[i]
    print(seq)
    curr_dir_to = os.path.join(dataset_dir, seq)
    print(in_dir, curr_dir_to)

    if not os.path.exists(curr_dir_to):
        os.mkdir(curr_dir_to)

    for out_file in files_to_process:
        out_file_pcd = os.path.join(curr_dir_to, out_file)
        in_file_pcd = os.path.join(in_dir, out_file)
        
        cp_cmd = 'cp '+in_file_pcd +' '+ out_file_pcd
        print(cp_cmd)
        os.system(cp_cmd)
    

    
