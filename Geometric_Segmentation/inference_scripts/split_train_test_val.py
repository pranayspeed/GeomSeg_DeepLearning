
import os
import numpy as np



#dataset_dir = "/home/geoseg/Work/git_repo/datasets/programmed_models/processed"
base_dir = "/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/"
#dataset_name = "models_ten"
#dataset_name = "new_street"
dataset_name = "models_ten_v2"

dataset_dir = base_dir+"datasets/"+dataset_name+"/processed"


print("Source Dir:", dataset_dir)
#print("Dest Dir:", save_dir)



seq_list = ['train','test','val']

in_dir = os.path.join(dataset_dir, "all")

list_files = os.listdir(in_dir)


pcd_files = list_files

print(pcd_files)
print(len(pcd_files))

first_split = np.array_split(pcd_files, 10, axis=0)
print(first_split)
print(len(first_split))



for i in range(len(first_split)):
    if i<8:
        seq = seq_list[0]
    elif i==8:
        seq = seq_list[1]
    else:
        seq = seq_list[2]
    #seq = seq_list[i]
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





