
import os
import numpy as np


in_dir = "/home/geoseg/Work/git_repo/datasets/programmed_models/processed"

out_dir = os.path.join(in_dir, "programmed")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
sub_dir = os.listdir(in_dir)
print(sub_dir)
for seq in sub_dir:
    curr_dir = os.path.join(in_dir, seq)
    print(curr_dir)
    files = os.listdir(curr_dir)
    for file_curr in files:
        curr_dir_file = os.path.join(curr_dir, file_curr)
        out_file = os.path.join(out_dir, seq+"_"+file_curr)
        print(curr_dir_file, out_file)
        os.system("cp "+ curr_dir_file +" "+ out_file)