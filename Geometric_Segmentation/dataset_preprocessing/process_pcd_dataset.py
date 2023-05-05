import os
import open3d
import numpy as np

import torch
from torch_geometric.data import Data

from pathlib import Path
# R = 0
# G = [0.0, 0.1, 0.2, 0.3, 0.4,0.9]
# B=[unique 0-> 1] for instance

color_map = {
"Plane":   0 ,
"Sphere":  0.1,
"Cylinder":0.2,
"Cone":    0.3,
"Cube":    0.4,
"Torus":  0.5,
"other":   0.9,
    }


LABELS = {
    -1: "unlabeled",
    0: "Plane",
    1: "Sphere",
    2: "Cylinder",
    3: "Cone",
    4: "Cube",
    5: "Torus", 
}

map_G_channel_to_shape_index = {
    0.0:0,
    0.1:1,
    0.2:2,
    0.3:3,
    0.4:4,
    0.5:0,
    0.9:0,
}

def load_pcd_to_torch_data(pcd_file, out_file, channel_color):
  CLASS_CHANNEL=channel_color
  pcd = open3d.io.read_point_cloud(pcd_file)
  
#   open3d.visualization.draw_geometries([pcd])
#   return None
  scan = np.asarray(pcd.points, dtype=np.float32)
  #print(scan.shape)
  colors =  np.asarray(pcd.colors, dtype=np.float32)
  
  y = np.array(colors[:, CLASS_CHANNEL]).round(1)*10
  print("1- unique class :", np.unique(y))
  if len(np.unique(y))<2:
    print("Not enough objects found in the scan: ", pcd_file)
    print("Discarding....")
    return
#   def mp(entry):
#     return map_G_channel_to_shape_index[entry] if entry in map_G_channel_to_shape_index else entry
#     mp = np.vectorize(mp)
#   y = mp(y)

  y = np.array([map_G_channel_to_shape_index[c/10] for c in y])
  print("2 - unique class :", np.unique(y))
  ins_label = np.array(colors[:, 2])
  #print("class labels:", y.shape, np.unique(y))
  #print("class labels:", ins_label.shape, len(np.unique(ins_label)))
  #np.unique(ins_label)
  ins_label = np.unique(ins_label, return_inverse = True)[1]  
  print("labels: ", np.unique(ins_label), np.min(ins_label), np.max(ins_label))
  # if np.max(ins_label)<5:
  #   print("Not enough objects found in the scan: ", pcd_file)
  #   print("Discarding....")
  #   return
  #return None

  data = Data(pos=torch.tensor(scan[:,:]), x=torch.ones((len(scan), 1)),)
  data.y = torch.tensor(y[:]).long()
  #[TODO] Pranay: FIX the instance label by regenerating  models with unique labels
  
  #data.instance_labels = torch.tensor(ins_label[:]).long()
  #out_file = "/home/yashturkar/00.pt"

  #exit(0)
  torch.save(data, out_file)
  return data
      
            
def save_custom_dataset_from_pcd_dir(dataset_dir, save_dir, channel_color):
  #dir_list = os.listdir(dataset_dir)

  paths = list(Path(dataset_dir).glob(f'**/*.pcd'))

  curr_save_dir = os.path.join(save_dir, "all")
  if not os.path.exists(curr_save_dir):
    os.mkdir(curr_save_dir)
  for pcd_file in paths:
    file_name = os.path.basename(pcd_file).split(".")[0]
    #print(file_name)
    #print(pcd_file)

    if ".pcd" in str(pcd_file) and "noisy" not in str(pcd_file):
      #par_dir = os.pardir(pcd_file)
      out_file = curr_save_dir+"/"+file_name+".pt"

      print(pcd_file, out_file)
      if os.path.exists(out_file):
        print("Exists: ", out_file)
        continue
      load_pcd_to_torch_data(str(pcd_file), out_file, channel_color)   


#dataset_dir = "/home/yashturkar/Work/git_repos/dataset/Geometry_corrected/raw"
#save_dir = "/home/yashturkar/Work/git_repos/dataset/Geometry_corrected/processed"
#dataset_dir = "/home/geoseg/Work/git_repo/datasets/Geometry_street/raw"
#save_dir = "/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_4cls"

base_dir = "/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/"

#dataset_name = "models_ten"
#dataset_name = "new_street"
dataset_name = "models_ten_v2"
dataset_dir = base_dir+"datasets/"+dataset_name+"/raw"
save_dir = base_dir+"datasets/"+dataset_name+"/processed"

channel_color_file = base_dir+"datasets/"+dataset_name+ "/channel_color.txt"

if not os.path.exists(channel_color_file):
    print("File not found:", channel_color_file)
    exit()
with open(channel_color_file, 'r') as f:
    channel_color = int(f.read().strip())

#channel_color = 
print(channel_color)



print("Source Dir:", dataset_dir)
print("Dest Dir:", save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_custom_dataset_from_pcd_dir(dataset_dir, save_dir, channel_color)
