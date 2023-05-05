import os
import open3d
import numpy as np

import torch
#from torch_geometric.data import Data

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


PLANE = [0, 125, 125]
CONE = [0, 0, 255]
CYLINDER = [0, 255, 0]
SPHERE = [255, 0, 0]

OTHER = [0,0,0]#[255, 255, 255]
color_map = {
    
    0: PLANE,
    1: SPHERE,
    2: CYLINDER,
    3: CONE,
    4: OTHER,
    }





def load_pcd_to_torch_data(pcd_file, class_channel):
  CLASS_CHANNEL=class_channel
  pcd = open3d.io.read_point_cloud(pcd_file)
  
#   open3d.visualization.draw_geometries([pcd])
#   return None
  scan = np.asarray(pcd.points, dtype=np.float32)
  #print(scan.shape)
  colors =  np.asarray(pcd.colors, dtype=np.float32)
  
  y = np.array(colors[:, CLASS_CHANNEL]).round(1)*10
  print("1- unique class :", np.unique(y))

#   def mp(entry):
#     return map_G_channel_to_shape_index[entry] if entry in map_G_channel_to_shape_index else entry
#     mp = np.vectorize(mp)
#   y = mp(y)

  y = np.array([map_G_channel_to_shape_index[c/10] for c in y])

  color_values = np.array([color_map[x] for x in y])
  pcd.colors = open3d.utility.Vector3dVector(color_values)

  new_colors = np.zeros((colors.shape[0], 3))

  open3d.visualization.draw_geometries([pcd])
      
            
def play_custom_dataset_from_pcd_dir(dataset_dir, class_channel, file_prefix="model_8"):
  #dir_list = os.listdir(dataset_dir)

  paths = list(Path(dataset_dir).glob(f'**/*.pcd'))
  for pcd_file in paths:
    file_name = os.path.basename(pcd_file).split(".")[0]
    #print(file_name)
    #print(pcd_file)

    if ".pcd" in str(pcd_file) and "noisy" not in str(pcd_file) and file_prefix in str(pcd_file):
      print(pcd_file)
      load_pcd_to_torch_data(str(pcd_file), class_channel)   


#dataset_dir = "/home/yashturkar/Work/git_repos/dataset/Geometry_corrected/raw"
#save_dir = "/home/yashturkar/Work/git_repos/dataset/Geometry_corrected/processed"
#dataset_dir = "/home/geoseg/Work/git_repo/datasets/Geometry_street/raw"
#save_dir = "/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_4cls"

base_dir = "/mnt/share/nas/lidar-research/Geometric_Seg_Pranay/"

#dataset_name = "models_ten" # very bad
#dataset_name = "new_street"
dataset_name = "models_ten_v2"
dataset_dir = base_dir+"datasets/"+dataset_name+"/raw"
#save_dir = base_dir+"datasets/"+dataset_name+"/processed"

print("Source Dir:", dataset_dir)

channel_color_file = base_dir+"datasets/"+dataset_name+ "/channel_color.txt"

if not os.path.exists(channel_color_file):
    print("File not found:", channel_color_file)
    exit()
with open(channel_color_file, 'r') as f:
    channel_color = int(f.read().strip())

#channel_color = 
print(channel_color)
play_custom_dataset_from_pcd_dir(dataset_dir, channel_color)
