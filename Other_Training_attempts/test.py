
import os
import torch
from models.kpconv import PartSegKPConv

from omegaconf import OmegaConf
import numpy as np

from points3d.dataset.bagsfit import BagsfitDataset
import time

from tqdm.auto import tqdm


import open3d

CONE = [0, 0, 255]
CYLINDER = [0, 255, 0]
SPHERE = [255, 0, 0]

OTHER = [0,0,0]#[255, 255, 255]
def visualize_test_case_cuda(scan_data_cuda, cls_cuda):
    color_map = {
    
    0: OTHER,
    1: OTHER,
    2: SPHERE,
    3: CYLINDER,
    4: CONE,
    -1: OTHER,
    }
    scan_data = np.asarray(scan_data_cuda.cpu().numpy())
    cls = cls_cuda.cpu().numpy()
    print(cls.shape, scan_data.shape)
    color_values = np.array([color_map[x] for x in cls])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scan_data)
    pcd.colors = open3d.utility.Vector3dVector(color_values)
    open3d.visualization.draw_geometries([pcd])


def visualize_test_case_cuda_combined(scan_data_cuda, cls_cuda_pred, cls_cuda_gt):
    color_map = {
    
    0: OTHER,
    1: OTHER,
    2: SPHERE,
    3: CYLINDER,
    4: CONE,
    -1: OTHER,
    }
    scan_data = np.asarray(scan_data_cuda.cpu().numpy())
    cls = cls_cuda_pred.cpu().numpy()
    print(np.unique(cls))
    color_values = np.array([color_map[x] for x in cls])
    pcd_pred = open3d.geometry.PointCloud()
    pcd_pred.points = open3d.utility.Vector3dVector(scan_data)
    pcd_pred.colors = open3d.utility.Vector3dVector(color_values)



    cls = cls_cuda_gt.cpu().numpy()
    print(np.unique(cls))
    color_values = np.array([color_map[x] for x in cls])

    pcd_gt = open3d.geometry.PointCloud()
    pcd_gt.points = open3d.utility.Vector3dVector(scan_data)
    pcd_gt.colors = open3d.utility.Vector3dVector(color_values)

    extents = pcd_gt.get_axis_aligned_bounding_box().get_extent()
    req_trans = np.max(extents)
    print(extents, req_trans)

    pcd_gt.translate([req_trans,0,0])

    open3d.visualization.draw_geometries([pcd_pred, pcd_gt])

shapenet_yaml = """
# @package data
class: bagsfit.BagsfitDataset
dataset_name: "Bagsfit"
task: segmentation
dataroot: /home/pranayspeed/Work/git_repo/datasets
grid_size: 0.1
process_workers: 8
apply_rotation: True
mode: "last"


train_transform:
  - transform: ElasticDistortion
  - transform: Random3AxisRotation
    params:
      apply_rotation: True
      rot_x: 2
      rot_y: 2
      rot_z: 180
  - transform: RandomScaleAnisotropic
    params:
      scales: [0.9, 1.1]
  - transform: GridSampling3D
    params:
      size: 0.1
      quantize_coords: True
      mode: "last"

val_transform:
  - transform: GridSampling3D
    params:
      size: 0.1
      quantize_coords: True
      mode: "last"

test_transform:
  - transform: GridSampling3D
    params:
      size: 0.01
      quantize_coords: True
      mode: "last"


"""


params = OmegaConf.create(shapenet_yaml)


dataset = BagsfitDataset(params)




#############################################################

cat_to_seg = {'Shape':[0,1,2,3,4]}
remapping_map = {            -1:-1,  # "unlabeled"
            0: 0, 
            1: 1,  
            2: 2, 
            3: 3,
            4: 4
            }

inverse_remap = {v: k for k, v in remapping_map.items()}

values_list = np.unique(list(remapping_map.values()))
print(values_list)
num_classes = len(np.unique(values_list))-1
print(num_classes)
model_name= "bagsfit_5"
######################################
# cat_to_seg = {'Shape':[0,1]}
# num_classes=2
# remapping_map = {            -1:-1,  # "unlabeled"
#             0: -1, 
#             1: -1,  
#             2: 0, 
#             3: 0,
#             4: 1
#             }
#inverse_remap = {v: k for k, v in remapping_map.items()}
# values_list = np.unique(list(remapping_map.values()))
# print(values_list)
# num_classes = len(np.unique(values_list))-1
# print(num_classes)
# model_name="bagsfit_2"
######################################
# Create a model
model = PartSegKPConv(cat_to_seg, num_classes=num_classes, remapping_map=remapping_map)



model_checkpoint =  "trained_models/" + model_name + ".pt"


if os.path.exists(model_checkpoint):
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

NUM_WORKERS = 1
BATCH_SIZE = 1
dataset.create_dataloaders(
    model,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS, 
    shuffle=False, 
    precompute_multi_scale=False 
    )

#sample = next(iter(dataset.train_dataloader))

def _inverse_remap_labels(semantic_label, LEARNING_MAP_INV):
    new_labels = semantic_label.clone()
    for source, target in LEARNING_MAP_INV.items():
        mask = semantic_label == source
        new_labels[mask] = target

    return new_labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_loader = dataset.test_dataloaders[0]
iter_data_time = time.time()
with tqdm(test_loader) as tq_test_loader:
    for i, data in enumerate(tq_test_loader):
        t_data = time.time() - iter_data_time
        iter_start_time = time.time()
        data.to(device)
        output = model.forward(data)   

        pred = torch.argmax(output, dim=1)
        #pred_cpu = pred.cpu().numpy()

        

        print(_inverse_remap_labels(pred, inverse_remap)[:20])
        print(data.y[:20])

        #visualize_test_case_cuda(data.pos, _inverse_remap_labels(pred, inverse_remap))
        #visualize_test_case_cuda(data.pos, data.y)
        print(torch.unique(_inverse_remap_labels(pred, inverse_remap)))
        print(torch.unique(data.y))
        visualize_test_case_cuda_combined(data.pos, _inverse_remap_labels(pred, inverse_remap), data.y)