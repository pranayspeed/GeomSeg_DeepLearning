

import os
import sys
from omegaconf import OmegaConf
#import pyvista as pv
import torch
import numpy as np


DIR = "" # Replace with your root directory, the data will go in DIR/data.

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
      size: 0.1
      quantize_coords: True
      mode: "last"


"""# % (os.path.join(DIR,"data"), USE_NORMALS) 

from omegaconf import OmegaConf
params = OmegaConf.create(shapenet_yaml)

from points3d.dataset.bagsfit import BagsfitDataset

dataset = BagsfitDataset(params)


from torch_points3d.models.segmentation.randlanet import RandLANetSeg





from models.kpconv import PartSegKPConv


# KPConv is implemented with PARTIAL_DENSE format. Therefore, data need an attribute batch containing the indice for each point

#############################################################

cat_to_seg = {'Shape':[0,1,2,3,4]}
remapping_map = {            -1:-1,  # "unlabeled"
            0: 0, 
            1: 1,  
            2: 2, 
            3: 3,
            4: 4
            }

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
# values_list = np.unique(list(remapping_map.values()))
# print(values_list)
# num_classes = len(np.unique(values_list))-1
# print(num_classes)
# model_name= "bagsfit_2"
########################################
# Create a model
model = PartSegKPConv(cat_to_seg, num_classes=num_classes, remapping_map=remapping_map)


# model = KPConv(
#     architecture="unet",  # Could be unet here to perform segmentation
#     input_nc=input_nc,  # KPconv is particular. Pos aren't features. It needs a tensor of ones + any features available as rgb or intensity
#     output_nc=num_classes,
#     num_layers=4,
# )

NUM_WORKERS = 4
BATCH_SIZE = 4
dataset.create_dataloaders(
    model,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS, 
    shuffle=True, 
    precompute_multi_scale=False 
    )


sample = next(iter(dataset.train_dataloader))

print(sample.batch.size(0), sample.pos.size(0), sample.y.shape)

#sample.keys


from tqdm.auto import tqdm
import time
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

class Trainer:
    def __init__(self,model, dataset, model_name = "bagsfit", num_epoch = 50, device=torch.device('cuda'), resume=True):
        self.num_epoch = num_epoch
        self._model = model
        self._dataset=dataset
        self.device = device

        self.model_name = model_name
        self.path = "trained_models/" + self.model_name + ".pt"

        self.checkpoint =None
        if resume:
            if os.path.exists(self.path):
                checkpoint = torch.load(self.path)
                self._model.load_state_dict(checkpoint['state_dict'])
                #self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.checkpoint = checkpoint
                print("Model loaded from checkpoint")

    def fit(self):
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.tracker = self._dataset.get_tracker(False, True)

        for i in range(self.num_epoch):
            print("=========== EPOCH %i ===========" % i)
            time.sleep(0.5)
            self.train_epoch()
            checkpoint = { 'state_dict': self._model.state_dict(),'optimizer' :self.optimizer.state_dict()}
            torch.save(checkpoint, self.path)            
            metrics = self.tracker.publish(i)
            self.test_epoch()
            self.tracker.publish(i)



    def train_epoch(self):
        self._model.to(self.device)
        self._model.train()
        self.tracker.reset("train")
        train_loader = self._dataset.train_dataloader
        iter_data_time = time.time()
        with tqdm(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self.optimizer.zero_grad()
                data.to(self.device)
                self._model.forward(data)
                self._model.backward()
                
                self.optimizer.step()
                if i % 10 == 0:
                    self.tracker.track(self._model)
                if i % 100 == 0:
                    checkpoint = { 'state_dict': self._model.state_dict(),'optimizer' :self.optimizer.state_dict()}
                    torch.save(checkpoint, self.path)  

                tq_train_loader.set_postfix(
                    **self.tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                )
                iter_data_time = time.time()
                ##{TODO: COMMENT THIS OUT}##
                #break

    def test_epoch(self):
        self._model.to(self.device)
        self._model.eval()
        self.tracker.reset("test")
        test_loader = self._dataset.test_dataloaders[0]
        iter_data_time = time.time()
        with tqdm(test_loader) as tq_test_loader:
            for i, data in enumerate(tq_test_loader):
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                data.to(self.device)
                self._model.forward(data)           
                self.tracker.track(self._model)

                tq_test_loader.set_postfix(
                    **self.tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                )
                iter_data_time = time.time()
                ##{TODO: COMMENT THIS OUT}##
                #break

trainer = Trainer(model, dataset,model_name=model_name, device=torch.device('cpu'))

trainer.fit()




