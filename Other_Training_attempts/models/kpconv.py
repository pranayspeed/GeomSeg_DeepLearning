
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



from torch_points3d.applications.kpconv import KPConv
from torch_geometric.data import Batch, Data

from torch_points3d.models.segmentation.kpconv import KPConvPaper

from torch_points3d.models.segmentation.randlanet import RandLANetSeg


from torch_points3d.applications.kpconv import KPConv

from torch_points3d.core.common_modules import MLP, UnaryConv
from torch_points3d.core.common_modules.dense_modules import Conv1D

class MultiHeadClassifier(torch.nn.Module):
    """ Allows segregated segmentation in case the category of an object is known. 
    This is the case in ShapeNet for example.

    Parameters
    ----------
    in_features -
        size of the input channel
    cat_to_seg
        category to segment maps for example:
        {
            'Airplane': [0,1,2],
            'Table': [3,4]
        }

    """

    def __init__(self, in_features, cat_to_seg, dropout_proba=0.5, bn_momentum=0.1):
        super().__init__()
        self._cat_to_seg = {}
        self._num_categories = len(cat_to_seg)
        self._max_seg_count = 0
        self._max_seg = 0
        self._shifts = torch.zeros((self._num_categories,), dtype=torch.long)
        for i, seg in enumerate(cat_to_seg.values()):
            self._max_seg_count = max(self._max_seg_count, len(seg))
            self._max_seg = max(self._max_seg, max(seg))
            self._shifts[i] = min(seg)
            self._cat_to_seg[i] = seg

        self.channel_rasing = MLP(
            [in_features, self._num_categories * in_features], bn_momentum=bn_momentum, bias=False
        )
        if dropout_proba:
            self.channel_rasing.add_module("Dropout", torch.nn.Dropout(p=dropout_proba))

        self.classifier = UnaryConv((self._num_categories, in_features, self._max_seg_count))
        self._bias = torch.nn.Parameter(torch.zeros(self._max_seg_count,))

    def forward(self, features, category_labels, **kwargs):
        assert features.dim() == 2
        self._shifts = self._shifts.to(features.device)
        in_dim = features.shape[-1]
        features = self.channel_rasing(features)
        features = features.reshape((-1, self._num_categories, in_dim))
        features = features.transpose(0, 1)  # [num_categories, num_points, in_dim]
        features = self.classifier(features) + self._bias  # [num_categories, num_points, max_seg]

        ind = category_labels.unsqueeze(-1).repeat(1, 1, features.shape[-1]).long()

        logits = features.gather(0, ind).squeeze(0)
        softmax = torch.nn.functional.log_softmax(logits, dim=-1)

        output = torch.zeros(logits.shape[0], self._max_seg + 1).to(features.device)
        cats_in_batch = torch.unique(category_labels)
        for cat in cats_in_batch:
            cat_mask = category_labels == cat
            seg_indices = self._cat_to_seg[cat.item()]
            probs = softmax[cat_mask, : len(seg_indices)]
            output[cat_mask, seg_indices[0] : seg_indices[-1] + 1] = probs
        
        return output



class SingleHeadClassifier(torch.nn.Module):
    """ Allows segregated segmentation in case the category of an object is known. 
    This is the case in ShapeNet for example.

    Parameters
    ----------
    in_features -
        size of the input channel
    cat_to_seg
        category to segment maps for example:
        {
            'Airplane': [0,1,2],
            'Table': [3,4]
        }

    """

    def __init__(self, in_features, cat_to_seg, dropout_proba=0.5, bn_momentum=0.1):
        super().__init__()
        self._cat_to_seg = {}
        self._num_categories = len(cat_to_seg)
        self._max_seg_count = 0
        self._max_seg = 0
        self._shifts = torch.zeros((self._num_categories,), dtype=torch.long)
        for i, seg in enumerate(cat_to_seg.values()):
            self._max_seg_count = max(self._max_seg_count, len(seg))
            self._max_seg = max(self._max_seg, max(seg))
            self._shifts[i] = min(seg)
            self._cat_to_seg[i] = seg

        self.channel_rasing = MLP(
            [in_features, self._num_categories * in_features], bn_momentum=bn_momentum, bias=False
        )
        if dropout_proba:
            self.channel_rasing.add_module("Dropout", torch.nn.Dropout(p=dropout_proba))

        self.classifier = UnaryConv((self._num_categories, in_features, self._max_seg_count))
        self._bias = torch.nn.Parameter(torch.zeros(self._max_seg_count,))

    def forward(self, features, category_labels, **kwargs):
        assert features.dim() == 2
        self._shifts = self._shifts.to(features.device)
        in_dim = features.shape[-1]
        features = self.channel_rasing(features)
        features = features.reshape((-1, self._num_categories, in_dim))
        features = features.transpose(0, 1)  # [num_categories, num_points, in_dim]
        features = self.classifier(features) + self._bias  # [num_categories, num_points, max_seg]


        #ind = category_labels.unsqueeze(-1).repeat(1, 1, features.shape[-1]).long()

        #logits = features.gather(0, ind).squeeze(0)
        output = torch.nn.functional.log_softmax(features, dim=-1)

        # #output = torch.zeros(logits.shape[0], self._max_seg + 1).to(features.device)
        # cats_in_batch = torch.unique(category_labels)
        # for cat in cats_in_batch:
        #     cat_mask = category_labels == cat
        #     seg_indices = self._cat_to_seg[cat.item()]
        #     probs = softmax[cat_mask, : len(seg_indices)]
        #     output[cat_mask, seg_indices[0] : seg_indices[-1] + 1] = probs
        
        return output


USE_NORMALS=False
IGNORE_LABEL=-1
from torch_points3d.core.common_modules.base_modules import Seq

class PartSegKPConv(torch.nn.Module):
    def __init__(self, cat_to_seg, num_classes=5, remapping_map={            -1:0,  # "unlabeled"
            0: 1, 
            1: 2,  
            2: 3, 
            3: 4,}):
        super().__init__()
        self.unet = KPConv(
            architecture="unet", 
            input_nc=USE_NORMALS * 3, 
            num_layers=4, 
            in_grid_size=0.02
            )
        self.REMAPPING_MAP = remapping_map
        self._num_categories = 0
        self._num_classes  = num_classes
        self.classifier = MultiHeadClassifier(self.unet.output_nc, cat_to_seg)
        # Last MLP
        last_mlp_opt ={"nn": [128, 128],
        "dropout": 0.5}

        self.FC_layer = Seq()
        last_mlp_opt["nn"][0] += self._num_categories
        for i in range(1, len(last_mlp_opt["nn"])):
            self.FC_layer.append(Conv1D(last_mlp_opt["nn"][i - 1], last_mlp_opt["nn"][i], bn=True, bias=False))
        if last_mlp_opt["dropout"]:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt["dropout"]))

        self.FC_layer.append(Conv1D(last_mlp_opt["nn"][-1], self._num_classes, activation=None, bias=True, bn=False))




    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.unet.conv_type
    
    def get_batch(self):
        return self.batch
    
    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output
    
    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels
    
    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss_seg": float(self.loss_seg)}

    def forward(self, data):


        #self.labels = data.y
        self.labels = self._remap_labels(data.y)
        self.batch = data.batch
        #print(self.batch.shape)
        # Forward through unet and classifier
        data_features = self.unet(data)

        last_feature = data_features.x

        #self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))
        
        category_labels = torch.zeros((data_features.x.shape[0]))
        self.output = self.classifier(data_features.x, category_labels )# data.category)
        
        #print(self.output.shape, self.labels.shape)
         # Set loss for the backward pass
        
        #print(torch.unique(self.labels))
        self.loss_seg = torch.nn.functional.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)
        return self.output

    def get_spatial_ops(self):
        return self.unet.get_spatial_ops()
        
    def backward(self):
         self.loss_seg.backward() 


    def _remap_labels(self, semantic_label):
        """ Remaps labels to [0 ; num_labels -1]. Can be overriden.
        """
        # REMAPPING_MAP = {
        #     -1:0,  # "unlabeled"
        #     0: 1, 
        #     1: 2,  
        #     2: 3, 
        #     3: 4,
        # }
        new_labels = semantic_label.clone()
        for source, target in self.REMAPPING_MAP.items():
            mask = semantic_label == source
            new_labels[mask] = target

        return new_labels
