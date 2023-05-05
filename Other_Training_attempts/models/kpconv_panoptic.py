
import os
import sys
from omegaconf import OmegaConf
#import pyvista as pv
import torch
import numpy as np




from torch_points3d.applications.kpconv import KPConv
from torch_geometric.data import Batch, Data





from torch_points3d.applications.kpconv import KPConv

from torch_points3d.core.common_modules import MLP, UnaryConv
from torch_points3d.core.common_modules.dense_modules import Conv1D


from torch_points_kernels import region_grow
from torch_scatter import scatter


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


USE_NORMALS=False
IGNORE_LABEL=-1
from torch_points3d.core.common_modules.base_modules import Seq



from torch_points3d.models.panoptic.structures import PanopticLabels, PanopticResults
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.core.losses import offset_loss, instance_iou_loss
from torch_points3d.applications.minkowski import Minkowski

score_params = """
  scorer_unet:
    class: minkowski.Minkowski_Model
    conv_type: "SPARSE"
    define_constants:
      in_feat: 96
    down_conv:
      module_name: ResNetDown
      dimension: 3
      down_conv_nn: [ [ in_feat, 2*in_feat ], [ 2*in_feat, 4*in_feat ] ]
      kernel_size: 3
      stride: 2
      N: 1
    up_conv:
      module_name: ResNetUp
      dimension: 3
      up_conv_nn:
        [ [ 4*in_feat, 2*in_feat ], [ 2*in_feat+ 2*in_feat, in_feat ] ]
      kernel_size: 3
      stride: 2
      N: 1
"""
options = OmegaConf.create(score_params)


USE_NORMALS=False
class PanopticKPConv(torch.nn.Module):
    def __init__(self, cat_to_seg, num_classes=5, remapping_map={            -1:0,  # "unlabeled"
            0: 1, 
            1: 2,  
            2: 3, 
            3: 4,}, device="cuda"):
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
        self.Semantic1 = MultiHeadClassifier(self.unet.output_nc, cat_to_seg)


        self.Semantic = (
            Seq()
            .append(MLP([self.unet.output_nc, self.unet.output_nc], bias=False))
            .append(torch.nn.Linear(self.unet.output_nc, self._num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )


        # instance segmentation
        cluster_voxel_size =  0.05 #option.get("cluster_voxel_size", 0.05)
        if cluster_voxel_size:
            self._voxelizer = GridSampling3D(cluster_voxel_size, quantize_coords=True, mode="mean")
        else:
            self._voxelizer = None

        self.ScorerUnet = Minkowski("unet", input_nc=self.unet.output_nc, num_layers=4, config=options.scorer_unet)

        self.ScorerMLP = MLP([self.unet.output_nc, self.unet.output_nc, self.ScorerUnet.output_nc])
        self.ScorerHead = Seq().append(torch.nn.Linear(self.ScorerUnet.output_nc, 1)).append(torch.nn.Sigmoid())



        self.raw_pos=None
        self.input=None

        self.device=device
        self.epoch = -1

        self._stuff_classes = torch.tensor([IGNORE_LABEL])
        self.cluster_radius_search = 0.05
        #cluster_radius_search: 1.5 * ${data.grid_size}

        self.min_iou_threshold=0.25
        self.max_iou_threshold= 0.75
        self.prepare_epoch=120

        self._scorer_type="MLP" #"MLP"

        self.loss_names = ["loss_seg","semantic_loss"]


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
        self.raw_pos = data.pos.to(self.device)
        self.input = data

        self.labels = PanopticLabels(None, self._remap_labels(data.y), 20, data.instance_labels, None, None)
        #self.labels = self._remap_labels(data.y)
        self.batch = data.batch

        # Forward through unet
        data_features = self.unet(data)
        backbone_features = data_features.x

        category_labels = torch.zeros((backbone_features.shape[0]))
        # for semantic segmentation
        #self.output = self.Semantic1(backbone_features, category_labels )# data.category)
        semantic_logits = self.Semantic(backbone_features)
        offset_logits = None
        
        # for instance segmentation
        # Grouping and scoring
        cluster_scores = None
        all_clusters = None
        cluster_type = None
        if self.epoch == -1 or self.epoch > self.prepare_epoch:  # Active by default
            all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
            if len(all_clusters):
                cluster_scores = self._compute_score(all_clusters, backbone_features, semantic_logits)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            cluster_type=cluster_type,
        )        
        
        # Set loss for the backward pass
        
        #print(torch.unique(self.labels))
        #self.loss_seg = torch.nn.functional.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL)
        return self.output

    def get_spatial_ops(self):
        return self.unet.get_spatial_ops()


    def _cluster(self, semantic_logits, offset_logits):
        """Compute clusters from positions and votes"""
        predicted_labels = torch.max(semantic_logits, 1)[1]
        clusters_pos = region_grow(
            self.raw_pos,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.cluster_radius_search,
        )
        # clusters_votes = region_grow(
        #     self.raw_pos + offset_logits,
        #     predicted_labels,
        #     self.input.batch.to(self.device),
        #     ignore_labels=self._stuff_classes.to(self.device),
        #     radius=self.cluster_radius_search,
        #     nsample=200,
        # )

        all_clusters = clusters_pos# + clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type


    def _compute_score(self, all_clusters, backbone_features, semantic_logits):
        """Score the clusters"""
        if self._scorer_type:
            # Assemble batches
            x = []
            coords = []
            batch = []
            pos = []
            for i, cluster in enumerate(all_clusters):
                x.append(backbone_features[cluster])
                coords.append(self.input.coords[cluster])
                batch.append(i * torch.ones(cluster.shape[0]))
                pos.append(self.input.pos[cluster])
            batch_cluster = Data(
                x=torch.cat(x),
                coords=torch.cat(coords),
                batch=torch.cat(batch),
            )

            # Voxelise if required
            if self._voxelizer:
                batch_cluster.pos = torch.cat(pos)
                batch_cluster = batch_cluster.to(self.device)
                batch_cluster = self._voxelizer(batch_cluster)

            # Score
            batch_cluster = batch_cluster.to("cpu")
            if self._scorer_type == "MLP":
                score_backbone_out = self.ScorerMLP(batch_cluster.x.to(self.device))
                cluster_feats = scatter(
                    score_backbone_out, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                )
            elif self._scorer_type == "encoder":
                score_backbone_out = self.ScorerEncoder(batch_cluster)
                cluster_feats = score_backbone_out.x
            else:
                score_backbone_out = self.ScorerUnet(batch_cluster)
                cluster_feats = scatter(
                    score_backbone_out.x, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                )
            cluster_scores = self.ScorerHead(cluster_feats).squeeze(-1)
        else:
            # Use semantic certainty as cluster confidence
            with torch.no_grad():
                cluster_semantic = []
                batch = []
                for i, cluster in enumerate(all_clusters):
                    cluster_semantic.append(semantic_logits[cluster, :])
                    batch.append(i * torch.ones(cluster.shape[0]))
                cluster_semantic = torch.cat(cluster_semantic)
                batch = torch.cat(batch)
                cluster_semantic = scatter(cluster_semantic, batch.long().to(self.device), dim=0, reduce="mean")
                cluster_scores = torch.max(cluster_semantic, 1)[0]
        return cluster_scores

    def _compute_loss(self):
        # Semantic loss
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, self.labels.y, ignore_index=IGNORE_LABEL
        )
        #self.loss = self.opt.loss_weights.semantic * self.semantic_loss
        self.loss_seg = self.semantic_loss

        # # Offset loss
        # self.input.instance_mask = self.input.instance_mask.to(self.device)
        # self.input.vote_label = self.input.vote_label.to(self.device)
        # offset_losses = offset_loss(
        #     self.output.offset_logits[self.input.instance_mask],
        #     self.input.vote_label[self.input.instance_mask],
        #     torch.sum(self.input.instance_mask),
        # )
        # for loss_name, loss in offset_losses.items():
        #     setattr(self, loss_name, loss)
        #     self.loss += self.opt.loss_weights[loss_name] * loss

        # Score loss
        if self.output.cluster_scores is not None and self._scorer_type:
            self.score_loss = instance_iou_loss(
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
                min_iou_threshold=self.min_iou_threshold,
                max_iou_threshold=self.max_iou_threshold,
            )
            self.loss_seg += self.score_loss# * self.opt.loss_weights["score_loss"]


    def backward(self):
         self._compute_loss()
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
