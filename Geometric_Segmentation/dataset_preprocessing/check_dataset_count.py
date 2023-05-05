import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

import open3d as o3d


from data import data_loaders


print('Loading data...')
#loader, _ = data_loaders(path)



#Namespace(dataset=PosixPath('/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_new'), epochs=50, load='', adam_lr=0.01, batch_size=1, decimation=4, dataset_sampling='geomseg', neighbors=16, scheduler_gamma=0.95, test_dir='test', train_dir='train', val_dir='val', logs_dir=PosixPath('runs'), gpu=device(type='cuda', index=0), name='2023-04-06_15:10', num_workers=0, save_freq=10)

dataset = Path('/home/geoseg/Work/git_repo/datasets/Geometry_street/processed_new')
print(dataset)
dataset_sampling = 'geomseg'

loader_train, loader_test= data_loaders(
    dataset,
    dataset_sampling,
    batch_size=1, #args.batch_size,
    num_workers=1 , #args.num_workers,
    pin_memory=True
)


import torch_geometric.transforms as T




def check_count(loader):

    min_points = 10000000000000

    for points, labels in loader:
        pt_count = points.shape[1]
        print(pt_count, end='\r')
        if min_points> pt_count:
            min_points = pt_count
    print(end='\r')
    print(min_points)
    return min_points

print("Train")
train_min = check_count(loader_train)

print("Test")
test_min = check_count(loader_test)

print("Overall min" ,min(train_min, test_min))
