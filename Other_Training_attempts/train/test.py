











#torch segmentation CNN model using pointclouds

from GeometrySegmentation import PointCloudSegmentation, PointCloudDataset, PointNetSegmentation

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F






# Create the training dataset

bagsfit_files_path = "/home/pranayspeed/Downloads/TEST-20s/"
test_dataset = PointCloudDataset(bagsfit_files_path)

batch_size=1

# Create the data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


number_of_classes = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = PointCloudSegmentation(num_shapes=number_of_classes)
model_file = "/home/pranayspeed/Downloads/point_cloud_segmentation.pt"
model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))


model = model.to(device)

dataiter = iter(test_loader)

input_points, labels = next(dataiter)

model.eval()


input_points_1 = input_points.view(batch_size, 3, -1)

input_points_1 = input_points_1.to(device)

probs = model(input_points_1)

_, labels = torch.max(probs, dim=1)
print(labels[:20])


import numpy as np

import open3d

def visualize_test_case(scan_data, cls):
    color_map = {
    4: [0, 0, 0],
    0: [0, 0, 255],
    1: [0, 255, 0],
    2: [255, 0, 0],
    3: [0, 255, 255],
    5: [255, 255, 0],
    }
    color_values = np.array([color_map[x] for x in cls])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scan_data)
    pcd.colors = open3d.utility.Vector3dVector(color_values)
    open3d.visualization.draw_geometries([pcd])

print(input_points.shape)
input_data = input_points[0].cpu().numpy()
cls = labels.cpu().numpy()
print(input_data.shape, cls.shape)
visualize_test_case(input_data, cls)