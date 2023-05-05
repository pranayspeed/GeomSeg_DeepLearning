











#torch segmentation CNN model using pointclouds

from GeometrySegmentation import PointCloudSegmentation, PointCloudDataset, PointNetSegmentation, PointCloudNormalDataset

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F

import os
import time
import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn


import sys
sys.path.append("../")

from bagsfit import BAGsFit_resnet101, WeightedMultiLabelSigmoidLoss, check_gpu


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
    print(cls.shape)
    cls = cls.reshape(-1)
    color_values = np.array([color_map[x] for x in cls])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scan_data)
    pcd.colors = open3d.utility.Vector3dVector(color_values)
    open3d.visualization.draw_geometries([pcd])

def visualize_test_case_cuda(scan_data_cuda, cls_cuda):
    color_map = {
    4: [0, 0, 0],
    0: [0, 0, 255],
    1: [0, 255, 0],
    2: [255, 0, 0],
    3: [0, 255, 255],
    5: [255, 255, 0],
    -1: [255, 255, 255],
    }
    scan_data = scan_data_cuda[0].cpu().numpy()
    cls = cls_cuda[0].cpu().numpy()
    print(cls.shape, scan_data.shape)
    color_values = np.array([color_map[x] for x in cls])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(scan_data)
    pcd.colors = open3d.utility.Vector3dVector(color_values)
    open3d.visualization.draw_geometries([pcd])


def main():
    number_of_classes = 6
    model = BAGsFit_resnet101(number_of_classes)

    # Create the training dataset

    #bagsfit_files_path = "/home/pranayspeed/Downloads/TRAIN-20s-normals/"
    bagsfit_files_path = "/home/pranayspeed/Downloads/TRAIN-20s/"
    #test_dataset = PointCloudNormalDataset(bagsfit_files_path)
    test_dataset = PointCloudDataset(bagsfit_files_path)

    batch_size=1

    # Create the data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    #model = PointCloudSegmentation(num_shapes=number_of_classes)
    #model_file = "point_cloud_segmentation_bagsfit.pt"
    #model_file = "point_cloud_segmentation_bagsfit_pointsonly.pt"
    model_file = "/home/pranayspeed/Downloads/point_cloud_segmentation_bagsfit_pointsonly_cuda.pt"
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))


    model = model.to(device)

    dataiter = iter(test_loader)

    input_points, labels = next(dataiter)

    print(input_points.cpu().numpy().shape, labels.cpu().numpy().shape)
    visualize_test_case_cuda(input_points, labels)


    model.eval()

    is_cuda = 0 if torch.cuda.is_available() else None  

    perm = torch.randperm(input_points.size(1))
    idx = perm[:440*440]
    samples_input = input_points[:,idx]
    labels = labels[:,idx]


    input_points1 = samples_input.view(batch_size, 3, 440,-1)
    # Input for Image CNN.
    img_var = check_gpu(is_cuda, input_points1) # BS X 3 X H X W

    labels[labels==-1]=5
    target_var = check_gpu(is_cuda, labels) # BS X H X W X NUM_CLASSES



    # input_points_1 = input_points.view(batch_size, 3, -1)

    # input_points_1 = input_points_1.to(device)

    prob_classes, _ = model(img_var)



    prob_classes = prob_classes.reshape(1,number_of_classes,-1)
    print(prob_classes.shape)
    
    print(prob_classes[0,:20])
    prob_classes = torch.argmax(prob_classes, dim=1)
    #prob_classes=torch.max(prob_classes.data,1)
    
    #prob_classes = torch.softmax(prob_classes, dim=1)

    print("this : ",prob_classes[:20])
    #prob_classes = torch.argmax(prob_classes, dim=1)
    print(labels[:20])
    print(prob_classes[:20])

    print(prob_classes.shape)
    visualize_test_case_cuda(samples_input, prob_classes.detach())

    # print(input_points.shape)
    # input_data = input_points[0].view(3, -1)
    # input_data = input_data.cpu().numpy()
    # cls = labels.cpu().numpy()
    # print(input_data.shape, cls.shape)

    # input_data = input_data.T
    # visualize_test_case(input_data, cls)


if __name__ == "__main__":
    main()