import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PointCloudSegmentation(nn.Module):
    def __init__(self, num_shapes):
        super(PointCloudSegmentation, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(256, num_shapes)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print("conv3", x.shape)
        #x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        x = self.fc(x)

        probs = F.softmax(x, dim=1)
        # print("probs", probs.shape)
        # print("probs[0]", probs[0])
        _, labels = torch.max(probs, dim=1)
        #print(labels.shape, x.shape)
        #print("labels[0:10]", labels[0:10])
        return labels
    


############################################################

class PointNetSegmentation(nn.Module):
    def __init__(self, num_shapes):
        super(PointNetSegmentation, self).__init__()

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc31 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 2048)
        self.conv1 = nn.Conv1d(2048, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(2048, num_shapes, 1)
        self.bn5 = nn.BatchNorm1d(num_shapes)


        self.conv6 = nn.Conv1d(1024, num_shapes, 1)

        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # x is a batch of 3D point clouds of shape (B, N, 3)

        x = x.permute(0, 2, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc31(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))

        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        print(x.shape)
        x = self.conv6(x)
        print(x.shape)
        x = x.permute(0, 2, 1)

        x = self.up1(x)
        print(x.shape)
        return x




#####################UTILS #########################

def get_prim_data(bagsfit_files_path, sequence, test_case):
    full_path_prim = bagsfit_files_path + f"{sequence:03d}" + "/" +f"{test_case:05d}" +".prim"
    prim_data = []
    with open(full_path_prim, "r") as data:        
        for line in data:
            curr_prim = {}
            curr_data = line.split(" ")
            #print(curr_data[0] )
            curr_prim['type'] = curr_data[0] 
            if curr_prim['type'] == "Plane":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['norm'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
            elif curr_prim['type'] == "Sphere":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['radius'] = float(curr_data[4])
            elif curr_prim['type'] == "Cylinder":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['axis'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
                curr_prim['radius'] = float(curr_data[7])
            elif curr_prim['type'] == "Cone":
                curr_prim['point'] = np.array([float(curr_data[1]), float(curr_data[2]), float(curr_data[3])])
                curr_prim['axis'] = np.array([float(curr_data[4]), float(curr_data[5]), float(curr_data[6])])
                curr_prim['angle'] = float(curr_data[7])
            prim_data.append(curr_prim)
    return prim_data

def get_test_case_data(bagsfit_files_path, sequence, test_case):
    full_path = bagsfit_files_path + f"{sequence:03d}" + "/" +f"{test_case:05d}" +".npz"
    geom_data={}
    with np.load(full_path) as data:
        geom_data['data'] = data['data'].reshape(3,-1).T
        geom_data['ins'] = np.array(data['ins']).reshape(-1,1).flatten()
        geom_data['cls']= np.array(data['cls']).reshape(-1,1).flatten()
        geom_data['scan'] = data['scan'].reshape(3,-1).T

    geom_data['prim'] = get_prim_data(bagsfit_files_path, sequence, test_case)
    return geom_data

import os

def get_all_test_cases(test_cases_path):
    test_cases = []

    for seq_index in range(len(os.listdir(test_cases_path))):
        cases_list = os.listdir(test_cases_path+"/"+ f"{seq_index:03d}")
        subtest_count = sum('.npz' in s for s in cases_list)
        for i in range(subtest_count):
            try:
                #test_cases.append(get_test_case_data(test_cases_path, seq_index, i))
                test_cases.append([test_cases_path, seq_index, i])
            except:
                break
    return test_cases

####################################


import torch
import torch.utils.data as data

class PointCloudDataset(data.Dataset):
    def __init__(self, datapath):

        self.datapath = datapath
        self.test_data_path = get_all_test_cases(datapath)

        #self.points = points
        #self.labels = labels

    def __len__(self):
        return len(self.test_data_path)

    def __getitem__(self, idx):
        #print(self.test_data_path[idx])
        test_data = get_test_case_data(self.test_data_path[idx][0], self.test_data_path[idx][1], self.test_data_path[idx][2])
        #self.test_data[idx]['data'] = torch.from_numpy(self.test_data[idx]['data']).float()
        return torch.from_numpy(test_data['data']).float(), torch.from_numpy(test_data['cls']).float()




# chunk open3d pointcloud to uniform size spatial chunks base on the bounding box
def chunk_pointcloud(pointcloud, chunk_size):
    # get bounding box
    min_bound = np.min(pointcloud, axis=0)
    max_bound = np.max(pointcloud, axis=0)
    # get chunk size
    chunk_size = np.array(chunk_size)
    # get chunk count
    chunk_count = np.ceil((max_bound - min_bound) / chunk_size).astype(np.int)
    # get chunk centers
    chunk_centers = np.zeros((np.prod(chunk_count), 3))
    for i in range(chunk_count[0]):
        for j in range(chunk_count[1]):
            for k in range(chunk_count[2]):
                chunk_centers[i * chunk_count[1] * chunk_count[2] + j * chunk_count[2] + k] = min_bound + chunk_size / 2 + np.array([i, j, k]) * chunk_size
    # get chunk indices
    chunk_indices = np.zeros((np.prod(chunk_count), 2), dtype=np.int)
    for i in range(chunk_count[0]):
        for j in range(chunk_count[1]):
            for k in range(chunk_count[2]):
                chunk_indices[i * chunk_count[1] * chunk_count[2] + j * chunk_count[2] + k] = np.array([i, j, k])
    # get chunk pointclouds
    chunk_pointclouds = []
    for i in range(np.prod(chunk_count)):
        chunk_pointclouds.append(pointcloud[np.all(pointcloud >= chunk_centers[i] - chunk_size / 2, axis=1) & np.all(pointcloud < chunk_centers[i] + chunk_size / 2, axis=1)])
    return chunk_pointclouds, chunk_centers, chunk_indices



# extract sub pointclouds from pointcloud
def extract_sub_pointclouds(pointcloud, sub_pointcloud_size, sub_pointcloud_count):
    # get bounding box
    min_bound = np.min(pointcloud, axis=0)
    max_bound = np.max(pointcloud, axis=0)
    # get sub pointcloud size
    sub_pointcloud_size = np.array(sub_pointcloud_size)
    # get sub pointcloud centers
    sub_pointcloud_centers = np.zeros((sub_pointcloud_count, 3))
    for i in range(sub_pointcloud_count):
        sub_pointcloud_centers[i] = min_bound + sub_pointcloud_size / 2 + np.random.rand(3) * (max_bound - min_bound - sub_pointcloud_size)
    # get sub pointclouds
    sub_pointclouds = []
    for i in range(sub_pointcloud_count):
        sub_pointclouds.append(pointcloud[np.all(pointcloud >= sub_pointcloud_centers[i] - sub_pointcloud_size / 2, axis=1) & np.all(pointcloud < sub_pointcloud_centers[i] + sub_pointcloud_size / 2, axis=1)])
    return sub_pointclouds, sub_pointcloud_centers

# 