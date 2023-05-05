import pickle, time, warnings
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from utils.tools import Config as cfg
from utils.tools import DataProcessing as DP

import os

class PointCloudsDataset(Dataset):
    def __init__(self, dir, labels_available=True):
        self.paths = list(dir.glob(f'*.npy'))
        self.labels_available = labels_available

    def __getitem__(self, idx):
        path = self.paths[idx]

        points, labels = self.load_npy(path)

        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()

        return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)

    def load_npy(self, path):
        r"""
            load the point cloud and labels of the npy file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_npy = np.load(path, mmap_mode='r').T
        points = cloud_npy[:,:-1] if self.labels_available else points

        if self.labels_available:
            labels = cloud_npy[:,-1]

            # balance training set
            points_list, labels_list = [], []
            for i in range(len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 8000)
                    points_list.append(points[labels==i][idx])
                    labels_list.append(labels[labels==i][idx])
                except ValueError:
                    continue
            if points_list:
                points = np.stack(points_list)
                labels = np.stack(labels_list)
                labeled = labels>0
                points = points[labeled]
                labels = labels[labeled]

        return points, labels




#from torch_geometric.data import Dataset, Data

map_G_channel_to_shape_index = {
        0.0:0,
        0.1:1,
        0.2:2,
        0.3:3,
        0.4:4,
        0.5:5,
     #   0.9:6,
    }


class GeomSegDataset(Dataset):
    def __init__(self, dir, points_count=110000, labels_available=True):
        self.paths = list(dir.glob(f'*.pt'))
        self.labels_available = labels_available

        self.points_count = points_count

    def __getitem__(self, idx):
        path = self.paths[idx]

        data = torch.load(path)

        points_tensor = data.pos#.float()
        labels_tensor = data.y#.long()
        #print("points_tensor: ", points_tensor.shape)
        #print("labels_tensor: ", labels_tensor.shape)
        ## Check if it works [Pranay]
        point_indices = np.random.choice(len(points_tensor),self.points_count)

        #print(points_tensor.shape, labels_tensor.shape, point_indices)
        points_tensor = points_tensor[point_indices, :]
        labels_tensor = labels_tensor[point_indices]

        #print(points_tensor.shape, labels_tensor.shape)


        return points_tensor, labels_tensor

        # points, labels = self.read_raw(path)

        # if points is None:
        #     points, labels = np.array([[0,0,0]]), np.array([0])
        # points_tensor = torch.from_numpy(points).float()
        # labels_tensor = torch.from_numpy(labels).long()

        # return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)
    
    def read_raw(self, scan_file):
        import open3d
        full_path = scan_file
        
        #geom_data={}
        print("Reading scan: ", str(full_path.stem))
        pcd = open3d.io.read_point_cloud(str(full_path.stem))
        
        scan = np.asarray(pcd.points, dtype=np.float32)

        colors =  np.asarray(pcd.colors, dtype=np.float32)        
        y = np.array(colors[:, 1]).round(1)*10
        y = np.array([map_G_channel_to_shape_index[c/10] for c in y])
        ins_label = np.array(colors[:, 2])

        ins_label = np.unique(ins_label, return_inverse = True)[1]  
        
        #print("labels: ", np.unique(ins_label), np.min(ins_label), np.max(ins_label))
        print(len(ins_label))
        if len(ins_label)<5:
            print("Not enough objects found in the scan: ", full_path)
            print("Discarding....")
            return None, None

        # ins_label = ins_label+1  # since instances should start with 1
        # data = Data(pos=torch.tensor(scan[:,:]), x=torch.ones((len(scan), 1)),)
        # data.y = torch.tensor(y[:]).long()
        # data.instance_labels = torch.tensor(ins_label[:]).long()
        # data.x = None

        #TBD: fill primitive data as well
        #geom_data['prim'] = get_prim_data(bagsfit_files_path, sequence, test_case)
        return scan, y






map_G_channel_to_shape_index_4cls = {
        0.0:0,
        0.1:1,
        0.2:2,
        0.3:3,
        0.4:4,
        0.5:0,
     #   0.9:6,
    }


class GeomSegDataset_4cls(Dataset):
    def __init__(self, dir, points_count=110000, labels_available=True):
        self.paths = list(dir.glob(f'*.pcd'))
        self.labels_available = labels_available

        self.points_count = points_count

    def __getitem__(self, idx):
        path = self.paths[idx]

        # data = torch.load(path)

        # points_tensor = data.pos#.float()
        # labels_tensor = data.y#.long()
        # #print("points_tensor: ", points_tensor.shape)
        # #print("labels_tensor: ", labels_tensor.shape)
        # ## Check if it works [Pranay]
        # point_indices = np.random.choice(len(points_tensor),self.points_count)

        # #print(points_tensor.shape, labels_tensor.shape, point_indices)
        # points_tensor = points_tensor[point_indices, :]
        # labels_tensor = labels_tensor[point_indices]

        # #print(points_tensor.shape, labels_tensor.shape)


        # return points_tensor, labels_tensor
        points, labels = self.read_raw(path)

        if points is None:
            points, labels = np.array([[0,0,0]]), np.array([0])
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()


        point_indices = np.random.choice(len(points_tensor),self.points_count)

        #print(points_tensor.shape, labels_tensor.shape, point_indices)
        points_tensor = points_tensor[point_indices, :]
        labels_tensor = labels_tensor[point_indices]

        return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)
    
    def read_raw(self, scan_file):
        import open3d

        full_path = scan_file
        
        #geom_data={}
        print("Reading scan: ", str(full_path.stem))
        pcd = open3d.io.read_point_cloud(str(full_path.stem))
        
        scan = np.asarray(pcd.points, dtype=np.float32)

        colors =  np.asarray(pcd.colors, dtype=np.float32)        
        y = np.array(colors[:, 1]).round(1)*10
        y = np.array([map_G_channel_to_shape_index_4cls[c/10] for c in y])
        ins_label = np.array(colors[:, 2])

        ins_label = np.unique(ins_label, return_inverse = True)[1]  
        
        #print("labels: ", np.unique(ins_label), np.min(ins_label), np.max(ins_label))
        print(len(ins_label))
        if len(ins_label)<5:
            print("Not enough objects found in the scan: ", full_path)
            print("Discarding....")
            return None, None

        # ins_label = ins_label+1  # since instances should start with 1
        # data = Data(pos=torch.tensor(scan[:,:]), x=torch.ones((len(scan), 1)),)
        # data.y = torch.tensor(y[:]).long()
        # data.instance_labels = torch.tensor(ins_label[:]).long()
        # data.x = None

        #TBD: fill primitive data as well
        #geom_data['prim'] = get_prim_data(bagsfit_files_path, sequence, test_case)
        return scan, y



class CloudsDataset(Dataset):
    def __init__(self, dir, data_type='npy'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        self.val_split = '1_'

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
            sub_npy_file = self.path / '{:s}.npy'.format(cloud_name)

            data = np.load(sub_npy_file, mmap_mode='r').T

            sub_colors = data[:,3:6]
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                print('Loading {:s}'.format(kd_tree_file.name), os.path.exists(kd_tree_file))
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size





class CloudsDataset_Pos(Dataset):
    def __init__(self, dir, data_type='npy'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        #self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        #self.val_split = '1_'
        self.val_split = 'val'

        self.load_data()
        print('Size of training : ', len(self.input_labels['training']))
        print('Size of validation : ', len(self.input_labels['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            par_name = str(file_path.parent)
            #print("\n\n"+cloud_name, file_path, par_name)
            if self.val_split in par_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
            sub_npy_file = self.path / '{:s}.npy'.format(cloud_name)

            data = np.load(sub_npy_file, mmap_mode='r').T

            #sub_colors = data[:,3:6]
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                print('Loading {:s}'.format(kd_tree_file.name), os.path.exists(kd_tree_file))
                #try:
                search_tree = pickle.load(f)
                #except:
                #    print('Error loading {:s}'.format(kd_tree_file.name))
                #    continue

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            #self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)

            size = sub_labels.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size
    
    # def mergeDatasets(self, other):
    #     self.input_trees['training'] += other.input_trees['training']
    #     self.input_trees['validation'] += other.input_trees['validation']
    #     self.input_labels['training'] += other.input_labels['training']
    #     self.input_labels['validation'] += other.input_labels['validation']
    #     self.input_names['training'] += other.input_names['training']
    #     self.input_names['validation'] += other.input_names['validation']
    #     self.size = len(self.input_trees['training']) + len(self.input_trees['validation'])


def mergeCloudsDataset(cloudsDataset1, cloudsDataset2):
    cloudsDataset1.input_trees['training'] += cloudsDataset2.input_trees['training']
    cloudsDataset1.input_trees['validation'] += cloudsDataset2.input_trees['validation']
    cloudsDataset1.input_labels['training'] += cloudsDataset2.input_labels['training']
    cloudsDataset1.input_labels['validation'] += cloudsDataset2.input_labels['validation']
    cloudsDataset1.input_names['training'] += cloudsDataset2.input_names['training']
    cloudsDataset1.input_names['validation'] += cloudsDataset2.input_names['validation']
    cloudsDataset1.size = len(cloudsDataset1.input_trees['training']) + len(cloudsDataset1.input_trees['validation'])
    return cloudsDataset1



class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training'):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            if cfg.sampling_type=='active_learning':
                # Generator loop

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[self.split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[self.split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[self.split][cloud_idx][queried_idx] += delta
                self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            # Simple random choice of cloud and points in it
            elif cfg.sampling_type=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), cfg.num_points)
                queried_pc_xyz = points[queried_idx]
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

            points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

            yield points, queried_pc_labels





class ActiveLearningSampler_Pos(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training'):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_labels[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            if cfg.sampling_type=='active_learning':
                # Generator loop

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[self.split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[self.split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = None #queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[self.split][cloud_idx][queried_idx] += delta
                self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            # Simple random choice of cloud and points in it
            elif cfg.sampling_type=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), cfg.num_points)
                queried_pc_xyz = points[queried_idx]
                #queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            #queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

            points = queried_pc_xyz # torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

            yield points, queried_pc_labels


# class FixedPointSampler(IterableDataset):

#     def __init__(self, dataset, points_count=110000):
#         self.dataset = dataset
#         self.points_count = points_count

#     def __iter__(self):
#         return self.random_gen()

#     def __len__(self):
#         return self.points_count # not equal to the actual size of the dataset, but enable nice progress bars

#     def random_gen(self):
#         # Choosing the least known point as center of a new cloud each time.

#         point_indices = np.random.choice(self.points_count,1)
#         self.dataset

#         points = None
#         labels =None
#         yield points, labels




def data_loaders(dataset1,dataset2, dataset3, sampling_method='active_learning', **kwargs):
    if sampling_method == 'active_learning':
        # dataset_val = CloudsDataset_Pos(dataset1 / 'val')
        path_val_1 = dataset1 / 'val'
        path_val_2 = dataset2 / 'val'
        path_val_3 = dataset3 / 'val'
        print ("Printing path_val_1")
        print (path_val_1)
        print (path_val_2)
        print (path_val_3)
        dataset_val1 = CloudsDataset_Pos(path_val_1)
        dataset_val2 = CloudsDataset_Pos(path_val_2)
        # dataset_val3 = CloudsDataset_Pos(path_val_3) 

        # Print sample from dataset 1
        print(len(dataset_val1))

        dataset12 = mergeCloudsDataset(dataset_val1, dataset_val2)
        # dataset_val = mergeCloudsDataset(dataset12, dataset_val3)
        dataset_val = dataset12

        # Print sample from merged dataset
        print(len(dataset_val))

        batch_size = kwargs.get('batch_size', 6)
        val_sampler = ActiveLearningSampler_Pos(
            dataset_val,
            batch_size=batch_size,
            split='validation'
        )
        # dataset = CloudsDataset_Pos(dir / 'train')
        path1 = dataset1 / 'train'
        path2 = dataset2 / 'train'
        path3 = dataset3 / 'train'
        dataset1 = CloudsDataset_Pos(path1)
        dataset2 = CloudsDataset_Pos(path2)
        dataset3 = CloudsDataset_Pos(path3)
        dataset12 = mergeCloudsDataset(dataset1, dataset2)
        # dataset = mergeCloudsDataset(dataset12, dataset3)

        dataset = dataset12

        train_sampler = ActiveLearningSampler_Pos(
            dataset,
            batch_size=batch_size,
            split='training'
        )

        print('Size of training : ', len(dataset.input_labels['training']))
        print('Size of validation : ', len(dataset_val.input_labels['validation']))
        return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    if sampling_method == 'naive':
        train_dataset = PointCloudsDataset(dir / 'train')
        val_dataset = PointCloudsDataset(dir / 'val')
        return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)
    if sampling_method == 'geomseg':
        train_dataset = GeomSegDataset(dir / 'train')
        val_dataset = GeomSegDataset(dir / 'val')
        return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)

    raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")

if __name__ == '__main__':
    dataset = CloudsDataset('datasets/s3dis/subsampled/train')
    batch_sampler = ActiveLearningSampler(dataset)
    for data in batch_sampler:
        xyz, colors, labels, idx, cloud_idx = data
        print('Number of points:', len(xyz))
        print('Point position:', xyz[1])
        print('Color:', colors[1])
        print('Label:', labels[1])
        print('Index of point:', idx[1])
        print('Cloud index:', cloud_idx)
        break
