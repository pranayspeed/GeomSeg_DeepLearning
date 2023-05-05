
import utils.UtilsPointcloud as Ptutils

from abc import ABC, abstractmethod

import utils.ICP as ICP

import open3d as o3d
import numpy as np

class KeyPointsExtractorBase(ABC):
    
    @abstractmethod
    def extract(source_pts):
        pass
    
    @abstractmethod
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        pass        

class NoSampleExtractor(KeyPointsExtractorBase):

    def extract(self, source_pts):
        #Just return the same input points
        return source_pts   
              
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        return odom_transform
    
class RandomSampleExtractor(KeyPointsExtractorBase):
    def __init__(self, num_icp_points):
        self.num_icp_points = num_icp_points


    def extract(self, source_pts):

        #Just get num_icp_points random samples
        return Ptutils.random_sampling(source_pts, num_points=self.num_icp_points)    

        
      
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        odom_transform, _, _ = ICP.icp(traget_kps, source_kps, init_pose=initial_transf, max_iterations=20)
        return odom_transform
    

import math   
    
class UniformSampleExtractor(KeyPointsExtractorBase):
    def __init__(self, num_icp_points):
        self.num_icp_points = num_icp_points

    def extract(self, source_pts):

        #Just get num_icp_points Uniform samples
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(source_pts) 
        total_len = len(source_pts)

        req_downsample_k = math.ceil(total_len/self.num_icp_points)
        pcd = pcd.uniform_down_sample(req_downsample_k)
        
        #print(len(pcd.points), req_downsample_k, end='\r')
        return np.asarray(pcd.points) 
          
      
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        return odom_transform
    
# for sift and harris 3d 
#import PCLKeypoint
    
#import pcl

class Harris3DExtractor(KeyPointsExtractorBase):

    def extract(self, source_pts):
        #pcl.
        return  PCLKeypoint.keypointHarris3D(source_pts)
          
      
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        return odom_transform
    
    
class SIFT3DExtractor(KeyPointsExtractorBase):

    def extract(self, source_pts):

        return  PCLKeypoint.keypointSift(source_pts)
          
      
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        return odom_transform
    
    
class ISSExtractor(KeyPointsExtractorBase):

    def extract(self, source_pts):
        return  PCLKeypoint.keypointIss(source_pts)
          
      
    def compute_transform(self, source_kps, traget_kps, initial_transf):
        min_len = min(len(traget_kps), len(source_kps))
        odom_transform, _, _ = ICP.icp(traget_kps[:min_len], source_kps[:min_len], init_pose=initial_transf, max_iterations=20)
        return odom_transform