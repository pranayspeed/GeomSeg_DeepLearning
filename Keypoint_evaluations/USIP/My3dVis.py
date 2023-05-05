import open3d as o3d
import numpy as np


import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4
def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.
    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix
    Raises:
        ValueError: if `len(rpy) != 3`.
    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx

def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.
    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix
    Raises:
        ValueError: if `len(xyzrpy) != 6`
    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 =  matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

class My3DVis:
    def __init__(self, generate_mesh=False):
        self.pcd = o3d.geometry.PointCloud()
        self.traj = None
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        self.mesh= None
        self.frames=0
        self.frame_size=1
        self.mesh_points=o3d.geometry.PointCloud()
        
        self.generate_mesh= generate_mesh
        
        self.pcd_all = [self.pcd]
        
        self.view_control = self.vis.get_view_control()
        params = self.view_control.convert_to_pinhole_camera_parameters()
        params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
        self.view_control.convert_from_pinhole_camera_parameters(params)
                
    def set_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)        
        self.pcd_all.append(pcd)
        
        self.pcd = pcd
        # original_cnt = len(self.pcd.points)
        # #cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20,
        # #                                            std_ratio=2.0)
        # # cl, ind = self.pcd.remove_radius_outlier(nb_points=16, radius=0.5)
        # # print("outliers :", original_cnt, "->",len(ind), end='\r')

        # # self.pcd.points = o3d.utility.Vector3dVector(points)
        # # self.vis.add_geometry(self.pcd)     
    def filter_mesh(self, curr_mesh , triangle_cnts=300):
        #print("Cluster connected triangles")

        curr_mesh = curr_mesh.filter_smooth_simple(number_of_iterations=500)
        curr_mesh = curr_mesh.simplify_quadric_decimation(target_number_of_triangles=200)     
        #with o3d.utility.VerbosityContextManager(
        #        o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
                curr_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        #mesh_0 = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < triangle_cnts
        curr_mesh.remove_triangles_by_mask(triangles_to_remove)


        # curr_mesh = curr_mesh.filter_smooth_taubin(number_of_iterations=100)
        # curr_mesh.compute_vertex_normals()

    def set_transform(self, transf):        
        self.pcd.transform(transf)
        
        self.mesh_points += self.pcd
        self.frames = (self.frames+ 1)% self.frame_size
        
        
        if self.frames ==0:

            if self.generate_mesh:
                old_mesh = self.mesh                

    #             self.mesh_points.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #             radii = [0.005, 0.01, 0.02, 0.04]
    #             self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #                 self.mesh_points, o3d.utility.DoubleVector(radii))

                alpha = 0.310
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.mesh_points, alpha)
                self.mesh.compute_vertex_normals()  
                
                # 
                
                #Pranay : following simplify_vertex_clustering is actually minimizing number of triangles significantly
                voxel_size = max(self.mesh.get_max_bound() - self.mesh.get_min_bound()) / 16
                print(f'voxel_size = {voxel_size:e}', end='\r')
                self.mesh = self.mesh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average)                
                self.mesh.orient_triangles()
                #self.mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
                if old_mesh is not None:                      
                    #self.filter_mesh(self.mesh)
                    self.mesh = old_mesh+ self.mesh
                self.vis.clear_geometries()                        
                self.vis.add_geometry(self.mesh)
            else:
                
                self.vis.add_geometry(self.mesh_points)
                    
            
            self.mesh_points =  o3d.geometry.PointCloud() # reset mesh points
            
                 
    def refresh(self):
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def refresh_transform(self, indx, transf):
        self.pcd_all[indx].transform(transf)
        
        self.vis.update_geometry(self.pcd_all[indx])
        self.vis.poll_events()
        self.vis.update_renderer()      
        
    def destroy(self):
        self.vis.destroy_window()    