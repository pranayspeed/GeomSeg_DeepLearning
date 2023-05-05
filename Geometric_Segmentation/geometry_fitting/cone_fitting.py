
import numpy as np
import open3d as o3d

def compute_cone_axis_from_points(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    A = np.vstack([x, y, np.ones(x.shape[0])]).T
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
    print(np.linalg.norm([a, b, -1]))
    #return np.array([a, b, -1]) / np.linalg.norm([a, b, -1])

    return np.array([b,c, -1]) / np.linalg.norm([b,c, -1])

def compute_centroid(points):
    return np.mean(points, axis=0)

def closest_point_on_line(points,lVersor,lPoint):
    """From a list of points in 3D space as Nx3 array, returns a Nx3 array with the corresponding closest points on the line."""
    #vd=lVersor/((np.array(lVersor)**2).sum())  #normalize vector, not sure it is necessary.
    vd=lVersor/np.sqrt(np.sum(np.array(lVersor)**2))  #normalize vector, not sure it is necessary.
    return lPoint+(points-lPoint).dot(vd)[:,np.newaxis]*(vd)

#Reference : https://github.com/webclinic017/pyXSurf/blob/c3cbf6ac5f42a8a769bce1e38ef63075a68478f5/pyxsurf/pySurf/fit_cylinder.py

def cone_error(odr=(0,0,0,0,0,0),points=None,retall=False,extra=False):  #origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit
    """Given a set of N points in format Nx3, returns the rms surface error on the cone defined by origin (intercept of the axis with x=0) and direction, 
    passed as 4-vector odr(origin_y,origin_z,direction_x,direction_z). 
    Best fit cone for odr is calculated from linear fit of data. 
    If retall is set, additional values are returned :
    coeff: best fit radius for the cone as [m,q] for x' distance from x=0 plan on cone axis R(x')=m x' + q. Half cone angle is atan(m). 
    deltaR[N,3]: deviation from radius for each point.
    extra is equivalent to retall, renamed 2018/02/15 kept for backward compatibility
    """
    #ca 400 ms per loop
    origin=odr[0:3]
    direction=odr[3:]
    vd=direction/np.sqrt((1+np.array(direction)**2).sum())  #normalize vector, not sure it is necessary.
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,vd,origin)
    Paxdist=np.sqrt(((Paxis-origin)**2).sum(axis=1)) #distance of each point from
    R=np.sqrt(((points-Paxis)**2).sum(axis=1)) #distance of each point from axis
    coeff=np.polyfit(Paxdist,R,1) #best fit cone
    deltaR=R-coeff[0]*Paxdist-coeff[1] #residuals
    fom=np.sqrt(((deltaR)**2).sum()/len(deltaR))
    residuals=np.hstack([x,y,deltaR[:,None]])
    retall=retall | extra
    if retall: return fom,residuals,coeff
    else: return fom  #,deltaR,radius



from scipy.optimize import minimize, least_squares


import time


import math

def get_fitted_cone(points, threshold=0.0009):

    centeroid = compute_centroid(points)
    cone_axis = compute_cone_axis_from_points(points)
    #uu, dd, vv = np.linalg.svd(points - centeroid)
    #print(vv[0])
    #cone_axis_new = vv[0]

    fom_func = cone_error
    odr2 = np.array([centeroid[0], centeroid[1], centeroid[2], cone_axis[0], cone_axis[1], cone_axis[2]])

    t0 = time.time()
    result=minimize(fom_func,x0=(odr2,),args=(points,),options={'maxiter':1000})

    #result = least_squares(fom_func, x0=odr2, args=(points,))#, method='lm')
    print ('-----------------------------------')
    print ('Results of fit on subset of points:', time.time() - t0, 'seconds')
    #print (result)

    fom,deltaR,coeff=fom_func(odr2,points,retall=True)
    m,q = coeff

    #print ('fom,deltaR,coeff',fom,deltaR,coeff)

    half_angle = math.atan(m)

    #result.x[3:] = result.x[3:] / np.linalg.norm(result.x[3:])
    cone_axis_new = result.x[3:]
    cone_center_new = result.x[:3]    
    error = result.fun
    apex_distance = np.linalg.norm(cone_center_new - centeroid)
    max_spread = np.max(np.linalg.norm(points - centeroid, axis=1))
    print("apex_distance: ", apex_distance, "half_angle: ", half_angle, "error: ", error)
    if error < threshold and half_angle >0 and half_angle < np.pi/4 and apex_distance < 3*max_spread:
        return [cone_axis_new, cone_center_new, half_angle, error]
    return None




def vec2vec_rotation(unit_vec_1, unit_vec_2):
    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))
    if angle < 1e-8:
        return np.identity(3, dtype=np.float64)

    if angle > (np.pi - 1e-8):
        # WARNING this only works because all geometries are rotationaly invariant
        # minus identity is not a proper rotation matrix
        return -np.identity(3, dtype=np.float64)

    rot_vec = np.cross(unit_vec_1, unit_vec_2)
    rot_vec /= np.linalg.norm(rot_vec)

    return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)


def get_cone_mesh(half_angle, cone_axis, cone_center):
    cone_mesh = o3d.geometry.TriangleMesh.create_cone(radius=math.tan(half_angle), height=1.0)
    
    cone_mesh.compute_vertex_normals()
    cone_mesh.remove_vertices_by_index([0])
    cone_mesh.translate(cone_center)
    cone_mesh.translate(-cone_axis)
    rotation = vec2vec_rotation( [0, 0, 1], cone_axis)
    cone_mesh.rotate(rotation)

    return cone_mesh

def get_sphere_mesh(radius, center):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.compute_vertex_normals()
    sphere_mesh.translate(center)
    return sphere_mesh

def main(points):
    # Load saved point cloud

    #points = np.loadtxt(points_file)

    resulted_fit = get_fitted_cone(points)
    if resulted_fit is None:
        return
    cone_axis_new, cone_center_new, half_angle, error = resulted_fit


    print("Error: ", error, "\nHalf angle: ", math.degrees(half_angle), "\nCone axis: ", cone_axis_new, "\nCone center: ", cone_center_new)
    ls1 = o3d.geometry.LineSet()
    ls1.points = o3d.utility.Vector3dVector([cone_center_new, cone_center_new-cone_axis_new])
    ls1.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0.706, 0])


    apex_sphere = get_sphere_mesh(0.1, cone_center_new)
    cone_mesh = get_cone_mesh(half_angle, cone_axis_new, cone_center_new)

    o3d.visualization.draw_geometries([ls1, pcd, cone_mesh, apex_sphere])



def old_main(points_file='cone_points_region_grow_0.txt'):
    # Load saved point cloud

    points = np.loadtxt(points_file)

    cone_axis = compute_cone_axis_from_points(points)
    print(cone_axis)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0.706, 0])
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))

    centeroid = compute_centroid(points)

    uu, dd, vv = np.linalg.svd(points - centeroid)

    print(vv[0])

    cone_axis_new = vv[0]
    

    fom_func = cone_error
    odr2 = np.array([centeroid[0], centeroid[1], centeroid[2], cone_axis[0], cone_axis[1], cone_axis[2]])
    result=minimize(fom_func,x0=(odr2,),args=(points,),options={'maxiter':1000})#,method='Nelder-Mead')
    print ('-----------------------------------')
    print ('Results of fit on subset of points:')
    print (result)

    #result.x[3:] = result.x[3:] / np.linalg.norm(result.x[3:])
    cone_axis_new = result.x[3:]
    cone_center_new = result.x[:3]

    ls1 = o3d.geometry.LineSet()
    ls1.points = o3d.utility.Vector3dVector([cone_center_new, cone_center_new-cone_axis_new])
    ls1.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])



    # normals = np.asarray(pcd.normals)
    # for i in range(len(normals)):
    #     if np.dot(normals[i], cone_axis) < 0:
    #         normals[i] = -normals[i]
    # pcd.normals = o3d.utility.Vector3dVector(normals)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([centeroid, centeroid+cone_axis])
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]])


    center = compute_centroid(points)
    projected_points = points# - center
    #Project on tertiary axis:
    axis_1 = cone_axis
    for i in range(len(projected_points)):
        v = center - projected_points[i]# - center
        d = np.dot(v, axis_1)
        projected_points[i] = projected_points[i] + d*axis_1

    pcd_projected = o3d.geometry.PointCloud()  # create point cloud object\
    pcd_projected.points = o3d.utility.Vector3dVector(projected_points)  # set pcd_np as the point cloud points
    pcd_projected.paint_uniform_color([0.5, 0, 0])


    o3d.visualization.draw_geometries([pcd, ls1])



from sklearn.cluster import  DBSCAN
from sklearn.preprocessing import StandardScaler


def region_grow_with_dbscan(points, eps=0.08, min_samples=50):

    #TO Visualize the projection:
    # points[:, 1] = 0
    # pcd_projected = o3d.geometry.PointCloud()  # create point cloud object
    # pcd_projected.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points
    # o3d.visualization.draw_geometries([pcd_projected])

    # projection: caonsider the x and z coordinates (y=0)
    points_xz = points[:, [0, 2]]

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points_xz)

    # Clustering:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_points)

    return model


def split_points_based_on_labels(points, model):
    labels = model.labels_
    unique_labels = np.unique(labels)
    clusters = []
    labels_new = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = points[labels == label]
        clusters.append(cluster)
        labels_new.append(label)
    return clusters, labels_new



def region_grow_DBSCAN(points, eps=0.15, min_samples=50):

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points)
    # Clustering:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_points)
    return model


def detect_cones(points, threshold=0.0009):

    model = region_grow_with_dbscan(points, eps=0.018, min_samples=100)
    clusters, labels_new = split_points_based_on_labels(points, model)
    cones = []
    cluster_points = []
    for i in range(len(clusters)):
        #visualize_project_of_eigen_vector(clusters[i], i)
        points_old = clusters[i]
        model_new = region_grow_DBSCAN(points_old, eps=0.45, min_samples=20)
        clusters_new, labels_new_2 = split_points_based_on_labels(points_old, model_new)
        if len(clusters_new) == 0:
            continue
        points = clusters_new[0]
        resulted_fit = get_fitted_cone(points, threshold)
        if resulted_fit is not None:
            cones.append(resulted_fit)
            cluster_points.append(points)
    return cones, cluster_points

import os

if __name__ == '__main__':

    datafile = '../cone_points.txt'

    points = np.loadtxt(datafile)

    cones, cluster_points = detect_cones(points, threshold=0.009)
    all_geom = []
    for i in range(len(cones)):
        cone_axis_new, cone_center_new, half_angle, error = cones[i]
        cluster_points_curr = cluster_points[i]


        print("Error: ", error, "\nHalf angle: ", math.degrees(half_angle), "\nCone axis: ", cone_axis_new, "\nCone center: ", cone_center_new)
        ls1 = o3d.geometry.LineSet()
        ls1.points = o3d.utility.Vector3dVector([cone_center_new, cone_center_new-cone_axis_new])
        ls1.lines = o3d.utility.Vector2iVector([[0, 1]])
        ls1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_points_curr)
        pcd.paint_uniform_color([0, 0.706, 0])


        apex_sphere = get_sphere_mesh(0.1, cone_center_new)
        apex_sphere.paint_uniform_color([0, 0, 1])
        cone_mesh = get_cone_mesh(half_angle, cone_axis_new, cone_center_new)

        all_geom.append(ls1)
        #all_geom.append(pcd)
        all_geom.append(cone_mesh)
        all_geom.append(apex_sphere)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0.706, 0])
    all_geom.append(pcd)
    o3d.visualization.draw_geometries(all_geom) #[ls1, pcd, cone_mesh, apex_sphere])


    # model = region_grow_with_dbscan(points, eps=0.018, min_samples=100)
    # clusters, labels_new = split_points_based_on_labels(points, model)
    # for i in range(len(clusters)):
    #     #visualize_project_of_eigen_vector(clusters[i], i)
    #     points_old = clusters[i]
    #     model_new = region_grow_DBSCAN(points_old, eps=0.45, min_samples=20)
    #     clusters_new, labels_new = split_points_based_on_labels(points_old, model_new)
    #     if len(clusters_new) == 0:
    #         continue
    #     points = clusters_new[0]

    #     main(points)


    # for i in range(7):
    #     if os.path.exists('cone_points_region_grow_'+str(i)+'.txt'):
    #         points = np.loadtxt('cone_points_region_grow_'+str(i)+'.txt')
    #         main(points)
