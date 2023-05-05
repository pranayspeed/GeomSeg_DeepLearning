import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler







def visualize_clusters(cluster_model, points):
    # Get labels:
    labels = cluster_model.labels_

    print(np.unique(labels, return_counts=True))
    # Get the number of colors:
    n_clusters = len(set(labels))
    print(n_clusters)
    # Mapping the labels classes to a color map:
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # Attribute to noise the black color:
    colors[labels < 0] = 0
    # Update points colors:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Display:
    o3d.visualization.draw_geometries([pcd])





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

def simplePCA(arr):
    '''
    :param arr: input array of shape shape[N,M]
    :return:
        mean - center of the multidimensional data,
        eigenvalues - scale,
        eigenvectors - direction
    '''

    # calculate mean
    m = np.mean(arr, axis=0)

    # center data
    arrm = arr-m

    # calculate the covariance, decompose eigenvectors and eigenvalues
    # M * vect = eigenval * vect
    # cov = M*M.T
    Cov = np.cov(arrm.T)
    eigval, eigvect = np.linalg.eig(Cov.T)

    # return mean, eigenvalues, eigenvectors
    return m, eigval, eigvect

def compute_centroid(points):
    return np.mean(points, axis=0)


def region_grow_DBSCAN(points, eps=0.15, min_samples=50):

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points)
    # Clustering:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_points)
    return model



def visualize_project_of_eigen_vector(points_old, index):

    model = region_grow_DBSCAN(points_old, eps=0.45, min_samples=20)
    clusters, labels_new = split_points_based_on_labels(points_old, model)
    if len(clusters) == 0:
        return
    points = clusters[0]

    np.savetxt('geometry_fitting/cone_points_region_grow_'+str(index)+'.txt', points)
    m, eigval, eigvect = simplePCA(points)

    center = compute_centroid(points)
    projected_points = points - center

    #Project on secondary axis:
    axis_1 = eigvect[1,:]
    for i in range(len(points)):
        v = center - points[i]# - center
        d = np.dot(v, axis_1)
        projected_points[i] = points[i] + d*axis_1

    #Project on tertiary axis:
    axis_1 = eigvect[2,:]
    for i in range(len(projected_points)):
        v = center - projected_points[i]# - center
        d = np.dot(v, axis_1)
        projected_points[i] = projected_points[i] + d*axis_1

    
    pcd_projected = o3d.geometry.PointCloud()  # create point cloud object\
    pcd_projected.points = o3d.utility.Vector3dVector(projected_points)  # set pcd_np as the point cloud points
    pcd_projected.paint_uniform_color([0.5, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0.706, 0])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([center, center + 5*eigval[0]*eigvect[0,:], center +  5*eigval[1]*eigvect[1,:], center +  5*eigval[2]*eigvect[2,:]])
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [0,2], [0,3]])
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(eigval)
    print(eigval[0]/eigval[1], eigval[1]/eigval[2], eigval[0]/eigval[2])

    o3d.visualization.draw_geometries([pcd_projected, pcd, ls])



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



if __name__ == '__main__':
    datafile = 'cone_points.txt'

    points = np.loadtxt(datafile)


    model = region_grow_with_dbscan(points, eps=0.018, min_samples=100)
    clusters, labels_new = split_points_based_on_labels(points, model)
    for i in range(len(clusters)):
        visualize_project_of_eigen_vector(clusters[i], i)


    # visualize_clusters(model, points)

    # model = region_grow_with_dbscan(points)
    # # # Normalisation:
    # # scaled_points = StandardScaler().fit_transform(points)
    # # # Clustering:
    # # model = DBSCAN(eps=0.15, min_samples=50)
    # # model.fit(scaled_points)

    # visualize_clusters(model, points)

