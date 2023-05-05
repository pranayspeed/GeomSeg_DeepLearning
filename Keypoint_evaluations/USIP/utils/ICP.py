"""
ref: https://github.com/ClayFlannigan/icp/blob/master/icp.py

try this later: https://github.com/agnivsen/icp/blob/master/basicICP.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.0001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    print("prev_error : ", prev_error)
    return T, distances, i



def icp_ransac(A, B, init_pose=None, max_iterations=20, tolerance=0.001):

    #T_init, _, _ = icp(A, B, init_pose, max_iterations, tolerance)
    src_pc = np.copy(A)
    dst_pc = np.copy(B)

    # apply the initial pose estimation
    if init_pose is not None:
        src_pc = Procrustes.transform_xyz(src_pc, init_pose)

        # xyz_h = np.hstack([src_pc, np.ones((len(src_pc), 1))])  # homogenize 3D pointcloud
        # xyz_h = np.dot(T_init, xyz_h.T).T
        # xyz_h =  xyz_h[:, :3]
    
    # estimate without ransac, i.e. using all
    # point correspondences
    naive_model = Procrustes()
    naive_model.estimate(src_pc, dst_pc)
    transform_naive = naive_model.params
    mse_naive = np.sqrt(naive_model.residuals(src_pc, dst_pc).mean())
    print("mse naive: {}".format(mse_naive))

    # estimate with RANSAC
    ransac = RansacEstimator(
        min_samples=3,
        residual_threshold=(0.001)**2,
        max_trials=100,
    )
    ret = ransac.fit(Procrustes(), [src_pc, dst_pc])
    transform_ransac = ret["best_params"]
    inliers_ransac = ret["best_inliers"]
    
    print("inliers_ransac: ", np.count_nonzero(inliers_ransac))
    mse_ransac = np.sqrt(Procrustes(transform_ransac).residuals(src_pc, dst_pc).mean())
    print("mse ransac all: {}".format(mse_ransac))
    mse_ransac_inliers = np.sqrt(
        Procrustes(transform_ransac).residuals(src_pc[inliers_ransac], dst_pc[inliers_ransac]).mean())
    print("mse ransac inliers: {}".format(mse_ransac_inliers))  
    
    return transform_ransac,   inliers_ransac, 0


def transform_from_rotm_tr(rotm, tr):
  transform = np.eye(4)
  transform[:3, :3] = rotm
  transform[:3, 3] = tr
  return transform

class Procrustes:
  """Determines the best rigid transform [1] between two point clouds.
  References:
    [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
  """
  def __init__(self, transform=None):
    self._transform = transform

  def __call__(self, xyz):
    return Procrustes.transform_xyz(xyz, self._transform)

  @staticmethod
  def transform_xyz(xyz, transform):
    """Applies a rigid transform to an (N, 3) point cloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
    xyz_t_h = (transform @ xyz_h.T).T  # apply transform
    return xyz_t_h[:, :3]

  def estimate(self, X, Y):
    # find centroids
    X_c = np.mean(X, axis=0)
    Y_c = np.mean(Y, axis=0)

    # shift
    X_s = X - X_c
    Y_s = Y - Y_c

    # compute SVD of covariance matrix
    cov = Y_s.T @ X_s
    u, _, vt = np.linalg.svd(cov)

    # determine rotation
    rot = u @ vt
    if np.linalg.det(rot) < 0.:
      vt[2, :] *= -1
      rot = u @ vt

    # determine optimal translation
    trans = Y_c - rot @ X_c

    self._transform = transform_from_rotm_tr(rot, trans)

  def residuals(self, X, Y):
    """L2 distance between point correspondences.
    """
    Y_est = self(X)
    sum_sq = np.sum((Y_est - Y)**2, axis=1)
    return sum_sq

  @property
  def params(self):
    return self._transform



class RansacEstimator:
  """Random Sample Consensus.
  """
  def __init__(self, min_samples=None, residual_threshold=None, max_trials=100):
    """Constructor.
    Args:
      min_samples: The minimal number of samples needed to fit the model
        to the data. If `None`, we assume a linear model in which case
        the minimum number is one more than the feature dimension.
      residual_threshold: The maximum allowed residual for a sample to
        be classified as an inlier. If `None`, the threshold is chosen
        to be the median absolute deviation of the target variable.
      max_trials: The maximum number of trials to run RANSAC for. By
        default, this value is 100.
    """
    self.min_samples = min_samples
    self.residual_threshold = residual_threshold
    self.max_trials = max_trials

  def fit(self, model, data):
    """Robustely fit a model to the data.
    Args:
      model: a class object that implements `estimate` and
        `residuals` methods.
      data: the data to fit the model to. Can be a list of
        data pairs, such as `X` and `y` in the case of
        regression.
    Returns:
      A dictionary containing:
        best_model: the model with the largest consensus set
          and lowest residual error.
        inliers: a boolean mask indicating the inlier subset
          of the data for the best model.
    """
    best_model = None
    best_inliers = None
    best_num_inliers = 0
    best_residual_sum = np.inf

    if not isinstance(data, (tuple, list)):
      data = [data]
    num_data, num_feats = data[0].shape

    if self.min_samples is None:
      self.min_samples = num_feats + 1
    if self.residual_threshold is None:
      if len(data) > 1:
        data_idx = 1
      else:
        data_idx = 0
      self.residual_threshold = np.median(np.abs(
        data[data_idx] - np.median(data[data_idx])))

    for trial in range(self.max_trials):
      # randomly select subset
      rand_subset_idxs = np.random.choice(
        np.arange(num_data), size=self.min_samples, replace=False)
      rand_subset = [d[rand_subset_idxs] for d in data]

      # estimate with model
      model.estimate(*rand_subset)

      # compute residuals
      residuals = model.residuals(*data)
      residuals_sum = residuals.sum()
      inliers = residuals <= self.residual_threshold
      num_inliers = np.sum(inliers)

      # decide if better
      if (best_num_inliers < num_inliers) or (best_residual_sum > residuals_sum):
        best_num_inliers = num_inliers
        best_residual_sum = residuals_sum
        best_inliers = inliers

    # refit model using all inliers for this set
    if best_num_inliers == 0:
      data_inliers = data
    else:
      data_inliers = [d[best_inliers] for d in data]
    model.estimate(*data_inliers)

    ret = {
      "best_params": model.params,
      "best_inliers": best_inliers,
    }
    return ret






def icp_robust(A, B, init_pose=None, max_iterations=20, tolerance=0.0001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    search_rad = 0.5
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        matches = indices.flatten()  
        distances = distances.flatten()
        final_indices_dst = [matches[i] for i in range(len(matches)) if distances[i]<search_rad]  
        final_indices_src = [i for i in range(len(matches)) if distances[i]<search_rad]
          
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,final_indices_src].T, dst[:m,final_indices_dst].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    #print("prev_error : ", prev_error)
    return T, distances, i

