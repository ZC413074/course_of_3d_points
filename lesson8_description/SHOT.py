import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from ISS import iss, pointCloudShow


def visualFeatureDescription(fpfh, keypoint_idx):
    for i in range(len(fpfh)):
        x = [i for i in range(len(fpfh[i]))]
        y = fpfh[i]
        plt.plot(x, y, label=keypoint_idx[i])
    plt.title('Description Visualization for Keypoints')
    plt.legend(bbox_to_anchor=(1, 1),  
               loc="upper right",  
               ncol=1,  
               mode="None", 
               borderaxespad=0, 
               title="keypoints",  
               shadow=False,  
               fancybox=True)  
    plt.xlabel("label")
    plt.ylabel("shot")
    plt.show()


def computePointLRF(point_cloud, nearest_idx, keypoint_id, radius):  
    key_nearest_idx = list(set(nearest_idx[keypoint_id]) - set([keypoint_id]))
    key_nearest_idx = np.asarray(key_nearest_idx)
    points = np.asarray(point_cloud)
    keypoint = np.asarray(point_cloud)[keypoint_id]
    neighborhood_points = points[key_nearest_idx]

    # step1 计算带权重的M矩阵
    pi_p = point_cloud[key_nearest_idx]-keypoint
    r_di = (radius - np.linalg.norm(pi_p, axis=1))[:,None]
    if(np.sum(r_di)==0):
        return []
    aa=r_di*pi_p
    M = np.dot((r_di*pi_p).transpose(),pi_p)/np.sum(r_di, axis=0)
    
    # step2 SVD 分解M矩阵，得到初始的x、y、z
    eigenvetors,eigenvalues,eigenvetorsT = np.linalg.svd(M)  
    eigval_sort_index = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[eigval_sort_index] 
    eigenvetors = eigenvetors[:,eigval_sort_index] 
    x = eigenvetors[:,0]
    y = eigenvetors[:,1]
    z = eigenvetors[:,2]

    # step3 sign disambiguation
    direction_x = np.dot(pi_p, x)
    print(np.sum(direction_x>0))
    if(np.sum(direction_x>=0) < np.sum(direction_x<0)):
        x = -x
    direction_y = np.dot(pi_p, y)
    print(np.sum(direction_y>0))
    if(np.sum(direction_y>=0)<np.sum(direction_y,0)):
        y = -y
    z = np.cross(x, y)
    xyz = np.asarray([x,y,z])
    return xyz


def descriptorSHOT(point_cloud, point_cloud_normals, nearest_idx, keypoint_id, radius, Bin = 11):
    key_nearest_idx = list(set(nearest_idx[keypoint_id]) - set([keypoint_id]))
    key_nearest_idx = np.asarray(key_nearest_idx)
    normals = np.asarray(point_cloud_normals)
    # parameters
    azimuth = 8
    elevation = 2
    radial = 2

    # step1 计算关键点的LRF with sign disambiguation
    xyz_vector = computePointLRF(point_cloud, nearest_idx, keypoint_id, radius)
    print(xyz_vector[0])

    # step2 计算cos theta
    keypoint_normal = normals[keypoint_id]
    neighborhood_points_normal = normals[key_nearest_idx]
    cos_theta = np.dot(neighborhood_points_normal,keypoint_normal)
    cos_theta[cos_theta<-1.0]=-1.0
    cos_theta[cos_theta> 1.0]= 1.0
    cos_theta = (1.0+cos_theta)*Bin/2

    # step3 插值划分区域
    pi_p = point_cloud[nearest_idx]-keypoint
    distance = np.linalg.norm(pi_p, axis=1)[:,None]
    x_in_feat_ref = np.dot(pi_p,xyz_vector[0])
    y_in_feat_ref = np.dot(pi_p,xyz_vector[1])
    z_in_feat_ref = np.dot(pi_p,xyz_vector[2])
    if(np.abs(x_in_feat_ref)<1E-30):
        x_in_feat_ref=0
    if(np.abs(y_in_feat_ref)<1E-30):
        y_in_feat_ref=0
    if(np.abs(z_in_feat_ref)<1E-30):
        z_in_feat_ref=0

    return []


if __name__ == '__main__':

    env_dist = os.environ
    dataset_path = env_dist.get('DATASET_INSTALL_PATH')
    root_dir = dataset_path + "/modelnet40_normal_resampled"
    dirs = os.listdir(root_dir)
    for path in dirs:
        filename = os.path.join(root_dir, path, path+'_0001.txt')  # 默认使用第一个点云
        if not os.path.exists(filename):
            continue
        print(filename)
        # step1 read point_cloud from txt
        point_original = np.loadtxt(filename, dtype="float64", delimiter=",")
        point_cloud = point_original[:, 0:3]
        point_cloud_normals = point_original[:, 3:6]

        # step2 detect keypoints from point clouds：ISS
        print(point_cloud.shape)
        feature_idx = iss(point_cloud)
        print("feature_idx:", feature_idx)
        feature_point = point_cloud[feature_idx]
        pointCloudShow(point_cloud, feature_point)

        # step3  build kdtree and compute RNN
        leaf_size = 4
        radius = 0.05
        search_tree = KDTree(point_cloud, leaf_size)
        nearest_idx = search_tree.query_radius(
            point_cloud, radius)  # 求解每个点的最邻近点

        # step4 description：FPFH
        Bin = 5
        radius2=radius*2
        SHOT = np.asarray([descriptorSHOT(point_cloud, point_cloud_normals, nearest_idx,
                                    keypoint_id, radius2, Bin) for keypoint_id in feature_idx])

        # step5 show SHOT 
        visualFeatureDescription(SHOT, feature_idx)

