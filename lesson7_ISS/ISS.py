import os
import numpy as np
import random
import matplotlib.pyplot as plt

import open3d as o3d
from sklearn.neighbors import KDTree 


# matplotlib显示点云函数
def pointCloudShow(point_cloud,feature_point):
    plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='spectral', s=10, linewidths=0, alpha=1, marker=".")
    ax.scatter(feature_point[:, 0], feature_point[:, 1], feature_point[:, 2], cmap='spectral', s=10, linewidths=5, alpha=1,marker=".",color='red')
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def computeCovarianceEigval(nearest_point_cloud, nearest_distance):
    nearest_distance = np.expand_dims(nearest_distance, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nearest_point_cloud= nearest_point_cloud/(nearest_distance)
        nearest_point_cloud[~np.isfinite(nearest_point_cloud)] = 0
    nearest_point_cov = np.cov(nearest_point_cloud.transpose())
    cov=nearest_point_cov*sum(nearest_distance)
    eigenvalues,eigenvalues,eigenvaluesT = np.linalg.svd(cov)  
    eigval_sort_index = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[eigval_sort_index]  
    return  eigenvalues      #返回特征值

def iss(data, gama21=0.05, gama32=0.7, nms_radius=0.3):
    #parameters
    feature_values = []
    keypoints = []
    keypoints_index_after_nms = []

    # step1 创建kd树，并以半径r建立近邻搜索，输出每个点的r近邻点的index以及距离
    leaf_size = 4
    radius = 0.1             
    tree = KDTree(data, leaf_size)
    nearest_index, nearest_distance= tree.query_radius(data, radius, return_distance=True)

    # step2 每个点的近邻点并结合其距离的倒数作为权值，计算其协方差矩阵, 并按降序输出其特征值
    eigvals = []
    for i in range(len(nearest_index)):
        eigval = computeCovarianceEigval(data[nearest_index[i]], nearest_distance[i])
        eigvals.append(eigval)
    eigvals = np.asarray(eigvals)

    # step3 根据公式判断是否为特征点，三个特征值相差不大为特征点，但是当前的判断条件并不好 todo 20210430
    gama21 = np.median(eigvals[:,1]/ eigvals[:,0],axis=0)
    gama32 = np.median(eigvals[:,2]/ eigvals[:,1],axis=0)
    lamda = np.median(eigvals[:, 2], axis=0)
    for i in range(eigvals.shape[0]):
        if(eigvals[i,1]/eigvals[i,0] < gama21 and  eigvals[i,2]/eigvals[i,1] < gama32 and eigvals[i,2] > lamda and eigvals[i,0]>eigvals[i,1]>eigvals[i,2]):  
        #if( eigvals[i,2] > lamda and eigvals[i,0]> eigvals[i,1] > eigvals[i,2]):
           # step 3.1 判断为特征点后，将特征值的第三维度添加进feature_values，用于下一步的nms
            feature_values.append(eigval[2])
            keypoints.append(data[i])
    feature_values = np.asarray(feature_values)  
    keypoints = np.asarray(keypoints) 
    
    #step4 NMS, 用关键点，重新建一颗kdtee，然后找出邻域中feature_values最大的为特征点，删掉邻域内的其他伪特征点，重复上述步骤直到访问完所有的keypoints
    leaf_size = 8
    # step4.1 建kdtree
    keypoints_tree = KDTree(keypoints, leaf_size)
    while keypoints[~np.isnan(keypoints)].shape[0]:
        # step4.2 找当前feature_values中的最大值，作为最可能的特征点
        feature_index = np.argmax(feature_values)
        feature_point = keypoints[feature_index]
        feature_point = np.expand_dims(feature_point, axis=0)
        # step4.3 在keypoints查找当前特征点的邻域，并将这些邻域点置为none，以及对应的feature_values值为0
        nearest_index = keypoints_tree.query_radius(feature_point, nms_radius)
        keypoints_index_after_nms.append(feature_index)
        keypoints[feature_index] = np.nan
        keypoints[nearest_index[0]] = np.nan
        feature_values[feature_index] = 0
        feature_values[nearest_index[0]] = 0
    return  np.asarray(keypoints_index_after_nms)

if __name__ == '__main__':
    env_dist = os.environ
    dataset_path = env_dist.get('DATASET_INSTALL_PATH')
    root_dir =  dataset_path + "/modelnet40_normal_resampled"
    dirs = os.listdir(root_dir)
    for path in dirs:
        filename = os.path.join(root_dir, path, path+'_0001.txt')  # 默认使用第一个点云
        if not os.path.exists(filename):
            continue
        print(filename)
        # step1 read point_cloud from txt and show(option)
        point_cloud = np.loadtxt(filename, dtype="float64", delimiter=",")[:, 0:3]
        print(point_cloud.shape)
        feature_idx = iss(point_cloud)
        #print("feature_idx", feature_idx)
        feature_point = point_cloud[feature_idx]
        #print(feature_point)
        pointCloudShow(point_cloud,feature_point)