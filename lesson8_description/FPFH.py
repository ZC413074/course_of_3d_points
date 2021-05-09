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
    plt.ylabel("fpfh")
    plt.show()


def getSpfh(point_cloud, point_cloud_normals, nearest_idx, keypoint_id, radius, Bin):   # single pfh

    points = np.asarray(point_cloud)
    keypoint = np.asarray(point_cloud)[keypoint_id]
    key_nearest_idx = list(set(nearest_idx[keypoint_id]) - set([keypoint_id]))
    key_nearest_idx = np.asarray(key_nearest_idx)

    # step1 计算u v w
    # step1.1 分别取出关键点与近邻点的法向量，并计算u
    keypoint_normal = np.asarray(
        point_cloud_normals[keypoint_id])  # keypoint 邻近点的法向量
    neighborhood_normal = np.asarray(point_cloud_normals[key_nearest_idx])
    u = keypoint_normal
    # step1.2 计算关键点与近邻点的向量，并计算v
    diff = points[key_nearest_idx] - keypoint
    diff /= np.linalg.norm(diff, ord=2, axis=1)[:, None]
    v = np.cross(u, diff)
    # step1.3 计算W
    w = np.cross(u, v)

    # step2 计算alpha  phi theta triplets
    alpha = np.multiply(v, neighborhood_normal).sum(axis=1)
    phi = np.multiply(u, diff).sum(axis=1)
    theta = np.arctan2(np.multiply(w, neighborhood_normal).sum(
        axis=1), (u * neighborhood_normal).sum(axis=1))

    # step3 计算直方图 histogram
    # step3.1  alpha histogram
    alpha_histogram = np.histogram(alpha, bins=Bin, range=(-1.0, +1.0))[0]
    alpha_histogram = alpha_histogram / alpha_histogram.sum()
    # step3.2  phi histogram
    phi_histogram = np.histogram(phi, bins=Bin, range=(-1.0, +1.0))[0]
    phi_histogram = phi_histogram / phi_histogram.sum()
    # step3.3  theta histogram
    theta_histogram = np.histogram(theta, bins=Bin, range=(-np.pi, +np.pi))[0]
    theta_histogram = theta_histogram / theta_histogram.sum()
    # step3.4  alpha+theta+phi histogram
    signature = np.hstack((alpha_histogram, phi_histogram, theta_histogram))
    return signature


def describe(point_cloud, point_cloud_normals, nearest_idx, keypoint_id, radius, Bin):   # single pfh
    # step1 计算关键点的SPFH
    keypoint_spfh = getSpfh(point_cloud, point_cloud_normals,
                            nearest_idx, keypoint_id, radius, Bin)

    # step2 计算关键点的RNN的带权重的SPFH
    # step2.1 计算权重
    points = np.asarray(point_cloud)
    keypoint = np.asarray(point_cloud)[keypoint_id]
    key_nearest_idx = list(set(nearest_idx[keypoint_id]) - set([keypoint_id]))
    key_nearest_idx = np.asarray(key_nearest_idx)  # np只接受list
    k = len(key_nearest_idx)
    W = 1.0 / np.linalg.norm(points[key_nearest_idx] - keypoint, ord=2, axis=1)
    # step2.2 计算RNN的SPFH
    neighborhood_spfh = np.asarray([getSpfh(
        point_cloud, point_cloud_normals, nearest_idx, i, radius, Bin) for i in key_nearest_idx])
    # step2.3 计算带权重的RNN的SPFH
    neighborhood_weight_spfh = 1.0 / (k) * np.dot(W, neighborhood_spfh)

    # step3 FPHF
    fpfh = keypoint_spfh + neighborhood_weight_spfh
    fpfh = fpfh / np.linalg.norm(fpfh)

    return fpfh


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
        FPFH = np.asarray([describe(point_cloud, point_cloud_normals, nearest_idx,
                                    keypoint_id, radius, Bin) for keypoint_id in feature_idx])

        # step5 show FPFH
        visualFeatureDescription(FPFH, feature_idx)

        # step6 test similarity point's FPFH
        test_keypoint_idx = [1834, 2727, 2818, 8357]
        test_FPFH = np.asarray([describe(point_cloud, point_cloud_normals, nearest_idx,
                                         keypoint_id, radius, Bin) for keypoint_id in test_keypoint_idx])
        visualFeatureDescription(test_FPFH, test_keypoint_idx)
