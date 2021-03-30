# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import os
import struct
import numpy as np
import open3d as o3d
from scipy import special
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for _, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def plane_fit_ransac(data, param):
    plane_coefficients = []
    # step1 judge whether the number of the data is bigger than the number of the minimum points of the model
    max_inner_nums = np.copy(param['max_inner_nums'])
    sample_nums = param['minimum points of model']
    size_data = data.shape[0]
    if(size_data <= sample_nums):
        return np.asarray(plane_coefficients)

    # step2 compute the max iterations
    max_iteration = np.longlong(special.comb(size_data, sample_nums))
    iterations = np.copy(max_iteration)
    for _ in range(iterations):
        # step3 random pick the minimum points of the model
        minimum_model_index = np.random.randint(0, size_data-1, sample_nums)
        point1 = data[minimum_model_index[0]]
        point2 = data[minimum_model_index[1]]
        point3 = data[minimum_model_index[2]]
    # step4 juge the three point and  compute the plane equation
        vector_point11to2 = point2-point1
        vector_point13to2 = point2-point3
        plane_normal = np.cross(vector_point11to2, vector_point13to2)
        if(np.all(vector_point11to2 == 0) or np.all(vector_point13to2 == 0) or np.all(plane_normal == 0)):
            return np.asarray(plane_coefficients)
    # step5 compute the plane model with one point and normal  Ax+By+Cz+D=0, and compute distance of every point to the plane
        #A = plane_normal[0]
        #B = plane_normal[1]
        #C = plane_normal[2]
        #D = -plane_normal.dot(point1)
        distance = abs((data - point1).dot(plane_normal)) / \
            np.linalg.norm(plane_normal, axis=0)
    # step6 compute the probability of inner point and update the interations
        inner_index = distance < param['threshold_distance']
        inner_nums = np.sum(inner_index == True)
        inner_probability = inner_nums / size_data
        if(inner_nums > max_inner_nums):
            max_inner_nums = inner_nums
            iterations = np.log(1-inner_probability) / \
                np.log(1-pow(inner_probability, sample_nums))
            A = plane_normal[0]
            B = plane_normal[1]
            C = plane_normal[2]
            D = -plane_normal.dot(point1)
            plane_coefficients = [A, B, C, D]
        if(inner_probability > param['inner_ratio']):
            break
    print("iterations:", iterations)
    if(plane_coefficients[2] < 0):
        plane_coefficients = - np.asarray(plane_coefficients)
    return plane_coefficients


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, param):
    # 作业1
    # 屏蔽开始

    # step1 plane fit with ransac， and return to its coefficients
    plane_coefficients = plane_fit_ransac(data, param)

    # step2 segment the ground points and others
    ax_by_cz = data.dot(plane_coefficients[:3])
    distance = (
        ax_by_cz + plane_coefficients[3]) / np.linalg.norm(plane_coefficients[:3])
    #distance = np.c_[data, one].dot(plane_coefficients) / np.linalg.norm(plane_coefficients[:3])
    ground_index_1 = np.abs(distance) < param['threshold_distance']
    ground_index_2 = ax_by_cz < - plane_coefficients[3]
    ground_index = np.logical_or(ground_index_1, ground_index_2)
    segmengted_index = np.logical_not(ground_index)
    segmengted_cloud = data[segmengted_index]
    # 屏蔽结束
    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）


def clustering(data):
    # 作业2
    # 屏蔽开始
    clusters_index = []
    dbscan_algorithm = cluster.DBSCAN(eps=1.2)
    dbscan_algorithm.fit(data)
    clusters_index = dbscan_algorithm.labels_.astype(np.int)
    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）


def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               s=2, color=colors[cluster_index])
    plt.show()


def main():
    root_dir = 'F:\\迅雷下载\\KITTI\\data_object_velodyne\\training\\velodyne'  # 数据集路径
    dirs = os.listdir(root_dir)
    print(dirs[0])

    params = {'threshold_distance': 1.0,
              'inner_ratio': .8,
              'max_inner_nums': -1,
              'minimum points of model': 3
              }
    for name in dirs:
        filename = os.path.join(root_dir, name)
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)

        point3d_show1 = o3d.geometry.PointCloud()
        point3d_show1.points = o3d.utility.Vector3dVector(origin_points)
        o3d.visualization.draw_geometries([point3d_show1])

        index_up = origin_points[:, 2] < 2.8
        index_down = origin_points[:, 2] > -3
        origin_points = origin_points[index_up & index_down]

        segmented_points = ground_segmentation(
            data=origin_points, param=params)
        point3d_show = o3d.geometry.PointCloud()
        point3d_show.points = o3d.utility.Vector3dVector(segmented_points)
        o3d.visualization.draw_geometries([point3d_show])

        cluster_index = clustering(segmented_points)
        plot_clusters(segmented_points, cluster_index)


if __name__ == '__main__':
    main()
