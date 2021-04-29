import os
import numpy as np
import random
import matplotlib.pyplot as plt

import open3d as o3d
from sklearn.neighbors import KDTree


# matplotlib显示点云函数
def Point_Cloud_Show(point_cloud, feature_point):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    ax.scatter(feature_point[:, 0], feature_point[:, 1], feature_point[:, 2],
               cmap='spectral', s=2, linewidths=5, alpha=1, marker=".", color='red')
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def compute_cov_eigval(point_cloud):
    x = np.asarray(point_cloud[:, 0])
    y = np.asarray(point_cloud[:, 1])
    z = np.asarray(point_cloud[:, 2])
    M = np.vstack((x, y, z))  # 每行表示一个属性， 每列代表一个点
    cov = np.cov(M)  # 使用每个点的坐标求解cov
    # 求解三个特征值，升序排列 linda1 < linda2 < linda3
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = eigval[np.argsort(-eigval)]  # 改为降序排列  linda1 > linda2 > linda3
    return eigval  # 返回特征值


def iss(data):
    # parameters
    eigvals = []
    feature = []
    T = set()  # T 关键点的集合
    linda3_threshold = None  # 阈值，初步筛选 ,各文件参数  airplane_0001:0.001; chair_0001:0.0001
    # 构建 kd_tree
    leaf_size = 4
    radius = 0.1              # 各文件参数  airplane_00001:0.1; chair_0001:0.1
    tree = KDTree(data, leaf_size)
    # step1 使用radius NN 得到n个初始关键点, threshold 阈值 ：每个radius内的linda大于某个数值
    nearest_idx = tree.query_radius(data, radius)
    for i in range(len(nearest_idx)):
        print(nearest_idx[i].shape)
    for i in range(len(nearest_idx)):
        eigvals.append(compute_cov_eigval(data[nearest_idx[i]]))
    eigvals = np.asarray(eigvals)  # 求解每个点在各自的 radius 范围内的linda
    print(eigvals)  # 打印所有的 特征值，供调试用
    # 根据linda3的数值 确定linda3_threshold(linda的阈值)
    # 阈值取大约 是所有linda3的 中值得5倍，  eg 为什么取5倍是个人调试决定，也可取1倍
    linda3_threshold = np.median(eigvals, axis=0)[2]*5
    print(linda3_threshold)
    for i in range(len(nearest_idx)):
        # compute_cov_eigval(data[nearest_idx[i]])[2] -> 每个radius 里的最小的特征值 linda3
        if eigvals[i, 2] > linda3_threshold:
            T.add(i)  # 获得初始关键点的索引
    print(T)  # 输出 初始关键点
    # step2   有 重叠(IOU)的 关键点群
    unvisited = T  # 未访问集合
    while len(T):
        unvisited_old = unvisited  # 更新访问集合
        core = list(T)[np.random.randint(0, len(T))]  # 从 关键点集T 中随机选取一个 关键点core
        # 把核心点标记为 visited,从 unvisited 集合中剔除
        unvisited = unvisited - set([core])
        visited = []
        visited.append(core)

        while len(visited):  # 遍历所有初始关键点
            new_core = visited[0]
            if new_core in T:
                # S : 当前 关键点(core) 的范围内所包含的其他关键点
                S = unvisited & set(nearest_idx[new_core])
                # print(T)
                # print(S)
                visited += (list(S))
                unvisited = unvisited - S
            visited.remove(new_core)  # new core 已做检测，去掉new core
        cluster = unvisited_old - unvisited  # cluster, 有 重叠(IOU)的 关键点群
        T = T - cluster  # 去掉该类对象里面包含的核心对象,差集
    # step3  NMS 非极大抑制，求解 一个关键点群的linda3最大 为  关键点
        cluster_linda3 = []
        for i in list(cluster):
            cluster_linda3.append(eigvals[i][2])  # 获取 每个关键点 的 linda3
        cluster_linda3 = np.asarray(cluster_linda3)
        NMS_OUTPUT = np.argmax(cluster_linda3)
        feature.append(list(cluster)[NMS_OUTPUT])  # 添加到 feature 特征点数组中
    # output
    return feature


if __name__ == '__main__':
    root_dir = 'F:/dataset/modelnet40_normal_resampled'  # 数据集路
    dirs = os.listdir(root_dir)
    # print(dirs)
    for path in dirs:
        filename = os.path.join(root_dir, path, path+'_0001.txt')  # 默认使用第一个点云
        if not os.path.exists(filename):
            continue
        print(filename)
        # step1 read point_cloud from txt and show(option)
        point_cloud = np.loadtxt(
            filename, dtype="float64", delimiter=",")[:, 0:3]
        print(point_cloud.shape)
        feature_idx = iss(point_cloud)
        #feature_point = point_cloud[feature_idx]
        # print(feature_point)
        # pointCloudShow(point_cloud,feature_point)
