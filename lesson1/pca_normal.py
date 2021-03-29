# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud

visualization=False
# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1jm
    # 屏蔽开始
    # step1 normalize the data to be zeros mean
    point_cloud_normalized_by_column = data - np.sum(
        data, axis=0
    ) / np.size(data, axis=0)
    # step2 the covariance matrix:transpose(A)A
    point_cloud_covariance = np.dot(
        np.transpose(point_cloud_normalized_by_column), point_cloud_normalized_by_column
    )
    # step3 SVD to compute pca
    eigenvectors, eigenvalues, eigenvectorsT = np.linalg.svd(point_cloud_covariance)
    # 屏蔽结束
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    root_dir = '/home/zc/MyFiles/points/datas/modelnet40_normal_resampled' # 数据集路
    dirs=os.listdir(root_dir)
    #print(dirs)
    for path in dirs:
        filename = os.path.join(root_dir, path, path+'_0001.txt') # 默认使用第一个点云
        if not os.path.exists(filename):
            continue
        print(filename)
        # step1 read point_cloud from txt and show(option)
        point_cloud_original = np.loadtxt(filename, dtype="float64", delimiter=",")[:, 0:3]
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_original)

        # 从点云中获取点，只对点进行处理
        points = point_cloud_original
        print("total points number is:", points.shape[0])

        #用PCA分析点云主方向
        w, v = PCA(points)
        point_cloud_vector = v[:, 2]  # 点云主方向对应的向量
        print("the main orientation of this pointcloud is: ", point_cloud_vector)
        # 此处显示点云，显示PCA
        point = [[0, 0, 0], point_cloud_vector]
        lines = [[0, 1]]
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(point)
        line.lines = o3d.utility.Vector2iVector(lines)
        colors = [[1, 0, 0]]
        line.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud_o3d, line])

        # 循环计算每个点的法向量
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        normals = []
        # 作业2
        # 屏蔽开始
        for i in range(points.shape[0]):
            [_, index, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 8)  # pick 8 nearest points to compute normal
            k_nearest_point = points[index, :]  
            w, v = PCA(k_nearest_point)
            normals.append(v[:, 2])
            # 屏蔽结束
        # 此处把法向量存放在了normals中
        normals = np.array(normals, dtype=np.float64)
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([point_cloud_o3d,line],point_show_normal=True)


if __name__ == "__main__":
    main()
