# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import os
import random
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, mode="centroid"):
    print(np.size(point_cloud,axis=0))
    if leaf_size<=0:
        print("the 'leaf_size' is not true , please input a positive integer")
        return []
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # step1 compute the min and max of the point set by column
    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)
    # step2 compute the dimension of the voxel grid
    d_x = (x_max - x_min) / leaf_size
    d_y = (y_max - y_min) / leaf_size
    d_z = (z_max - z_min) / leaf_size
    #step3 compute voxel index for each point
    h=[]
    count=0
    for point in point_cloud:
        h_x=np.floor((point[0]-x_min)/leaf_size)
        h_y=np.floor((point[1]-y_min)//leaf_size)
        h_z=np.floor((point[2]-z_min)//leaf_size)
        h.append(h_x+h_y*d_x+h_z*d_x*d_y)
    #step4 sort the point by index
    h = np.array(h, dtype=np.float64)
    h_order=np.argsort(h)
    h_ascending=h[h_order]
    #step5 iterate the sorted points
    for i in range(len(h_ascending)-1):
        if h_ascending[i]==h_ascending[i+1]:
            continue
        else:
            point_voxel_index=h_order[count:i+1]
            if mode=="centroid":
                points_voxel_centroid=np.mean(point_cloud[point_voxel_index],axis=0)
                filtered_points.append(points_voxel_centroid)
                count=i
            elif mode=="random":
                points_voxel_random=random.choice(point_cloud[point_voxel_index])
                filtered_points.append(points_voxel_random)
                count=i
            else:
                print("the mode is not true, please input the 'centroid' or 'random'")
                return []
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print(np.size(filtered_points,axis=0))
    return filtered_points


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
        o3d.visualization.draw_geometries([point_cloud_o3d],window_name="orininal point clouds") # 显示原始点云

        # 调用voxel滤波函数，实现滤波
        points=point_cloud_original
        filtered_cloud = voxel_filter(points, 0.05,'centroid')
        point_cloud_sampled = o3d.geometry.PointCloud()
        point_cloud_sampled.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_sampled],window_name="sampled point clouds")


if __name__ == "__main__":
    main()
