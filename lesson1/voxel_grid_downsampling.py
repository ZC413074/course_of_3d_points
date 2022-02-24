import os
import random
import numpy as np
import matplotlib.pyplot as plt


# def voxel_select_point(filtered_points):
#     return filtered_points

def voxel_grid_sample(point_cloud, grid_size, container_num, filter_type='random'):
    # point_cloud  Nx3
    assert filter_type == 'random' or filter_type == 'mean', 'filter type is not true!'

    filtered_points=[]

    min_point_cloud = np.min(point_cloud, 0)
    grid_dimension = (np.max(point_cloud, 0) - min_point_cloud)/grid_size
    hash_table_point = {}
    hash_table_index = {}
    for point in point_cloud:
        point_coord_index=np.floor(point-min_point_cloud)/grid_size
        h=point_coord_index[0]+grid_dimension[0]*point_coord_index[1]+point_coord_index[2]*grid_dimension[1]*grid_dimension[2]
        hash_index=np.mod(h,container_num)
        if not hash_index in hash_table_index:
            hash_table_index[hash_index]=h
            hash_table_point[hash_index]=[point]
        elif h==hash_table_index[hash_index]:
            hash_table_point[hash_index].append(point)
        else:
            if filter_type=='random':
                filtered_points.append(random.choice(hash_table_point[hash_index]))
            elif filter_type=='mean':
                filtered_points.append(list(np.mean(np.array(hash_table_point[hash_index]),0)))
            hash_table_index[hash_index]=h
            hash_table_point[hash_index]=[point]
    for hash_index in hash_table_index:
        if filter_type=='random':
            filtered_points.append(random.choice(hash_table_point[hash_index]))
        elif filter_type=='mean':
            filtered_points.append(list(np.mean(np.array(hash_table_point[hash_index]),0)))
    hash_table_point.clear()
    hash_table_index.clear()
    return np.array(filtered_points)
if __name__ == "__main__":
    point_cloud_original = np.loadtxt('lesson1/chair_0001.txt', dtype="float64", delimiter=",")[:, 0:3]
    point_cloud_sampled = voxel_grid_sample(point_cloud_original,[0.001,0.001,0.001],100)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(point_cloud_sampled[:, 0], point_cloud_sampled[:, 1], point_cloud_sampled[:, 2],s=2)
    ax1 = plt.figure().add_subplot(111, projection='3d')
    ax1.scatter(point_cloud_original[:, 0], point_cloud_original[:, 1], point_cloud_original[:, 2],s=2)

    plt.show()
    
