import os
import random
from matplotlib import axis
import open3d as o3d
from re import search
import struct
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time 
from ISS import iss, pointCloudShow
from FPFH import describe


def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.reshape(data_np, (int(data_np.shape[0]/6), 6))


def visualFeatureDescription(fpfh, keypoint_idx):
    for i in range(len(fpfh)):
        x = [i for i in range(len(fpfh[i]))]
        y = fpfh[i]
        plt.plot(x, y, label=keypoint_idx[i])
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

def evalute_icp(source, target, source_normal, target_normal, voxel_size, initial_transformation):
    transformation = compute_transformation(source, target)
    R, t = transformation[0:3, 0:3], transformation[0:3, 3][:, None]
    R_initial, t_initial = initial_transformation[0:3, 0:3], initial_transformation[0:3, 3][:, None]
    if(np.sum(abs(R-R_initial))>1 and np.sum(abs(t-t_initial))>1):
        return None
    source_to_target = np.dot(R,source.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        E= np.sum(np.linalg.norm(target, source_to_target, axis=1), axis=0)/source.shape[0]
        E[~np.isfinite(E)] = 0
    if(E>10):
        return None
    return  transformation

def compute_transformation(source, target):
    transformation = np.zeros((4, 4))
    mean_p = np.mean(source, axis=0)
    mean_q = np.mean(target, axis=0)
    p = source - mean_p
    q = target - mean_q
    u,sigma,v_t=np.linalg.svd(np.dot(q.T, p))
    r = np.dot(u, v_t)
    t = mean_q.T - np.dot(r, mean_p.T)
    transformation[0:3,0:3] = r
    transformation[0:3,3] = t
    transformation[3,3] = 1.0
    return transformation


def iter_match(source, target, source_normal, target_normal, source_feature_index, target_feature_index, variables, voxel_size):
    source_index = variables[:,0]
    target_index = variables[:,1]
    points_source = np.asarray(source[source_feature_index[source_index],:])
    points_target = np.asarray(target[target_feature_index[target_index],:])

    normals_source = np.asarray(source_normal)[source_feature_index[source_index],:]
    normals_target = np.asarray(target_normal)[target_feature_index[target_index],:]
    normal_cos_distances = (normals_source*normals_target).sum(axis = 1)
    is_valid_normal_match = np.all(normal_cos_distances >= np.cos(90))
    if not is_valid_normal_match:
        return None

    transformation = compute_transformation(points_source, points_target)
    #deviation：偏差  区分 inline outline 通过 距离判断
    R, t = transformation[0:3, 0:3], transformation[0:3, 3][:, None]
    deviation = np.linalg.norm(points_target.T - np.dot(R, points_source.T) - t, axis = 0)
    print("max deviation:",np.max(deviation, axis=0))
    is_valid_correspondence_distance = np.all(deviation <= voxel_size*4)
    return transformation if is_valid_correspondence_distance else None

def compute_initial_pose_based_feature(source, target, source_normal, target_normal, source_feature_index, target_feature_index, source_fpfh, target_fpfh, voxel_size):
    
    # step1 build pairs based knn
    leaf_size = 4
    radius = voxel_size*1.5
    target_fpfh_tree = KDTree(target_fpfh, leaf_size)
    pairs = []
    for i in range(source_fpfh.shape[0]):
        query_data = source_fpfh[i,:][:, None]
        _, query_index = target_fpfh_tree.query(query_data.T, k=1)
        pairs.append([i,query_index[0,0]]) 
    pairs = np.asarray(pairs)

    # step2 pick 4 points from pairs
    N, _ = pairs.shape
    idx_matches = np.arange(N)
    variables_generator = (pairs[np.random.choice(idx_matches, 4, replace=False)] for _ in iter(int, 1))

    validator = lambda variables: iter_match(source, target,  source_normal, target_normal, source_feature_index, target_feature_index, variables, voxel_size)
    i=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for transformation in map(validator, variables_generator):
            i=i+1
            print(i,"th:",":",transformation)
            if not (transformation is None):
                break
    
    return transformation

def icp_feature(source_down, target_down,source_normal,target_normal, source_feature_index, target_feature_index, source_fpfh, target_fpfh, voxel_size):

    # step1 compute initial pose based on feature
    transformation = compute_initial_pose_based_feature(source_down, target_down,source_normal,target_normal, source_feature_index, target_feature_index, source_fpfh, target_fpfh, voxel_size)

    # step2 icp
    transformation = evalute_icp(source_down, target_down,source_normal,target_normal,voxel_size,transformation)
    print(transformation)
    return transformation


if __name__ == '__main__':

    a=np.asarray([[1,2,3],[4,5,6]])
    print("a",np.sum(a))

    env_dist = os.environ
    dataset_path = env_dist.get('DATASET_INSTALL_PATH')
    root_dir = dataset_path + "/shenlan_registration_dataset/registration_dataset"
    match_pair_txt = os.path.join(root_dir, 'reg_result.txt')
    with open(match_pair_txt) as txt:
        content = txt.readlines()  # 读全部行
        txt.close()
    for index, filename in enumerate(content):
        if(index == 0):
            continue
        filename = filename.split(",")
        path_filename_pair1 = os.path.join(
            root_dir, 'point_clouds', filename[0] + '.bin')  # 默认使用第一个点云
        path_filename_pair2 = os.path.join(
            root_dir, 'point_clouds', filename[1] + '.bin')  # 默认使用第一个点云
        if not (os.path.exists(path_filename_pair1) and os.path.exists(path_filename_pair2)):
            continue

        # step1 read point pair from bin to point and normals
        points_source = read_oxford_bin(path_filename_pair1)
        #np.savetxt("data0.txt",points_source)
        points_target = read_oxford_bin(path_filename_pair2)
        #np.savetxt("data1.txt",points_target)
        point_cloud_source = o3d.geometry.PointCloud()
        point_cloud_source.points = o3d.utility.Vector3dVector(points_source[:, 0:3])
        point_cloud_source.normals = o3d.utility.Vector3dVector(points_source[:, 3:])
        point_cloud_target = o3d.geometry.PointCloud()
        point_cloud_target.points = o3d.utility.Vector3dVector(points_target[:, 0:3])
        point_cloud_target.normals = o3d.utility.Vector3dVector(points_target[:, 3:])
        # o3d.visualization.draw_geometries([point_cloud_source])
        # o3d.visualization.draw_geometries([point_cloud_target])

        # step2 downsample
        original_voxel_size = 0.2
        points_source_dawnsample, _= point_cloud_source.remove_statistical_outlier(nb_neighbors=40,
                                                    std_ratio=1.0)
        points_target_dawnsample, _= point_cloud_target.remove_statistical_outlier(nb_neighbors=40,
                                                    std_ratio=1.0)
        #np.savetxt("data3.txt",np.asarray(points_source_dawnsample.points))
        #np.savetxt("data4.txt",np.asarray(points_target_dawnsample.points))
        points_source_dawnsample = points_source_dawnsample.voxel_down_sample(voxel_size=original_voxel_size*2)
        points_target_dawnsample = points_target_dawnsample.voxel_down_sample(voxel_size=original_voxel_size*2)
        #np.savetxt("data5.txt",np.asarray(points_source_dawnsample.points))
        #np.savetxt("data6.txt",np.asarray(points_target_dawnsample.points))
        # o3d.visualization.draw_geometries([points_source_dawnsample])
        # o3d.visualization.draw_geometries([points_target_dawnsample])


        # step3 iss 特征提取
        points_source_dawnsample_numpy = np.asarray(points_source_dawnsample.points)
        points_source_dawnsample_numpy_normal = np.asarray(points_source_dawnsample.normals)
        points_target_dawnsample_numpy = np.asarray(points_target_dawnsample.points)
        points_target_dawnsample_numpy_normal  = np.asarray(points_target_dawnsample.normals)
        points_source_iss = iss(data=points_source_dawnsample_numpy, radius=original_voxel_size*3, nms_radius = original_voxel_size*12)
        points_target_iss = iss(data=points_target_dawnsample_numpy, radius=original_voxel_size*3, nms_radius = original_voxel_size*12)
        print('source iss shape:',points_source_iss.shape)
        print('target iss shape:',points_target_iss.shape)
        pointCloudShow(points_source_dawnsample_numpy,points_source_dawnsample_numpy[points_source_iss])
        pointCloudShow(points_target_dawnsample_numpy,points_target_dawnsample_numpy[points_target_iss])

        # step4  build kdtree and compute RNN
        leaf_size = 4
        radius = original_voxel_size*4
        source_search_tree = KDTree(points_source_dawnsample_numpy, leaf_size)
        target_search_tree = KDTree(points_target_dawnsample_numpy, leaf_size)
        source_nearest_idx = source_search_tree.query_radius(points_source_dawnsample_numpy, radius)  # 求解每个点的最邻近点
        target_nearest_idx = target_search_tree.query_radius(points_target_dawnsample_numpy, radius)  # 求解每个点的最邻近点

        # step5 description：FPFH
        Bin = 5
        points_source_fpfh = np.asarray([describe(points_source_dawnsample_numpy, points_source_dawnsample_numpy_normal, source_nearest_idx,
                                    keypoint_id, radius, Bin) for keypoint_id in points_source_iss])
        points_target_fpfh = np.asarray([describe(points_target_dawnsample_numpy, points_target_dawnsample_numpy_normal, target_nearest_idx,
                                    keypoint_id, radius, Bin) for keypoint_id in points_target_iss])
        visualFeatureDescription(points_source_fpfh, points_source_iss)
        visualFeatureDescription(points_target_fpfh, points_target_iss)

        # step6 icp
        transformation = icp_feature(points_source_dawnsample_numpy, points_target_dawnsample_numpy, points_source_dawnsample_numpy_normal,points_target_dawnsample_numpy_normal, points_source_iss, points_target_iss, points_source_fpfh, points_target_fpfh, original_voxel_size*2)
        o3d.visualization.draw_geometries([point_cloud_target, point_cloud_source.transform(transformation)])
