import os
import random
from re import search
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import open3d as o3d
import time 
from ISS import iss, pointCloudShow


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

def execute_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):

    o3d.visualization.draw_geometries([source_down])
    # time.sleep(2)
    o3d.visualization.draw_geometries([target_down])
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh)
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, True,
    #     distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     3, 
    #     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds()
    print(result.transformation)
    return result


if __name__ == '__main__':

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
        # print(path_filename_pair1)
        # print(path_filename_pair2)
        if not (os.path.exists(path_filename_pair1) and os.path.exists(path_filename_pair2)):
            continue

        # step1 read point pair from bin to point and normals
        points_source = read_oxford_bin(path_filename_pair1)
        np.savetxt("data0.txt",points_source)
        points_target = read_oxford_bin(path_filename_pair2)
        point_cloud_source = o3d.geometry.PointCloud()
        point_cloud_source.points = o3d.utility.Vector3dVector(points_source[:, 0:3])
        point_cloud_source.normals = o3d.utility.Vector3dVector(points_source[:, 3:])
        point_cloud_target = o3d.geometry.PointCloud()
        point_cloud_target.points = o3d.utility.Vector3dVector(points_target[:, 0:3])
        point_cloud_target.normals = o3d.utility.Vector3dVector(points_target[:, 3:])
        # o3d.visualization.draw_geometries([point_cloud_source, point_cloud_target])

        # step2 downsample
        points_source_dawnsample = point_cloud_source.voxel_down_sample(voxel_size=0.05)
        points_target_dawnsample = point_cloud_target.voxel_down_sample(voxel_size=0.05)
        # np.savetxt("data1.txt", np.asarray(points_source_dawnsample.points))
        points_source_dawnsample, _= points_source_dawnsample.remove_statistical_outlier(nb_neighbors=40,
                                                    std_ratio=1.0)
        points_target_dawnsample, _= points_target_dawnsample.remove_statistical_outlier(nb_neighbors=40,
                                                    std_ratio=1.0)
        # o3d.visualization.draw_geometries([points_source_dawnsample])
        # o3d.visualization.draw_geometries([points_target_dawnsample])
        # step3 iss 特征提取
        points_radius = 0.5
        points_source_iss = o3d.geometry.keypoint.compute_iss_keypoints(points_source_dawnsample,
                                                                        salient_radius=points_radius,
                                                                        non_max_radius=points_radius*2,
                                                                        gamma_21=0.5,
                                                                        gamma_32=0.5)
        points_target_iss = o3d.geometry.keypoint.compute_iss_keypoints(points_target_dawnsample,
                                                                        salient_radius=points_radius,
                                                                        non_max_radius=points_radius*2,
                                                                        gamma_21=0.5,
                                                                        gamma_32=0.5)
        print('source iss shape:',np.asarray(points_source_iss.points).shape)
        print('target iss shape:',np.asarray(points_target_iss.points).shape)
        points_source_iss.paint_uniform_color([1.0, 0, 0.0])
        points_source_dawnsample.paint_uniform_color([0.5, 0.5, 0.5])
        points_target_iss.paint_uniform_color([1.0, 0, 0.0])
        points_target_dawnsample.paint_uniform_color([0.5, 0.5, 0.5])
        # ,point_show_normal=True)
        # o3d.visualization.draw_geometries([points_source_iss, points_source_dawnsample])
        # o3d.visualization.draw_geometries([points_target_iss, points_target_dawnsample])

        # step4 fpfh 特征点描述
        search_type = o3d.geometry.KDTreeSearchParamRadius(radius=points_radius*4)
        points_source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(points_source_iss, search_type)
        keypoint_index = range(points_source_fpfh.data.shape[0])
        # visualFeatureDescription(points_source_fpfh.data, keypoint_index)
        points_target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(points_target_iss, search_type)
        keypoint_index = range(points_target_fpfh.data.shape[0])
        # visualFeatureDescription(points_target_fpfh.data, keypoint_index)

        # step4 ransac icp
        result = execute_global_registration(points_source_dawnsample, points_target_dawnsample, points_source_fpfh, points_target_fpfh, points_radius*4)
        print(result)

        # # step2 detect keypoints from point clouds：ISS
        # print(point_cloud.shape)
        # feature_idx = iss(point_cloud)
        # print("feature_idx:", feature_idx)
        # feature_point = point_cloud[feature_idx]
        # pointCloudShow(point_cloud, feature_point)

        # # step3  build kdtree and compute RNN
        # leaf_size = 4
        # radius = 0.05
        # search_tree = KDTree(point_cloud, leaf_size)
        # nearest_idx = search_tree.query_radius(
        #     point_cloud, radius)  # 求解每个点的最邻近点

        # # step4 description：FPFH
        # Bin = 5
        # FPFH = np.asarray([describe(point_cloud, point_cloud_normals, nearest_idx,
        #                             keypoint_id, radius, Bin) for keypoint_id in feature_idx])

        # # step5 show FPFH
        # visualFeatureDescription(FPFH, feature_idx)

        # # step6 test similarity point's FPFH
        # test_keypoint_idx = [1834, 2727, 2818, 8357]
        # test_FPFH = np.asarray([describe(point_cloud, point_cloud_normals, nearest_idx,
        #                                  keypoint_id, radius, Bin) for keypoint_id in test_keypoint_idx])
        # visualFeatureDescription(test_FPFH, test_keypoint_idx)
