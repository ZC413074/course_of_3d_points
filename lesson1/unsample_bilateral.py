import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(x, sigma):
    gaussian_kernel_function = 1/(2*np.pi*sigma)*np.exp(-x/(2*sigma*sigma))
    return gaussian_kernel_function

def compute_gaussian_function(kernel_size, sigma):
    gaussian_function = np.zeros((kernel_size, kernel_size))
    half_kernel_size = int(np.floor(kernel_size*0.5))
    for i in range(-half_kernel_size, half_kernel_size+1, 1):
        for j in range(-half_kernel_size, half_kernel_size+1, 1):
            x = i*i+j*j
            gaussian_function[i+half_kernel_size, j +
                              half_kernel_size] = gaussian_kernel(x, sigma)
    gaussian_function = gaussian_function/np.sum(gaussian_function)
    return gaussian_function

def unsample_bilateral_filter(image, cloud_points, image_gaussain_scale, cloud_gaussain_scale, kernel_size):
    cloud_points_filted = np.zeros(cloud_points.shape)
    cloud_gaussain = compute_gaussian_function(
        kernel_size, cloud_gaussain_scale)
    # rgb2gray
    image = 0.299 * image[:, :, 2] + 0.587 * \
        image[:, :, 1] + 0.114 * image[:, :, 0]
    half_nr = int(np.floor(kernel_size*0.5))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i-half_nr < -1e-6 or i+half_nr >= image.shape[0]:
                continue
            if j-half_nr < -1e-6 or j+half_nr >= image.shape[1]:
                continue
            image_gaussain = [gaussian_kernel(
                x, image_gaussain_scale) for x in image[i-half_nr:i+half_nr+1, j-half_nr:j+half_nr+1]]
            image_gaussain = image_gaussain/np.sum(image_gaussain)
            cloud_points_region = cloud_points[i -
                                               half_nr:i+half_nr+1, j-half_nr:j+half_nr+1]
            cloud_gaussain_temp = np.copy(cloud_gaussain)
            cloud_gaussain_temp[cloud_points_region == -1] = 0
            if np.sum(cloud_gaussain_temp) == 0:
                continue
            weight = cloud_gaussain_temp*image_gaussain
            weight = weight/np.sum(weight)
            cloud_points_filted[i, j] = np.sum(weight*cloud_points_region)
    return cloud_points_filted

if __name__ == "__main__":
    data_root = r"F:\dataset\KITTI\data_depth_selection\depth_selection\val_selection_cropped"
    image_path = "image"
    depth_path = "velodyne_raw"
    data_name = "2011_09_26_drive_0002_sync_image_0000000005_image_02.png"
    imageBGR = cv2.imread(os.path.join(
        data_root, image_path, data_name), cv2.IMREAD_COLOR)
    depth_data = cv2.imread(os.path.join(data_root, depth_path, data_name.replace(
        image_path, depth_path, 1)), cv2.IMREAD_ANYDEPTH)
    point_cloud_filted = unsample_bilateral_filter(imageBGR, depth_data, 50.0, 50.0, 7)

    cv2.imshow("origin",depth_data)
    cv2.imshow("filted",point_cloud_filted)
    cv2.waitKey(0)
