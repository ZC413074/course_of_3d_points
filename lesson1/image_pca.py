import numpy as np
import cv2 as cv

if __name__ == "__main__":
    imagePath = '0.jpg'
    image = cv.imread(imagePath)
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rows,cols=image.shape
    cv.imshow("original image",image)
    cv.waitKey(0)
    mean_image=np.mean(image,axis=0)
    mean_image=np.tile(mean_image,(rows,1))
    normal_image=image-mean_image
    normal_image=np.array(normal_image)
    cov_image=np.dot(normal_image.T,normal_image)
    U,sigmas,VT=np.linalg.svd(cov_image)
    arg_sort=np.argsort(sigmas)
    sigmas = sigmas[arg_sort]
    U = U[:, arg_sort]
    image_output=np.dot(normal_image,U[:,:500])+mean_image[:,:500]
    image_output=image_output.astype(np.int)
    cv.imshow("pca image",image_output)
    cv.waitKey(0)