# 文件功能： 实现 K-Means 算法
import random
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.000001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, datas):
        # 作业1
        # 屏蔽开始
        # step1 select k points as the center of cluster
        self.centers_ = datas[random.sample(range(datas.shape[0]), self.k_)]
        old_centers = np.copy(self.centers_)
        # step2 show the point with two center
        #plt.figure(figsize=(10, 10))
        #plt.title(u"scatter with center before kmeans")
        #ax3 = plt.axes(projection='3d')
        #ax3.scatter3D(datas[:, 0], datas[:, 1], datas[:, 2], c='g')
        #ax3.scatter3D(old_centers[:, 0], old_centers[:, 1],old_centers[:, 2], c='r')
        #plt.show()
        #plt.figure(figsize=(10, 10))
        #plt.title(u"scatter with center before kmeans")
        #plt.scatter(datas[:, 0], datas[:, 1], c='g')
        #plt.scatter(old_centers[:, 0], old_centers[:, 1], c='r')
        #plt.show()
        leaf_size = 1
        k = 1
        for i in range(self.max_iter_):
            # step3 knn find the nearest distance between query point with the centers
            labels = [[] for i in range(self.k_)]
            root = spatial.KDTree(self.centers_, leafsize=leaf_size)
            #print("kmeans iterator:",i)
            for i in range(datas.shape[0]):
                # return the nearest distance and order
                _, query_index = root.query(datas[i], k)
                #print(distance)
                #print(query_index)
                labels[query_index].append(datas[i])
            for i in range(self.k_):
                points = np.array(labels[i])
                self.centers_[i] = points.mean(axis=0)
            if np.sum(np.abs(self.centers_ - old_centers)) < self.tolerance_ * self.k_:
                break
            old_centers = np.copy(self.centers_)  
        #plt.figure(figsize=(10, 10))
        #plt.title(u"scatter with center after kmeans")
        #ax3 = plt.axes(projection='3d')
        #ax3.scatter3D(datas[:, 0], datas[:, 1],datas[:, 2], c='g')
        #ax3.scatter3D(old_centers[:, 0], old_centers[:, 1],old_centers[:, 2], c='r')
        #plt.show()
        #plt.figure(figsize=(10, 10))
        #plt.title(u"scatter with center after kmeans")
        #plt.scatter(datas[:, 0], datas[:, 1], c='g')
        #plt.scatter(old_centers[:, 0], old_centers[:, 1], c='r')
        #plt.show()
        self.fitted = True

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        if not self.fitted:
            print('Unfitter. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers_ - point, axis=1)
            result.append(np.argmin(diff))
        # 屏蔽结束
        return result


if __name__ == '__main__':

    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    cat = k_means.predict(x)
    color = np.array(['r', 'g'])
    cats = np.array(cat)
    print(cats.shape)
    plt.figure(figsize=(10, 10))
    plt.title(u"scatter clustter after kmeans")
    plt.scatter(x[:, 0], x[:, 1], c=color[cats])
    plt.show()
    print(cat)
