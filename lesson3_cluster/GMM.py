# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import KMeans
plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=500, tolerance = 0.00001):
        self.n_clusters_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

        # 屏蔽开始
        # 更新W
        self.posteriori_ = None

        # 更新pi
        self.prior_ = None

        # 更新Mu mean
        self.mu_ = None

        # 更新Var covariance
        self.cov_ = None

        # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # step1: initial the attribute of gmm by kmeans
        k_means = KMeans.K_Means(self.n_clusters_)
        k_means.fit(data)
        self.mu_ = np.asarray(k_means.centers_)
        print(self.n_clusters_)
        self.prior_ = np.asarray(
            [1/self.n_clusters_]*self.n_clusters_).reshape(self.n_clusters_, 1)
        self.posteriori_ = np.zeros((self.n_clusters_, len(data)))
        self.cov_ = np.asarray([eye(2, 2)]*self.n_clusters_)
        # step2：iteration
        Likelihood_value_before = -inf
        for i in range(self.max_iter_):
        # step3: E-step   generate probability density distribution for every point and normalize
            print("gmm iterator:",i)
            for k in range(self.n_clusters_):
                self.posteriori_[k] = multivariate_normal.pdf(x=data, mean=self.mu_[k], cov=self.cov_[k])  
            self.posteriori_ = np.dot(
                diag(self.prior_.ravel()), self.posteriori_)
            self.posteriori_ /= np.sum(self.posteriori_, axis=0)
            #posteriori=np.asarray(self.posteriori_)
            #print(posteriori.shape)
        # step4: M-step   update the parameters of generate probability density distribution for every point int E-step and stop when reached threshold
            self.Nk_ = np.sum(self.posteriori_, axis=1)
            self.mu_ = np.asarray([np.dot(self.posteriori_[k], data) / self.Nk_[k] for k in range(self.n_clusters_)])
            self.cov_ = np.asarray([np.dot((data-self.mu_[k]).T, np.dot(np.diag(self.posteriori_[k].ravel( )), data-self.mu_[k])) / self.Nk_[k] for k in range(self.n_clusters_)])
            self.prior_ = np.asarray( [self.Nk_ / self.n_clusters_]).reshape(self.n_clusters_, 1)
            Likelihood_value_after = np.sum(np.log(self.posteriori_))
            print(Likelihood_value_after - Likelihood_value_before)
            if np.abs(Likelihood_value_after - Likelihood_value_before) < self.tolerance_ * self.n_clusters_:
                break
            Likelihood_value_before=np.copy(Likelihood_value_after)
        self.fitted = True
        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = []
        if not self.fitted:
            print('Unfitter. ')
            return result
        result = np.argmax(self.posteriori_, axis=0)  
        return result

        # 屏蔽结束

        # 生成仿真数据


def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.title(u"scatter before gmm")
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5,  c='r')
    plt.scatter(X2[:, 0], X2[:, 1], s=5,  c='r')
    plt.scatter(X3[:, 0], X3[:, 1], s=5,  c='r')
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化
    K = 3

    # visualize:
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    cluster = [[] for i in range(K)]  # 用于分类所有数据点
    for i in range(len(X)):
        if cat[i] == 0:
            cluster[0].append(X[i])
        elif cat[i] == 1:
            cluster[1].append(X[i])
        elif cat[i] == 2:
            cluster[2].append(X[i])
    clusters = np.asarray(cluster)
    clusters1 = np.asarray(clusters[0])
    clusters2 = np.asarray(clusters[1])
    clusters3 = np.asarray(clusters[2])
    plt.figure(figsize=(10, 8))    
    plt.axis([-10, 15, -5, 15])
    plt.title(u"scatter after gmm")
    plt.scatter(clusters1[:, 0], clusters1[:, 1], c="r")
    plt.scatter(clusters2[:, 0], clusters2[:, 1], c="b")
    plt.scatter(clusters3[:, 0], clusters3[:, 1], c="y")
    plt.show()
