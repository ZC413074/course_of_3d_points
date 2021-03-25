# 文件功能：实现 Spectral 谱聚类 算法

import numpy as np
import scipy 
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.neighbors import kneighbors_graph
import KMeans
plt.style.use('seaborn')

class Spectral(object):
    def __init__(self, n_clusters, n_neighbors = 10):
        self.n_clusters_ = n_clusters
        self.n_neighbors_ = n_neighbors
        self.weight_ = None
        self.degree_ = None
        self.laplacians_ = None
        self.eigen_vector_ = None

    def fit(self, data):

        # step1 construct weight matrix for every point
        #weight = kneighbors_graph(data, n_neighbors = self.n_neighbors_,mode='distance', include_self = False)
        weight = kneighbors_graph(data, n_neighbors = self.n_neighbors_,mode='connectivity', include_self = False)
        weight = 0.5 * (weight + weight.T)
        self.weight_ = weight.toarray()
        self.degree_ = np.diag(np.sum(self.weight_, axis = 0).ravel()) 

        # step2 construct Laplacian matrix for every point, and normalize
        self.laplacians_ = self.degree_ - self.weight_
        #unit_arrary = np.ones([data.shape[0],data.shape[0]],dtype=np.float64)
        #with np.errstate(divide='ignore'): 
        #    degree_nor = unit_arrary/np.sqrt(self.degree_) 
        #    degree_nor[self.degree_ == 0] = 0
        degree_nor=np.sqrt(np.linalg.inv(self.degree_))
        self.laplacians_ = np.dot(degree_nor, self.laplacians_)  
        self.laplacians_ = np.dot(self.laplacians_, degree_nor)#normalize

        #step3 compute minimun k eigenvalues corresponding to eigenvectors and normalize
        eigen_values, eigen_vector  = np.linalg.eigh(self.laplacians_)
        sort_index = eigen_values.argsort()
        eigen_vector = eigen_vector[:,sort_index]
        self.eigen_vector_ = np.asarray([eigen_vector[:,i] for i in range(self.n_clusters_)]).T
        #self.eigen_vector_ /= np.sqrt(np.sum(self.eigen_vector_**2, axis = 1)).reshape(data.shape[0], 1 )
        self.eigen_vector_ /= np.linalg.norm(self.eigen_vector_, axis=1).reshape(data.shape[0], 1 )
        
        #step4  kmeans with eigenvectors 
        spectral_kmeans = KMeans.K_Means(n_clusters=self.n_clusters_)
        spectral_kmeans.fit(self.eigen_vector_)
        spectral_label = spectral_kmeans.predict(self.eigen_vector_)
        self.label_ = spectral_label
        self.fitted = True

    def predict(self, data):
        result = []
        if not self.fitted:
            print('Unfitter. ')
            return result
        return np.copy(self.label_)



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
    plt.axis([-10, 15, -5, 15])
    plt.title(u"scatter clustter before spectral")
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
    #X = np.array([[1, 2], [2, 3], [5, 8], [8, 8], [1, 6], [9, 11]])

    spectral = Spectral(n_clusters=3)
    K = 3
    spectral.fit(X)
    cat = spectral.predict(X)
    print(cat)
    cluster =[[] for i in range(K)]
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
    plt.title(u"scatter clustter after spectral")   
    plt.axis([-10, 15, -5, 15])
    plt.scatter(clusters1[:, 0], clusters1[:, 1], s=5, c="r")
    plt.scatter(clusters2[:, 0], clusters2[:, 1], s=5, c="b")
    plt.scatter(clusters3[:, 0], clusters3[:, 1], s=5, c="y")
    plt.show()
