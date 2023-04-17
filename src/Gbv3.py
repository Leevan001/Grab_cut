# -*- coding = utf-8 -*-
# @Time : 2023/4/13 12:22
# @Author : CQU20205644
# @File : Gbv1.py
# @Software : PyCharm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pdb
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import time
from sklearn.cluster import KMeans
import cv2
import networkx as nx
from scipy.stats import multivariate_normal
import maxflow
import time


# class GMM:
#     def __init__(self, n_components=5, max_iter=10, tol=0):
#         self.n_components = n_components
#         self.max_iter = max_iter
#         self.tol = tol
#         self.weights_ = None
#         self.means_ = None
#         self.covariances_ = None
#         self.clustered_data_ = None  # 用于存储聚类结果
#
#     def fit(self, X):
#         # 进行聚类
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.tol)
#         compactness, labels, centers = cv2.kmeans(X, self.n_components, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
#
#         # 将聚类好的点分别放到五个类，并分别计算他们的高斯分布模型和权重
#         self.weights_ = []
#         self.means_ = []
#         self.covariances_ = []
#         self.clustered_data_ = {}
#         for i in range(self.n_components):
#             # 找到标签为 i 的像素的索引
#             class_indices = np.where(labels == i)[0]
#
#             # 从原始数据中提取属于该类的像素数据
#             data_i = X[class_indices]
#
#             # 计算该类的高斯分布模型和权重
#             weight_i = len(class_indices) / len(labels)
#             mean = np.mean(data_i, axis=0)
#             cov = np.cov(data_i.T)
#             self.weights_.append(weight_i)
#             self.means_.append(mean)
#             self.covariances_.append(cov)
#             # 将该类的数据点存储到聚类结果字典中
#             self.clustered_data_[i] = data_i
#         print("weight: ", self.weights_)
#
#     def clear(self):
#         for key in self.clustered_data_:
#             self.clustered_data_[key] = np.zeros((0, 3))
#
#     def multivariate_normal(self, X, mu, cov):
#         diff = X - mu
#         det = np.linalg.det(cov)
#         mult = np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
#         p = 1.0 / np.sqrt(det) * np.exp(-0.5 * mult)
#         return p
#
#     def add_points_to_cluster(self, point):
#         # 计算新数据点属于每个高斯分布模型的概率
#         probabilities = self.predic_each_proba(point)
#
#         # 将新的数据点添加到概率最大的高斯分布模型中
#         max_prob_idx = np.argmax(probabilities)
#         self.clustered_data_[max_prob_idx] = np.vstack((self.clustered_data_[max_prob_idx], point))
#
#     def predic_each_proba(self, X):
#         # 计算每个数据点属于每个高斯分布模型的概率
#         probabilities = np.zeros(self.n_components)
#         for i in range(self.n_components):
#             probabilities[i] = self.multivariate_normal(X, self.means_[i], self.covariances_[i])
#         return probabilities
#
#     def update_cluster_models(self):
#         for i in range(self.n_components):
#             # 获取当前聚类簇的数据点
#             data_i = self.clustered_data_[i]
#
#             # 计算当前聚类簇的权重
#             weight_i = len(data_i) / len(self.clustered_data_.values())
#
#             # 计算当前聚类簇的均值和协方差
#             mean = np.mean(data_i, axis=0)
#             cov = np.cov(data_i.T)
#
#             # 更新当前聚类簇的高斯分布模型和权重
#             self.weights_[i] = weight_i
#             self.means_[i] = mean
#             self.covariances_[i] = cov
#         print("weight: ", self.weights_)
#
#     def predict_proba(self, X):
#         # 计算每个样本属于每个类的概率
#         probs = np.zeros(self.n_components)
#         total = 0
#         for i in range(self.n_components):
#             prob_i = self.multivariate_normal(X, self.means_[i], self.covariances_[i])
#             probs[i] = prob_i
#             total = total + self.weights_[i] * prob_i
#         return total
class GMM:
    def __init__(self, n_components=5, max_iter=10, tol=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.det = None
        self.inv = None
        self.clustered_data_ = None  # 用于存储聚类结果

    def fit(self, X):
        # 进行聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.tol)
        compactness, labels, centers = cv2.kmeans(X, self.n_components, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # 将聚类好的点分别放到五个类，并分别计算他们的高斯分布模型和权重
        self.weights_ = []
        self.means_ = []
        self.covariances_ = []
        self.det = []
        self.clustered_data_ = {}
        self.inv = []
        for i in range(self.n_components):
            # 找到标签为 i 的像素的索引
            class_indices = np.where(labels == i)[0]

            # 从原始数据中提取属于该类的像素数据
            data_i = X[class_indices]

            # 计算该类的高斯分布模型和权重
            weight_i = len(class_indices) / len(labels)
            mean = np.mean(data_i, axis=0)
            cov = np.cov(data_i.T)
            self.weights_.append(weight_i)
            self.means_.append(mean)
            self.covariances_.append(cov)
            # 将该类的数据点存储到聚类结果字典中
            self.clustered_data_[i] = data_i
            self.det.append(np.linalg.det(cov))
            self.inv.append(np.linalg.inv(cov))
        print("weight: ", self.weights_)

    def clear(self):
        for key in self.clustered_data_:
            self.clustered_data_[key] = np.zeros((0, 3))

    def multivariate_normal(self, X, mu, det, inv_):
        diff = X - mu
        mult = np.dot(np.dot(diff.T, inv_), diff)
        p = 1.0 / np.sqrt(det) * np.exp(-0.5 * mult)
        return p

    def add_points_to_cluster(self, point):
        # 计算新数据点属于每个高斯分布模型的概率
        probabilities = self.predic_each_proba(point)

        # 将新的数据点添加到概率最大的高斯分布模型中
        max_prob_idx = np.argmax(probabilities)
        self.clustered_data_[max_prob_idx] = np.vstack((self.clustered_data_[max_prob_idx], point))

    def predic_each_proba(self, X):
        # 计算每个数据点属于每个高斯分布模型的概率
        probabilities = np.zeros(self.n_components)
        for i in range(self.n_components):
            probabilities[i] = self.multivariate_normal(X, self.means_[i], self.det[i], self.inv[i])
        return probabilities

    def update_cluster_models(self):
        for i in range(self.n_components):
            # 获取当前聚类簇的数据点
            data_i = self.clustered_data_[i]

            # 计算当前聚类簇的权重
            weight_i = len(data_i) / len(self.clustered_data_.values())

            # 计算当前聚类簇的均值和协方差
            mean = np.mean(data_i, axis=0)
            cov = np.cov(data_i.T)
            det_ = np.linalg.det(cov)
            # 更新当前聚类簇的高斯分布模型和权重
            self.weights_[i] = weight_i
            self.means_[i] = mean
            self.covariances_[i] = cov
            self.det[i] = det_
            self.inv[i] = np.linalg.inv(cov)
        print("weight: ", self.weights_)

    def predict_proba(self, X):
        # 计算每个样本属于每个类的概率
        probs = np.zeros(self.n_components)
        total = 0
        for i in range(self.n_components):
            prob_i = self.multivariate_normal(X, self.means_[i], self.det[i], self.inv[i])
            probs[i] = prob_i
            total = total + self.weights_[i] * prob_i
        return total


def calcBeta(img):
    beta = 0
    rows, cols = img.shape[:2]
    diff_left = img[:, :-1] - img[:, 1:]
    diff_up = img[:-1, :] - img[1:, :]
    diff_upleft = img[:-1, :-1] - img[1:, 1:]
    diff_upright = img[:-1, 1:] - img[1:, :-1]
    beta = np.sum(diff_left ** 2) + np.sum(diff_up ** 2) + np.sum(diff_upleft ** 2) + np.sum(diff_upright ** 2)
    if beta <= np.finfo(float).eps:
        beta = 0
    else:
        beta = 1.0 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
    #     print(type(beta))
    return beta


def calc_n_weights(img, beta, gamma):
    gamma_div_sqrt2 = gamma / np.sqrt(2.0)
    n, m, _ = img.shape
    left_w = np.zeros((n, m), dtype=np.float64)
    upleft_w = np.zeros((n, m), dtype=np.float64)
    up_w = np.zeros((n, m), dtype=np.float64)
    upright_w = np.zeros((n, m), dtype=np.float64)

    # 计算相邻像素之间的差异
    diff_left = img[:, :-1] - img[:, 1:]
    diff_up = img[:-1, :] - img[1:, :]
    diff_upleft = img[:-1, :-1] - img[1:, 1:]
    diff_upright = img[:-1, 1:] - img[1:, :-1]

    # 计算相邻像素之间的权重
    left_w[:, 1:] = gamma * np.exp(-beta * np.sum(diff_left ** 2, axis=2))
    upleft_w[1:, 1:] = gamma_div_sqrt2 * np.exp(-beta * np.sum(diff_upleft ** 2, axis=2))
    up_w[1:, :] = gamma * np.exp(-beta * np.sum(diff_up ** 2, axis=2))
    upright_w[1:, :-1] = gamma_div_sqrt2 * np.exp(-beta * np.sum(diff_upright ** 2, axis=2))

    return left_w, upleft_w, up_w, upright_w


def constructGCGraph(img, mask, bgdGMM, fgdGMM, lamda, leftW, upleftW, upW, uprightW):
    vtxCount = img.shape[0] * img.shape[1]
    edgeCount = (4 * vtxCount - 3 * (img.shape[0] + img.shape[1]) + 2)
    g = maxflow.Graph[float](vtxCount, edgeCount)
    nodelist = g.add_nodes(vtxCount)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = i * img.shape[1] + j
            color = img[i, j, :]
            if mask[i, j] == cv2.GC_PR_BGD or mask[i, j] == cv2.GC_PR_FGD:
                try:
                    fromSource = -np.log(bgdGMM.predict_proba(color) + np.finfo(np.float64).eps)
                    toSink = -np.log(fgdGMM.predict_proba(color) + np.finfo(np.float64).eps)
                except:
                    print("wrong(bgd): ", bgdGMM.predict_proba(color))
                    print("wrong(fgd): ", fgdGMM.predict_proba(color))

            elif mask[i, j] == cv2.GC_BGD:
                fromSource = lamda
                toSink = 0
            else:
                fromSource = 0
                toSink = lamda

            g.add_tedge(nodelist[idx], fromSource, toSink)

            if j > 0:
                w = leftW[i, j]
                g.add_edge(nodelist[idx], nodelist[idx - 1], w, w)
            if i > 0 and j > 0:
                w = upleftW[i, j]
                g.add_edge(nodelist[idx], nodelist[idx - img.shape[1] - 1], w, w)
            if i > 0:
                w = upW[i, j]
                g.add_edge(nodelist[idx], nodelist[idx - img.shape[1]], w, w)
            if i > 0 and j < img.shape[1] - 1:
                w = uprightW[i, j]
                g.add_edge(nodelist[idx], nodelist[idx - img.shape[1] + 1], w, w)
    return g


st = time.time()
img = cv2.imread('./dog.jpg')
img = cv2.resize(img, (500, 400))
# 交互式，返回 (x_min, y_min, w, h)
r = cv2.selectROI('input', img, True)
print(r)
print('-' * 20)
# roi区域
roi = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# 原图mask，与原图等大小
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))  # 包括前景的矩形，格式为(x,y,w,h)

# rect=(248, 85, 300, 320)
# rect = (177, 135, 221, 309)
gamma = 2
lambda_ = 9 * gamma
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_PR_FGD
beta = calcBeta(img)
left_w, upleft_w, up_w, upright_w = calc_n_weights(img, beta, gamma)

for iter_ in range(5):
    img = img.astype(np.float32)  # 防止溢出
    bg_pixels = img[np.logical_or(mask == cv2.GC_BGD, mask == cv2.GC_PR_BGD)].astype(np.float32)
    #     bg_pixels = img[mask == 2].astype(np.float32)
    pfg_pixels = img[mask == cv2.GC_PR_FGD].astype(np.float32)
    if iter_ == 0:
        bg_gmm = GMM()
        bg_gmm.fit(bg_pixels)
        fg_gmm = GMM()
        fg_gmm.fit(pfg_pixels)
    else:
        bg_gmm.clear()
        fg_gmm.clear()




        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (mask[i][j] == cv2.GC_PR_BGD or mask[i][j] == cv2.GC_BGD):
                    bg_gmm.add_points_to_cluster(img[i][j])
                else:
                    fg_gmm.add_points_to_cluster(img[i][j])


        bg_gmm.update_cluster_models()
        fg_gmm.update_cluster_models()


    graph = constructGCGraph(img, mask, bg_gmm, fg_gmm, lambda_, left_w, upleft_w, up_w, upright_w)
    flow = graph.maxflow()
    print("flow", flow)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == cv2.GC_PR_BGD or mask[i, j] == cv2.GC_PR_FGD:
                idx = i * mask.shape[1] + j
                if graph.get_segment(idx) == 0:
                    mask[i, j] = cv2.GC_PR_FGD
                else:
                    mask[i, j] = cv2.GC_PR_BGD
    print('unique', np.unique(mask))
    print('count', np.count_nonzero(mask == 3))
    print('count', np.count_nonzero(mask == 2))

    # 可视化掩码
    # plt.imshow(mask, cmap='gray')
    # plt.title(f"Iteration {iter_+1}")
    # plt.show()

    mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    img = img.astype(np.uint8)
    result = cv2.bitwise_and(img, img, mask=mask2)
    # cv2.imshow('mask', mask2)
    # cv2.imshow('roi', roi)
    cv2.imshow(F"result {iter_ + 1}", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

end = time.time()
print("time", end - st)

