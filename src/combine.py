# -*- coding = utf-8 -*-
# @Time : 2023/4/12 18:14
# @Author : CQU20205644
# @File : combine.py
# @Software : PyCharm
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import time
from sklearn.cluster import KMeans
import cv2
import networkx as nx
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components=5, max_iter=10, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.fit_called = False

    def fit(self, X):
        # 进行聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.tol)
        compactness, labels, centers = cv2.kmeans(X, self.n_components, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # 将聚类好的点分别放到五个类，并分别计算他们的高斯分布模型和权重
        self.weights_ = []
        self.means_ = []
        self.covariances_ = []
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

        self.fit_called = True

    def predict_proba(self, X):
        if not self.fit_called:
            raise ValueError("GMM has not been fit yet. Call the fit method first.")

        # 计算每个样本属于每个类的概率
        probs = np.zeros(self.n_components)
        total = 0
        for i in range(self.n_components):
            prob_i = multivariate_normal(mean=self.means_[i], cov=self.covariances_[i]).pdf(X)
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
    beta = np.sum(diff_left**2) + np.sum(diff_up**2) + np.sum(diff_upleft**2) + np.sum(diff_upright**2)
    if beta <= np.finfo(float).eps:
        beta = 0
    else:
        beta = 1.0 / (2 * beta / (4*cols*rows - 3*cols - 3*rows + 2))
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
    left_w[:, 1:] = gamma * np.exp(-beta * np.sum(diff_left**2, axis=2))
    upleft_w[1:, 1:] = gamma_div_sqrt2 * np.exp(-beta * np.sum(diff_upleft**2, axis=2))
    up_w[1:, :] = gamma * np.exp(-beta * np.sum(diff_up**2, axis=2))
    upright_w[1:, :-1] = gamma_div_sqrt2 * np.exp(-beta * np.sum(diff_upright**2, axis=2))

    return left_w, upleft_w, up_w, upright_w


def constructGCGraph(img, mask, bgdGMM, fgdGMM, lamda, leftW, upleftW, upW, uprightW):
    vtxCount = img.shape[0] * img.shape[1]
    edgeCount = 2 * (4 * vtxCount - 3 * (img.shape[0] + img.shape[1]) + 2)
    graph = nx.DiGraph()
    #     graph = nx.DiGraph(sparse=True)

    graph.add_nodes_from(range(vtxCount))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = i * img.shape[1] + j
            color = img[i, j, :]
            fromSource, toSink = 0, 0

            if mask[i, j] == cv2.GC_PR_BGD or mask[i, j] == cv2.GC_PR_FGD:
                fromSource = -np.log(bgdGMM.predict_proba(color))
                toSink = -np.log(fgdGMM.predict_proba(color))

            elif mask[i, j] == cv2.GC_BGD:
                fromSource = lamda
                toSink = 0
            else:
                fromSource = 0
                toSink = lamda
            graph.add_edge('s', idx, capacity=fromSource)
            graph.add_edge(idx, 't', capacity=toSink)
            if j > 0:
                w = leftW[i, j]
                graph.add_edge(idx, idx - 1, capacity=w)
            if j > 0 and i > 0:
                w = upleftW[i, j]
                graph.add_edge(idx, idx - img.shape[1] - 1, capacity=w)
            if i > 0:
                w = upW[i, j]
                graph.add_edge(idx, idx - img.shape[1], capacity=w)
            if j < img.shape[1] - 1 and i > 0:
                w = uprightW[i, j]
                graph.add_edge(idx, idx - img.shape[1] + 1, capacity=w)

    return graph