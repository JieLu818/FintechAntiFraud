# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from featureProcess import *
from data import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from confusionMetrics import *
from sklearn.model_selection import GridSearchCV


def Kernel(x1, x2, type='linear'):
    """
    :param x1:
    :param x2: 样本，行向量表示
    :param type: 核函数。这里默认使用线性核作为演示
    :return: 核函数下的内积
    """
    if type == 'linear':
        ker = x1 * x2.T
    # 可以添加更多类型的核函数
    return ker[0, 0]


def SVM_Function(alpha, b, X, y, x):
    """
    :param alpha: 列向量，拉格朗日乘子
    :param b: 偏置项
    :param X: 样本矩阵，行表示样本，列表示特征
    :param y: 列向量，用1和-1表示样本类别
    :param x: 需要计算类别的样本，行向量
    :return: 计算结果，符号表示所示样本的类别
    """
    w = np.multiply(X, np.multiply(alpha, y)).sum(axis=0).T
    s = x * w + b
    return s[0, 0]


def calc_lh(y_i, y_j, a_i, a_j, C):
    """
    :param y_i，y_j: 第i和j个样本所属类别
    :param a_i，a_j: 第i和j个样本对应的拉格朗日乘子
    :param C: 分类错误惩罚项
    :return: 当前第i和j个拉格朗日乘子的上限和下限
    """
    if y_i != y_j:
        L = max(0, a_j - a_i)
        H = min(C, C + a_j - a_i)
    else:
        L = max(0, a_j + a_i - C)
        H = min(C, a_j + a_i)
    return [L, H]


def SMO(X, y, C=10, kernel_type='linear', max_passes=100, tol=0.0001):
    """
    :param X: 样本矩阵，行表示样本，列表示特征
    :param y: 列向量，用1和-1表示样本类别
    :param C: 分类错误惩罚项
    :param kernel_type: 核函数。这里默认使用线性核作为演示
    :param max_passes: 当目前发现没有样本可以进行优化时，仍需迭代的最大迭代步数，以保证结果的准确性
    :param tol: 检查KKT条件时的容忍误差
    :return: 样本的拉格朗日乘子，SVM的权重w和偏置项intercept
    """
    # 初始化参数，包括拉格朗日乘子alpha和偏置项b
    (m, p) = X.shape
    alpha, b = np.array([0.0] * m).reshape((m, 1)), 0.0
    # 遍历次数累加器。当目前没有样本需要改进时，passes加1.否则passes设置为0
    passes = 0
    while passes <= max_passes:
        num_changed_alphas = 0  # 检查每一次外循环后，有多少对alpha加以优化
        for i in range(m):
            # 内循环进行之前，计算每个样本在当前SVM参数下的预测误差
            E = [SVM_Function(alpha, b, X, y, X[ii,]) - y[ii, 0] for ii in range(m)]
            E_i = E[i]
            # 对于每一个边界上的样本，检查是否不满足KKT条件。对于不满足KKT条件的边界样本进行优化
            if (y[i, 0] * E_i < -tol and alpha[i, 0] < C) or (y[i, 0] * E_i > tol and alpha[i, 0] > 0):
                # 为了提升计算效率，在内循环里，优先选择计算误差与外循环样本的计算误差的差值最大的样本
                diff_E_inner = np.abs(E - E_i)
                while 1:
                    j = diff_E_inner.argmax()
                    # 如果没有需要优化的内循环样本，则直接进入下一步的外循环中
                    if max(diff_E_inner) < 0:
                        break
                    # 如果当前的内循环样本不满足条件，则进入下一步的内循环中
                    if i == j:
                        diff_E_inner[j] = -1
                        continue
                    E_j = E[j]
                    a_i_old, a_j_old = alpha[i, 0], alpha[j, 0]
                    # 计算alpha_j的上下限
                    [L, H] = calc_lh(y[i, 0], y[j, 0], a_i_old, a_j_old, C)
                    if L == H:
                        diff_E_inner[j] = -1
                        continue
                    eta = 2 * Kernel(X[i,], X[j,], type=kernel_type) - Kernel(X[i,], X[i,], type=kernel_type) - Kernel(
                        X[j,], X[j,], type=kernel_type)
                    if eta >= 0:
                        diff_E_inner[j] = -1
                        continue
                    # 更新当前的alpha_j,并进行适当的修剪
                    a_j_new = a_j_old - y[j, 0] * (E_i - E_j) / eta
                    if a_j_new > H:
                        a_j_new = H
                    elif a_j_new < L:
                        a_j_new = L
                    if np.abs(a_j_new - a_j_old) < 0.000001:
                        diff_E_inner[j] = -1
                        continue
                    # 更新当前的alpha_
                    a_i_new = a_i_old - y[i, 0] * y[j, 0] * (a_j_new - a_j_old)
                    # 更新偏置
                    b1 = b - E_i - y[i, 0] * (a_i_new - a_i_old) * Kernel(X[i,], X[i,], type=kernel_type) - y[j, 0] * (
                                a_j_new - a_j_old) * Kernel(X[i,], X[j,], type=kernel_type)
                    b2 = b - E_j - y[i, 0] * (a_i_new - a_i_old) * Kernel(X[i,], X[j,], type=kernel_type) - y[j, 0] * (
                                a_j_new - a_j_old) * Kernel(X[j,], X[j,], type=kernel_type)
                    if 0 < a_i_new < C:
                        b = b1
                    elif 0 < a_j_new < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    # 如果有参数对被优化，计数器加一，并更新参数列表
                    num_changed_alphas = num_changed_alphas + 1
                    alpha[i, 0], alpha[j, 0] = a_i_new, a_j_new
                    E[i] = SVM_Function(alpha, b, X, y, X[i,]) - y[i, 0]
                    E[j] = SVM_Function(alpha, b, X, y, X[j,]) - y[j, 0]
                    break
        # 如果当前的外、内循环都没有参数进行更新，则遍历次数累加器加一
        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes == 0
    # 全部更新完后，计算权重
    w = np.multiply(X, np.multiply(alpha, y)).sum(axis=0).T
    return {'alpha': alpha, 'w': w, 'intercept': b}

