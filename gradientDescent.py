import pandas as pd
import numpy as np
import statsmodels.api as sm
from featureProcess import *


def batch_gradient_descent(X,y, w_initial, h=0.01, maxSteps=100000):
    """
    批量梯度下降法求逻辑回归参数
    :param X: 输入特征矩阵，第一列为向量1
    :param y: 标签，用1、0表示
    :param w_initial: 初始化权重
    :param h: 固定步长
    :param maxSteps: 最大迭代步数
    :return: 权重的估计值
    """

    w0 = w_initial
    for i in range(maxSteps):
        s = np.exp(X*w0)/(1+np.exp(X*w0))
        # 梯度
        descent = X.T*y-X.T*s
        # 梯度上升
        w1 = w0 + descent*h
        w0 = w1
        # 若当前的权重的更新很小时，认为迭代已经收敛，可以提前退出迭代
        if max(abs(descent*h)) < 0.00001:
            break
    return w1

