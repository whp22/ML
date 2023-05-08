#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML 
@File ：Linear Regression.py
@Author ：Haopeng Wang
@Date ：2023-04-16 5:26 p.m. 
'''
import pandas as pd
import numpy as np


def load_data(file):
    df = pd.read_csv(file)
    data = np.array(df, dtype=float)
    normalization(data)
    # print(data)
    return data[:,:2], data[:,-1]

def normalization(data):
    for i in range(0, data.shape[1]):
        data[:,i] = (data[:,i]-np.mean(data[:,i]))/np.std(data[:,i])


# h shape 46x1
def h(x,theta):

    h = np.matmul(x,theta)
    return h


def gradient_descent(x,y,theta, lr, epoch):
    m = x.shape[0]
    J_all = []
    for _ in range(epoch):
        h_x = h(x, theta)
        grad_ = (1/m)*(x.T@(h_x-y))
        theta = theta - (lr*grad_)
        J = cost_function(x,y,theta)
        J_all.append(J)
        print("loss:", J[0][0])
    return theta, J_all

def cost_function(x,y,theta):

    m1 = (h(x,theta)-y).T
    m2 = h(x,theta)-y
    M = m1 @ m2

    return M/(2*(y.shape[0]))

x,y = load_data('house_price_data.txt')

#y shape 46x1
y = np.reshape(y,(46,1))
#x shape 46x3
x = np.hstack((np.ones((x.shape[0],1)), x))
# theta shape 3x1
theta = np.zeros((x.shape[1],1))
lr=0.01
epoch=50


theta, J_all = gradient_descent(x,y,theta,lr,epoch)
# J = cost_function(x,y,theta)


