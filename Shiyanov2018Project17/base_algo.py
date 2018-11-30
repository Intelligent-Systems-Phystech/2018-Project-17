#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

def flatten_ndarray(arr):
    res = []
    for element in arr:
        res.append(element.flatten())

    return np.array(res)

def get_data():
    # downloading from google drive via python seems to be very complex process and is not the actual 
    # aim of this document, hence I will just assume, that data is already in current dir
    # download link: https://drive.google.com/file/d/157SPnufv1VkxazY3H58HHqYJYpZ76Ghw/view?usp=sharing
    assert os.path.exists('./ECOG_X_test.mat'), 'Current directory should contain train and test mats'

    X_test = loadmat('./ECOG_X_test.mat')
    X_train = loadmat('./ECoG_X_train.mat')
    y_train = loadmat('./ECoG_Y_train.mat')
    y_test = loadmat('./ECoG_Y_test.mat')

    X_train = X_train['X_train']
    X_train = flatten_ndarray(X_train)

    X_test = X_test['X_hold_out']
    X_test = flatten_ndarray(X_test)

    y_train = y_train['Y_train']
    y_test = y_test['Y_hold_out']

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    pls = PLSRegression(n_components=2)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)

    plt.rc('font', family = 'serif', size=16)

    plt.figure(figsize=(30, 15))
    plt.suptitle("mean squared error is %f" % mean_squared_error(y_pred, y_test), fontsize=20)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(y_test[:,i], label='actual result')
        plt.plot(y_pred[:,i], label='prediction')
        plt.grid(True)
        plt.legend(loc='upper right')

    plt.savefig('pls.png')
