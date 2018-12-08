#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def flatten_ndarray(arr):
    res = []
    for element in arr:
        res.append(element.flatten())

    return np.array(res)

def load_mats():
    # downloading from google drive via python seems to be very complex process and is not the actual 
    # aim of this document, hence I will just assume, that data is already in current dir
    # download link: https://drive.google.com/file/d/157SPnufv1VkxazY3H58HHqYJYpZ76Ghw/view?usp=sharing
    assert os.path.exists('./ECoG_X_test.mat'), 'Current directory should contain train and test mats'

    X_test = loadmat('./ECoG_X_test.mat')
    X_train = loadmat('./ECoG_X_train.mat')
    y_train = loadmat('./ECoG_Y_train.mat')
    y_test = loadmat('./ECoG_Y_test.mat')
    return X_train['X_train'], X_test['X_hold_out'], y_train['Y_train'], y_test['Y_hold_out']

def get_data():
    X_train, X_test, y_train, y_test = load_mats()
    return flatten_ndarray(X_train), flatten_ndarray(X_test), y_train, y_test

def plot_res(y_train, y_pred, outfile):
    plt.rc('font', size=30)
    axes = ['$x$', '$y$', '$z$']
    plt.figure(figsize=(30, 30))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.ylabel(axes[i], rotation=0, labelpad=20)
        plt.xlabel('$t$', labelpad=20)
        plt.plot(y_test[0:1000,i], label='наблюдаемое движение', linewidth=3.0)
        plt.plot(y_pred[0:1000,i], label='предсказание', linewidth=3.0)
        plt.grid(True)
        plt.legend(loc='upper right')

    plt.savefig(outfile)

def pls_train_and_predict(n_comps, X_train, y_train, X_test):
    pls = PLSRegression(n_components=n_comps)
    pls.fit(X_train, y_train)
    return pls.predict(X_test)

def test_pls(n_comps, X_train, y_train, X_test, y_test):
    y_pred = pls_train_and_predict(n_comps, X_train, y_train, X_test)
    print('n_comps = {}; mae = {}, mse = {}, r2 = {}'.format(
            n_comps, 
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            r2_score(y_test, y_pred)))
    return y_pred

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    y_pred = test_pls(2, X_train, y_train, X_test, y_test)
    plot_res(y_test, y_pred, 'pls.pdf')

    for n_comps in range(3, 101):
        test_pls(n_comps, X_train, y_train, X_test, y_test, n_comps == 2)

