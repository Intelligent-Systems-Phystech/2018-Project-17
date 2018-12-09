#!/usr/bin/env python3

from base_algo import load_mats, plot_res, test_pls, flatten_ndarray
import numpy as np

centers = np.array([(3, 1), (5, 1), (3, 3), (5, 3), (3, 5),
                    (5, 5), (7, 5), (6, 6), (3, 7), (5, 7), 
                    (7, 7), (2, 8), (4, 8), (6, 8), (1, 9), 
                    (3, 9), (5, 9), (7, 9), (2, 10), (4, 10),
                    (6, 10), (1, 11), (3, 11), (5, 11), (2, 12),
                    (4, 12), (6, 12), (1, 13), (3, 13), (5, 13),
                    (1, 15), (3, 15)])

def get_dist_params_for_every_freq(x, get_dist_params):
    dist_params_for_every_freq = []
    for freq in x:
        dist_params_for_every_freq.append(get_dist_params(freq))

    return np.array(dist_params_for_every_freq)

def local_model(X, get_dist_params):
    res = []
    for x in X:
        dist_params_for_every_freq = get_dist_params_for_every_freq(x, get_dist_params)
        res.append(dist_params_for_every_freq.flatten())

    return res

def get_mean(freq):
    intens_sum = 0
    x, y = 0, 0
    for (center_x, center_y), intens in zip(centers, freq):
        intens = abs(intens)
        intens_sum += intens
        x += center_x * intens
        y += center_y * intens

    x /= intens_sum
    y /= intens_sum
    return [x, y]

def get_disp(freq, mean):
    mean_x, mean_y = mean
    intens_sum = 0
    disp_x, disp_y = 0, 0
    for (center_x, center_y), intens in zip(centers, freq):
        intens = abs(intens)
        intens_sum += intens
        disp_x += (center_x - mean_x)**2 * intens
        disp_y += (center_y - mean_y)**2 * intens

    disp_x /= intens_sum
    disp_y /= intens_sum
    return [disp_x, disp_y]

def get_norm_dist_params(freq):
    dist_params = []
    dist_params.extend(get_mean(freq))
    dist_params.extend(get_disp(freq, dist_params))
    return np.array(dist_params)


if __name__ == '__main__':
    norm_dist_local_model = lambda X: local_model(X, get_norm_dist_params)

    X_train, X_test, y_train, y_test = load_mats()
    X_train = norm_dist_local_model(X_train)
    X_test = norm_dist_local_model(X_test)

    y_pred = test_pls(2, X_train, y_train, X_test, y_test)
    plot_res(y_test, y_pred, 'main_algo.pdf')

    for n_comps in range(3, 101):
        test_pls(n_comps, X_train, y_train, X_test, y_test)

