import numpy as np
import math


def reward(cent, sigma, X_train, y_train):
    row = X_train.shape[0]
    column = cent.shape[0]
    G = np.empty((row, column), dtype=float)
    for i in range(row):
        for j in range(column):
            dist = np.linalg.norm(X_train[i] - cent[j])
            G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))

    GTG = np.dot(G.T, G)
    GTG_inv = np.linalg.pinv(GTG)
    fac = np.dot(GTG_inv, G.T)
    W = np.dot(fac, y_train)

    y_pred = np.dot(G, W)

    y_diff = y_pred - y_train

    error = np.asscalar(np.dot(y_diff.T, y_diff))

    return -error, y_pred
