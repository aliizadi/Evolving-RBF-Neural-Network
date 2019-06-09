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

    y_pred_softmax = softmax(y_pred)

    error = -np.sum(np.sum(np.multiply(y_train, np.log(y_pred_softmax)), axis=1))

    y_class = y_pred_softmax.argmax(1)
    return -error, y_class, W


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def predict(cent, sigma, W, X_test):
    row = X_test.shape[0]
    column = cent.shape[0]
    G = np.empty((row, column), dtype=float)
    for i in range(row):
        for j in range(column):
            dist = np.linalg.norm(X_test[i] - cent[j])
            G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))

    y_pred = np.dot(G, W)

    y_pred_softmax = softmax(y_pred)

    y_class = y_pred_softmax.argmax(1)

    return y_class
