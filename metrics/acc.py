from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
# import collections
import torch
import torch.nn as nn


def cluster_acc_old(y_true, y_pred):
    """
    Compute clustering accuracy via the Kuhn-Munkres algorithm, also called the Hungarian matching algorithm.
    This algorithm provides a 1 to 1 matching between VaDE clusters and ground truth classes.
    Therefore, it is only valid when n_clusters is equal to the number of ground truth classes.

    y_pred and y_true contain integers, each indicating the cluster number a sample belongs to.
    y_pred therefore induces the predicted partition of all samples.
    However, the integers in y_pred are arbitrary and do not have to match the integers chosen for the true partition.
    We align the integers of y_true and y_pred through the Kuhn-Munkres algorithm and subsequently compute
    accuracy as usual.

    Code as modified from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch.

    Args:
        y_true: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.
        y_pred: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.

    Returns:
        A scalar indicating the clustering accuracy.
    """
    assert y_pred.size == y_true.size  # Arguments y_true and y_pred must be of equal shape.
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # perform the Kuhn-Munkres algorithm to obtain the pairs
    ind = linear_assignment(w.max() - w)  # ind is a list of 2 numpy arrays where ind[0][k] and ind[1][k] form a pair for k=0,...,n_clusters-1
    # add the corresponding
    acc_and_w = sum([w[i, j] for (i, j) in zip(ind[0].tolist(), ind[1].tolist())]) * 1.0 / y_pred.size, w
    return acc_and_w


def cluster_acc_and_conf_mat(y_true, y_pred, conf_mat_option="absolute"):
    """
    Compute clustering accuracy.
    In this version, each cluster is assigned to the class with the largest number of observations in the cluster.
    Different from the cluster_acc_old function, this function allows multiple clusters to the same class.
    Therefore, n_cluster can be larger than the number of ground truth classes.

    As a by-product, the square confusion matrix is also computed.

    Args:
        y_true: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.
        y_pred: 1-D numpy array containing integers between 0 and N-1, where N indicates the number of ground truth classes.

    Code as modified from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch.

    Returns:
        A scalar indicating the clustering accuracy.
    """
    assert y_pred.size == y_true.size  # Arguments y_true and y_pred must be of equal shape.
    D_pred = y_pred.max() + 1
    D_true = y_true.max() + 1
    w = np.zeros((D_pred, D_true), dtype=np.int64)
    conf_mat = np.zeros((D_true, D_true), dtype=np.int64)
    # w[i, j] is the count of data points that lie in both VaDE cluster i and true class j
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_pred = np.arange(D_pred)
    ind_true = np.zeros(D_pred, dtype=np.int64)
    # for each VaDE cluster, find the class with the largest number of observations in the cluster and record in ind_true
    for i in range(D_pred):
        ind_max = np.argmax(w[i, :])
        ind_true[i] = ind_max
        # add the count into the corresponding row of the confusion matrix
        conf_mat[ind_max, :] += w[i, :]
    ind = (ind_pred, ind_true)
    acc = sum([w[i, j] for (i, j) in zip(ind[0].tolist(), ind[1].tolist())]) * 1.0 / y_pred.size
    return acc, conf_mat, w
