import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from itertools import permutations
from sklearn.neighbors.nearest_centroid import NearestCentroid

def get_co_membership_matrix(labels):
    """ Construct co-membership matrix
        Args:
            labels (list): list length n containing labels of clusters for each datapoint

        Returns:
            co_membership_matrix (nd.array): nxn matrix with pairwise information of comembership in clusters of all data points
    """
    n = len(labels)
    co_membership_matrix = np.zeros((n, n))

    for i, label in enumerate(np.unique(labels)):  # for each cluster
        cluster_i = np.where(labels == label)[0]  # get bursts of cluster_i
        combinations = combinations_with_replacement(cluster_i,2)  # indicate bursts i,j falling into the same cluster by assigning 1 to entry [i,j] and [j,i] in co-membership-matrix
        for ij in combinations:
            co_membership_matrix[ij[0], ij[1]] = 1
            co_membership_matrix[ij[1], ij[0]] = 1

    return co_membership_matrix


def calculate_centroids_and_predict_validation_data(train_data,train_labels, valid_data):
    """ Calculate centroids of each cluster found in training data and assign labels to validation data based on nearest centroid
    Args:
        train_data (nd.array): Matrix containing training data
        train_labels (list): List of cluster labels for each data point in trainig set
        valid_data (nd.array): Matrix containing validation data

    Returns:
        centroids (list of nd.arrays): list containing centroids for each cluster (index corresponds to cluster)
        labels_predicted (list): list containing the cluster labels assigned to validation data based on centroids
    """
    clf = NearestCentroid(metric = "euclidean")
    clf.fit(train_data, train_labels) # calculate centroid for each cluster in training data
    labels_predicted = clf.predict(valid_data) # label validation data based on centroids
    centroids = clf.centroids_

    return centroids, labels_predicted


def calculate_prediction_strength(labels_fitted, labels_centroids_based, unique_labels):
    """ Calculate prediction strength for each cluster
    Args:
        labels_fitted (list): labels found by separately clustering of validation data
        labels_centroid (list): labels found by assignend labels to validation data based on cluster centroids found by clustering training data
        unique_labels (list): list of unique labels

    Returns:
        cluster_prediction_strengths (list): list of prediction strenghts for each cluster
    """
    A_valid_indices = [np.where(labels_fitted == i)[0] for i in unique_labels]  # get indices for each clusters in validation data
    cluster_sizes = [len(A) for A in A_valid_indices]  # get size of each cluster
    co_membership_matrix_centroids = get_co_membership_matrix(labels_centroids_based)  # compute co-membership for validation data labeld by nearest centroid
    cluster_prediction_strengths = []

    for i,c in enumerate(unique_labels):
        A_i = A_valid_indices[i]  # cluster_i indices
        co_membership_sum = 0

        for x1x2 in permutations(A_i, 2):  # get each pair of different bursts in cluster
            co_membership_sum += co_membership_matrix_centroids[x1x2[0], x1x2[1]] #check whether data from fitted validation set fall in same clusters based on training centroids

        if cluster_sizes[i] <1:  # no or only one burst in cluster
            prediction_sterngth_c = 0
        else:
            prediction_sterngth_c = 1 / (cluster_sizes[i] * (cluster_sizes[i] - 1)) * co_membership_sum  # prediction strength for cluster_i weighted by cluster size

        cluster_prediction_strengths.append(prediction_sterngth_c)
    return cluster_prediction_strengths, cluster_sizes


def calculate_prediction_strength_per_sample(labels_fitted, labels_centroids_based, unique_labels):
    """ Calculate prediction strength for each cluster
    Args:
        labels_fitted (list): labels found by separately clustering of validation data
        labels_centroid (list): labels found by assignend labels to validation data based on cluster centroids found by clustering training data
        unique_labels (list): list of unique labels

    Returns:
        cluster_prediction_strengths (list of list): list of prediction strenghts for each cluster per sample
    """

    A_valid_indices = [np.where(labels_fitted == i)[0] for i in unique_labels]  # get indices for each clusters in validation data
    cluster_sizes = [len(A) for A in A_valid_indices]  # get size of each cluster
    co_membership_matrix_centroids = get_co_membership_matrix(labels_centroids_based)  # compute co-membership for validation data labeld by nearest centroid

    cluster_prediction_strengths_per_sample = []
    for i,c in enumerate(unique_labels):
        A_c = A_valid_indices[i]  # cluster_i indices
        prediction_strength_c = []
        for index, x1 in enumerate(A_c):
            co_membership_sum = 0
            A_i = np.delete(A_c, index)
            for x2 in A_i:
                co_membership_sum += co_membership_matrix_centroids[x1, x2]

            prediction_strength_i = 1 / len(A_i) * co_membership_sum
            prediction_strength_c.append(prediction_strength_i)
        cluster_prediction_strengths_per_sample.append(prediction_strength_c)
    return cluster_prediction_strengths_per_sample, cluster_sizes


def get_prediction_strength_per_k(data, train_indices, valid_indices, train_labels, valid_labels, per_sample = False):
    """ Calculate prediction strength for each cluster obtained by clustering with k clusters
    Args:
        data (nd.array): Array containing data (n x m)
        train_indices (nd.arrays): training set indices
        valid_indices (nd.arrays): valid set indices
        train_labels (dict): dictionary of lists containing the cluster labels for each point in training set after clustering into k clusters
                             key = k (number of clusters in clustering)
                             value = label for each point
        valid_labels (dict): dictionary of lists containing the cluster labels for each point in validation set after clustering into k clusters
                             key = k (number of clusters in clustering)
                             value = label for each point
        per_sample (bool): boolean whether prediction strength is calculated per sample or per cluster

    Returns:
        prediction_strengths_per_k (dict): dictionary containing the prediction strength for each cluster/sample in c after clustering into k clusters
    """

    k_clusters = list(train_labels.keys())
    prediction_strengths_per_k = {} # key: k (number of clusters in clustering) value: prediction strength for each cluster i/ each point in cluster i
    cluster_sizes_per_k = {}

    if per_sample:
        print("Calculate Predictions Strength per Sample for each Clustering!")
    else:
        print("Calculate Predictions Strength per Cluster for each Clustering!")

    training_set = data[train_indices]
    validation_set = data[valid_indices]

    for k in k_clusters:
        train_labels_k = train_labels[k]
        valid_labels_k = valid_labels[k]

        if k > 1:
            centroids_k, labels_centroids_based = calculate_centroids_and_predict_validation_data(training_set,train_labels_k,validation_set)
        else:
            labels_centroids_based = np.zeros(len(validation_set))

        if per_sample:
            cluster_prediction_strengths, cluster_sizes = calculate_prediction_strength_per_sample(valid_labels_k,labels_centroids_based, np.sort(np.unique(valid_labels_k)))  # list of predictions strengths for each cluster i after clustering with k clusters
        else:
            cluster_prediction_strengths, cluster_sizes = calculate_prediction_strength(valid_labels_k, labels_centroids_based,np.sort(np.unique(valid_labels_k)))  # list of lists of predictions strengths for each sample in cluster i after clustering with k clusters

        prediction_strengths_per_k[k] = cluster_prediction_strengths
        cluster_sizes_per_k[k] = cluster_sizes

    return prediction_strengths_per_k, cluster_sizes_per_k


def get_clusters_below_threshold(prediction_strengths_per_k, cluster_sizes_per_k,threshold):
    """ Calculate clusters with prediction strength below threshold and get proportion of datapoints from validation set falling into these clusters per clustering with k clusters
    Args:
        prediction_strengths_per_k (dict): Dictonary containig for each clustering with n_clusters the prediction strength for each cluster in each fold
                                          keys = n_clusters  values = folds x n_clusters
        prediction_strengths_per_k (dict): dictonary containing the prediction strength for each cluster/sample in c after clustering into k clusters

        threshold (int): Prediction strength threshold indicating low prediction strength

    Returns:

        """
    k_clusters = list(prediction_strengths_per_k.keys())
    low_clusters_per_k = {}
    low_cluster_sizes_per_k = {}
    low_cluster_sizes_percent_per_k = {}

    for k in k_clusters:
        prediction_strengths = prediction_strengths_per_k[k]
        low_clusters_idx = np.where(prediction_strengths < threshold)
        low_cluster_sizes = cluster_sizes_per_k[k][low_clusters_idx]
        low_cluster_sizes_percent = ((np.asarray(cluster_sizes_per_k[k])/ np.sum(cluster_sizes_per_k[k])) * 100)[low_clusters_idx]

        low_clusters_per_k[k] = low_clusters_idx[0]
        low_cluster_sizes_per_k[k] = low_cluster_sizes
        low_cluster_sizes_percent_per_k[k] = low_cluster_sizes_percent

    return low_clusters_per_k, low_cluster_sizes_per_k, low_cluster_sizes_percent_per_k

def get_points_in_clusters_below_and_above_threshold(predictions_strengths_per_sample_per_k, valid_indices, valid_labels, threshold):
    """ extract burst indices for burst with low individual prediction strength for clustering with k clusters per cluster
    Args:
            predictions_strengths_per_sample_per_k (dict): dictionary containing the prediction strength for each sample in cluster_i after clustering into k clusters
            valid_indices (nd.arrays): valid set indices
            valid_labels (dict): dictionary of lists containing the cluster labels for each point in validation set after clustering into k clusters
                                key = k (number of clusters in clustering)
                                value = label for each point
            threshold (float): threshold for defining low individual prediction strength

        Returns:
            low_predictive_points_in_clusters_per_k (dict): dictionary containing the indices of datapoints with individual prediction strength below threshold after clustering with k clusters
                                key = k (number of clusters in clustering)
                                value = list of lists with indicies of low prediction strength datapoints for each cluster
            low_predictive_points_in_clusters_per_k_sizes (dict): dictionary containing the number of datapoints in cluster_i with individual prediction strength below threshold after clustering with k clusters
                                key = k (number of clusters in clustering)
                                value = list of number of low prediction strength datapoints in each cluster
            low_predictive_points_in_clusters_per_k_percent (dict): dictionary containing the percent of datapoints in cluster_i with individual prediction strength below threshold after clustering with k clusters
                                key = k (number of clusters in clustering)
                                value = list of percent of low prediction strength datapoints in each cluster
            low_predictive_points_labels_per_k (dict): dictionary containing the labels corresponding to the indices of datapoints with prediction strength below threshold after clustering with k clusters
            high_predictive_points_in_clusters_per_k (dict): dictionary containing the indices of datapoints with individual prediction strength below threshold after clustering with k clusters
                                key = k (number of clusters in clustering)
                                value = list of lists with indicies of low prediction strength datapoints for each cluster
            high_predictive_points_labels_per_k (dict): dictionary containing the labels corresponding to the indices of datapoints with prediction strength above threshold after clustering with k clusters
        """

    k_clusters = list(predictions_strengths_per_sample_per_k.keys())
    low_predictive_points_in_clusters_per_k = {}
    low_predictive_points_in_clusters_per_k_sizes = {}
    low_predictive_points_in_clusters_per_k_percent = {}

    high_predictive_points_in_clusters_per_k = {}

    low_predictive_points_labels_per_k = {}
    high_predictive_points_labels_per_k = {}

    for k in k_clusters:
        valid_labels_k = valid_labels[k]
        low_individual_in_clusters_k = []
        low_individual_in_clusters_k_sizes = []
        low_individual_in_clusters_k_percent = []

        high_individual_in_clusters_k = []

        low_individual_labels_k = np.zeros(len(valid_labels_k)) - 1
        high_individual_labels_k = np.zeros(len(valid_labels_k)) - 1

        for i in range(k):
            cluster_i_indices = np.where(valid_labels_k == i)[0]
            index_in_cluster = np.where(np.asarray(predictions_strengths_per_sample_per_k[k][i]) < threshold)  # get position relative to cluster
            low_predictive_points = valid_indices[cluster_i_indices[index_in_cluster]]
            high_predictive_points = valid_indices[np.delete(cluster_i_indices, index_in_cluster)]

            low_individual_in_clusters_k += list(low_predictive_points)  # get indices (relative to overall data) of low individual data points in cluster_i
            low_individual_in_clusters_k_sizes.append(len(low_predictive_points)) # get number of datapoints below threshold in cluster_i
            low_individual_in_clusters_k_percent.append((len(low_predictive_points) / len(valid_indices[np.where(valid_labels_k == i)[0]])) * 100) # get percent of cluster_i

            high_individual_in_clusters_k += list(high_predictive_points)

            low_individual_labels_k[low_predictive_points] = i
            high_individual_labels_k[high_predictive_points] = i



        low_predictive_points_in_clusters_per_k[k] = np.sort(low_individual_in_clusters_k)
        low_predictive_points_in_clusters_per_k_sizes[k] = low_individual_in_clusters_k_sizes
        low_predictive_points_in_clusters_per_k_percent[k] = low_individual_in_clusters_k_percent

        high_predictive_points_in_clusters_per_k[k] = np.sort(high_individual_in_clusters_k)

        low_predictive_points_labels_per_k[k] = list(filter(lambda x: x != -1, low_individual_labels_k))
        high_predictive_points_labels_per_k[k] = list(filter(lambda x: x != -1, high_individual_labels_k))

    return low_predictive_points_in_clusters_per_k, low_predictive_points_in_clusters_per_k_sizes, low_predictive_points_in_clusters_per_k_percent, low_predictive_points_labels_per_k, high_predictive_points_in_clusters_per_k, high_predictive_points_labels_per_k






from asymmetric_laplacian_distribution import get_index_per_class


data_dir = "data/"

data = np.load(data_dir + "clearly_separated_data_F_signal_noise.npy")

amplitude_conditions =["S", "M", "L"] # ["S", "S/M", "M", "M/L", "L"]
time_constant_conditions = ["equal_sharp", "equal_wide", "wide_sharp_negative_skew", "sharp_wide_positive_skew"]#["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
ambiguous_conditions = []#["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]

samples_per_condition = 1000
samples_per_ambiguous_condition = 400

cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)


save_dir = "Toy_data/Clearly_Separated/Prediction_Strength/"

import training_set_split
train_fold_indices, train_fold_indices = training_set_split.get_training_folds(data, cluster_dict)



training_set = data[train_fold_indices[0]]
validation_set = data[train_fold_indices[1]]


training_set_labels = np.load(save_dir + "Labels/labels_k=10_reg=None_training.npy")
validation_set_labels = np.load(save_dir + "Labels/labels_k=10_reg=None_validation.npy")

train_labels = {}
valid_labels = {}
for i,labels in enumerate(training_set_labels):
    train_labels[i+1] = labels
    valid_labels[i+1] = validation_set_labels[i]


prediction_strengths, cluster_sizes = get_prediction_strength_per_k(data, train_fold_indices[0], train_fold_indices[1], train_labels, valid_labels, per_sample = False)

import functions_for_plotting

true_labels = np.repeat(np.arange(0,12), 500)

functions_for_plotting.plot_clusters(training_set, true_labels,train_labels[12], 12, 3,4, figsize=(20,20),n_bursts = 100,y_lim = (0,14))

functions_for_plotting.plot_mean_prediction_strengths(prediction_strengths, cluster_sizes,threshold=None, figsize = (40, 5), size_weighted=False,title = "")
