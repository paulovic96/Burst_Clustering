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


def convert_to_dict(list, keys):
    dict = {}
    for i, key in enumerate(keys):
        dict[key] = list[i]
    return dict


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





#######################################################################################################################
############################################## Old Functions ##########################################################
#######################################################################################################################


def cross_valdation_prediction_strength(data, train_folds, valid_folds, train_fold_labels, valid_fold_labels, n_clusters=range(20), per_sample=False, valid_fold_labels_predicted_precomputed=None):
    """ Calculate prediction strength for each cluster and each fold in cross validation
    Args:
        data (nd.array): Array containing data (n x m)
        train_folds (list of nd.arrays): list of k training set indices each with dimension n-(n/k) x m
        valid_folds (list of nd.arrays): list of k validation set indices each with dimension n/k x m
        train_fold_labels (list of lists): list of lists containing the cluster labels for each point in training set
        valid_fold_labels (list of lists): list of lists containing the cluster labels for each point in validation set
        valid_fold_labels_predicted_precomputed (list of lists): list of lists containing the predicted cluster labels for each point in validation set based on the training set

    Returns:
        predictions_strengths_cv (list of lists): list of lists containing the prediction strenghts for each cluster in clustering with k clusters for each fold (e.g for k ranging from 1 to 20 n_folds X 20 X 1...20)
                                                   1.dim: n-folds  2.dim:. n-clusters 3.dim: prediction strength for each cluster i
        valid_fold_labels_predicted (list of lists): list of lists containing the cluster labels assigned to validation data based on centroids in clustering with k clusters for each fold
    """

    n_clusters = range(len(train_fold_labels[0]))
    k_folds = len(train_folds)
    valid_fold_labels_predicted = []

    predictions_strengths_cv = []  # 1.dim: k-folds  2.dim:.n-clusters 3.dim: prediction strength for each cluster i

    for k in range(k_folds):  # for each fold
        print("Calculate Predictions Strength for %d. Fold" % (k + 1))

        training_set = data[train_folds[k]]  # training set for fold k splitting
        valid_set = data[valid_folds[k]]  # validation set for fold k splitting

        train_labels = train_fold_labels[k]  # labels for training set
        valid_labels = valid_fold_labels[k]  # labels for validation set

        if valid_fold_labels_predicted_precomputed:
            valid_labels_predicted = valid_fold_labels_predicted_precomputed[k]
        else:
            valid_labels_predicted = []

        predictions_strengths_k = []  # prediction strength for fold k

        for c in n_clusters:  # for each clustering with 1 up to n_clusters
            # labels for clustering with c clsuters
            train_labels_c = train_labels[c]
            valid_labels_c = valid_labels[c]

            if valid_fold_labels_predicted_precomputed:
                labels_predicted_c = valid_labels_predicted[c]

            else:
                if c > 0:
                    centroids_c, labels_predicted_c = calculate_centroids_and_predict_validation_data(training_set,
                                                                                                      train_labels_c,
                                                                                                      valid_set)  # calculate centroids
                else:
                    labels_predicted_c = np.zeros(len(valid_set))

            if per_sample:
                cluster_prediction_strengths = calculate_prediction_strength_per_sample(valid_labels_c,
                                                                                        labels_predicted_c, np.sort(
                        np.unique(train_labels_c)))  # list of predictions strengths for each cluster i in c clusters
            else:
                cluster_prediction_strengths = calculate_prediction_strength(valid_labels_c, labels_predicted_c,
                                                                             np.sort(np.unique(
                                                                                 train_labels_c)))  # list of predictions strengths for each cluster i in c clusters
            predictions_strengths_k.append(cluster_prediction_strengths)

            if not (valid_fold_labels_predicted_precomputed):
                valid_labels_predicted.append(labels_predicted_c)

        predictions_strengths_cv.append(predictions_strengths_k)
        valid_fold_labels_predicted.append(valid_labels_predicted)
    return predictions_strengths_cv, valid_fold_labels_predicted


def calculate_prediction_strength_per_k(predictions_strengths_cv, valid_fold_labels, valid_fold_labels_predicted,
                                        strength_sorted=True):
    """ Calculate prediction strength for each cluster and each fold in cross validation
    Args:
        predictions_strengths_cv (list of lists): list of lists containing the prediction strenghts for each cluster in clustering with k clusters for each fold (e.g for k ranging from 1 to 20 n_folds X 20 X 1...20)
                                                   1.dim: n-folds  2.dim:. n-clusters 3.dim: prediction strength for each cluster i
        valid_fold_labels (list of lists): list of lists containing the cluster labels for each point in validation set
        valid_fold_labels_predicted (list of lists): list of lists containing the cluster labels assigned to validation data based on centroids in clustering with k clusters for each fold
        strength_sorted (bool): If true sort cluster according to prediction strenght in descending order

    Returns:
        k_predictions_strength_cv (dict): Dictonary containig for each clustering with n_clusters the prediction strength for each cluster in each fold
                                          keys = n_clusters  values = folds x n_clusters
        k_valid_fold_labels_predicted (dict): containing for each clustering with n_cluster the validation set labels predicted by training set centroids for each fold
                                              keys = n_clusters  values = k_folds x n labels for each point in validation set

        k_valid_fold_labels (dict): containing for each clustering with n_cluster the validation set labels by separately clustering the validation set for each fold
                                    keys = n_clusters  values = k_folds x n labels for each point in validation set

        valid_cluster_size (dict): containing for each clustering with n_cluster the cluster size of the separately clustered validation set for each fold
                                  keys = n_clusters  values = k_folds x size for each cluster

        valid_cluster_size_predicted (dict): containing for each clustering with n_cluster the cluster size of the centroid assigned validation set clusters for each fold
                                             keys = n_clusters  values = k_folds x size for each cluster
    """

    k_predictions_strength_cv = {}
    k_valid_fold_labels_predicted = {}
    k_valid_fold_labels = {}
    valid_cluster_size = {}
    valid_cluster_size_predicted = {}

    for i in range(1, len(predictions_strengths_cv[0]) + 1):  # for each clustering ranging from 1 to max n_clusters
        k_predictions_strength_cv[i] = []
        k_valid_fold_labels_predicted[i] = []
        k_valid_fold_labels[i] = []
        valid_cluster_size[i] = []
        valid_cluster_size_predicted[i] = []

    for i, fold in enumerate(predictions_strengths_cv):  # for each fold
        for j, k in enumerate(fold):  # for each clustering j with k clusters prediction strenght of fold

            k_valid_fold_labels_predicted[j + 1].append(valid_fold_labels_predicted[i][j])
            k_valid_fold_labels[j + 1].append(valid_fold_labels[i][j])

            valid_cluster_size_c = []
            valid_cluster_size_c_predicted = []
            for c in range(len(k)):  # np.sort(np.unique(valid_fold_labels[i][j])):#range(j+1):
                cluster_size = len(np.where(valid_fold_labels[i][j] == c)[0])
                cluster_size_predicted = len(np.where(valid_fold_labels_predicted[i][j] == c)[0])
                valid_cluster_size_c.append(cluster_size)
                valid_cluster_size_c_predicted.append(cluster_size_predicted)

            strength_and_size = list(zip(np.round(k, 2), valid_cluster_size_c))
            strength_and_size_predicted = list(zip(np.round(k, 2), valid_cluster_size_c_predicted))

            if strength_sorted:
                strength_and_size = sorted(strength_and_size, key=lambda e: (e[0], e[1]), reverse=True)
                strength_and_size_predicted = sorted(strength_and_size_predicted, key=lambda e: (e[0], e[1]),
                                                     reverse=True)

            k_predictions_strength_cv[j + 1].append(np.asarray(strength_and_size)[:, 0])

            valid_cluster_size[j + 1].append(np.asarray(strength_and_size)[:, 1])
            valid_cluster_size_predicted[j + 1].append(np.asarray(strength_and_size_predicted)[:, 1])

    return k_predictions_strength_cv, k_valid_fold_labels_predicted, k_valid_fold_labels, valid_cluster_size, valid_cluster_size_predicted


def calculate_bad_cluster_sizes_cv(k_predictions_strength_cv, valid_cluster_size, valid_cluster_size_predicted,
                                   n_clusters=range(1, 21), folds=5, threshold=0.8):
    """ Calculate proportion of datapoints from validation set falling into clusters with low prediction strength (below threshold = 0.8) per clustering with k clusters based on:
       a) clustres found by spectral clustering
       b) clusters bound by centroid labeling based on training dataset

    Args:
        k_predictions_strength_cv (dict): Dictonary containig for each clustering with n_clusters the prediction strength for each cluster in each fold
                                          keys = n_clusters  values = folds x n_clusters
        valid_cluster_size (dict): containing for each clustering with n_cluster the cluster size of the separately clustered validation set for each fold
                                  keys = n_clusters  values = k_folds x size for each cluster
        valid_cluster_size_predicted (dict): containing for each clustering with n_cluster the cluster size of the centroid assigned validation set clusters for each fold
                                             keys = n_clusters  values = k_folds x size for each cluster
        n_clusters (list): list number of k-clusters used in clustering
        folds (int): number of folds used in cross validation
        threshold (int): Prediction strength threshold indicating low prediction strength

    Returns:

        bad_cluster_sizes_cv (dict):  for each clustering with k_cluster proportion of datapoints from validation set falling into clusters with low prediction strength for each fold
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength

        bad_cluster_sizes_cv_predicted (dict):  for each labeling by k_cluster centroids based on training proportion of datapoints from validation set falling into clusters with low prediction strength for each fold
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength
    """

    bad_cluster_sizes_cv = {}
    bad_cluster_sizes_cv_predicted = {}
    for i in n_clusters:
        bad_cluster_sizes_cv[i] = []
        bad_cluster_sizes_cv_predicted[i] = []

    for f in range(folds):
        for i in n_clusters:
            bad_cluster_idx = np.where(
                k_predictions_strength_cv[i][f] < threshold)  # get index of cluster with prediction strength below 0.8

            # get proportion of burst in validation set falling into low predictive strength clusters for clustering with k clusters based on spectral cluster labels
            bad_cluster_sizes = (np.asarray((valid_cluster_size[i])[f] / np.sum(valid_cluster_size[i][f])) * 100)[
                bad_cluster_idx]

            # get proportion of burst in validation set falling into low predictive strength clusters for clustering with k clusters based on centroid labels
            bad_cluster_sizes_predicted = \
            (np.asarray((valid_cluster_size_predicted[i])[f] / np.sum(valid_cluster_size_predicted[i][f])) * 100)[
                bad_cluster_idx]  #

            bad_cluster_sizes_cv[i].append(list(bad_cluster_sizes))
            bad_cluster_sizes_cv_predicted[i].append(list(bad_cluster_sizes_predicted))

    return bad_cluster_sizes_cv, bad_cluster_sizes_cv_predicted


def calculate_average_bad_cluster_sizes(bad_cluster_sizes_cv, bad_cluster_sizes_cv_predicted, n_clusters=range(1, 21),
                                        folds=5):
    """ Calculate average proportion of datapoints from validation set falling into clusters with low prediction strength (below threshold = 0.8) per clustering with k clusters based on:
       a) clustres found by spectral clustering

    Args:
        bad_cluster_sizes_cv (dict):  for each clustering with k_cluster proportion of datapoints from validation set falling into clusters with low prediction strength for each fold
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength

        bad_cluster_sizes_cv_predicted (dict):  for each labeling by k_cluster centroids based on training proportion of datapoints from validation set falling into clusters with low prediction strength for each fold
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength
        n_clusters (list): list number of k-clusters used in clustering
        folds (int): number of folds used in cross validation

    Returns:

        average_bad_cluster_size (dict):  for each clustering with k_cluster proportion of datapoints from validation set falling into clusters with low prediction strength averaged by folds
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength

        average_bad_cluster_size_predicted (dict):  for each labeling by k_cluster centroids based on training proportion of datapoints from validation set falling into clusters with low prediction strength averaged by folds
                                      keys = n_clusters  values = k_folds x proportion of datapoints from validation set falling into clusters with low prediction strength
    """

    average_bad_cluster_size = []
    average_bad_cluster_size_predicted = []

    for i in n_clusters:
        bad_cluster_fold_sum = np.sum(sum(bad_cluster_sizes_cv[i], []))
        bad_cluster_fold_sum_predicted = np.sum(sum(bad_cluster_sizes_cv_predicted[i], []))

        average_bad_cluster_size.append(bad_cluster_fold_sum / folds)
        average_bad_cluster_size_predicted.append(bad_cluster_fold_sum_predicted / folds)

    return average_bad_cluster_size, average_bad_cluster_size_predicted


def get_low_individual_ps_bursts(data, train_folds, valid_folds, train_fold_labels, valid_fold_labels,
                                 predictions_strengths_cv_per_samples, n_clusters=range(1, 21), threshold=0.8):
    """ extract burst indices for burst with low individual prediction strength for clustering with k clusters per cluster
    Args:
        data (nd.array): Array containing data (n x m)
        train_folds (list of nd.arrays): list of k training set indices each with dimension n-(n/k) x m
        valid_folds (list of nd.arrays): list of k validation set indices each with dimension n/k x m
        train_fold_labels (list of lists): list of lists containing the cluster labels for each point in training set
        valid_fold_labels (list of lists): list of lists containing the cluster labels for each point in validation set
        predictions_strengths_cv_per_samples(list of lists): list of lists containin the prediction strength for individual bursts in each cluster by clustering with k clusters for each folds
                                                             1. dim n-folds 2.dim n-clusters 3.dim prediction strength for each cluster i in clustering with k clusters
        n_clusters (nd.array): range of clusters to use for clustering
        n_folds (int): number of folds the data is splitted
        threshold (float): cutoff for defining low individual prediction strength

    Returns:
        k_low_individual_ps_cv (list of lists): list of lists containin the burst indices for bursts with individual ps below threshold  strength for individual bursts in each cluster by clustering with k clusters for each folds
                                                keys = n_clusters  values = k_folds x n_clusters (burst indices of bursts from validation set with individual ps below threshold for each cluster)

    """

    k_low_individual_ps_cv = {}
    k_low_individual_ps_cv_sizes = {}
    k_low_individual_ps_cv_sizes_prop = {}

    for i in n_clusters:  # for each clustering ranging from 1 to max n_clusters
        k_low_individual_ps_cv[i] = []
        k_low_individual_ps_cv_sizes[i] = []
        k_low_individual_ps_cv_sizes_prop[i] = []

    for f, fold in enumerate(predictions_strengths_cv_per_samples):  # for each fold
        train_fold = train_folds[f]  # training set for fold k splitting
        valid_fold = valid_folds[f]  # validation set for fold k splitting

        train_labels = train_fold_labels[f]  # labels for training set
        valid_labels = valid_fold_labels[f]  # labels for validation set

        for j, k in enumerate(fold):  # for each clustering j with k clusters prediction strenght of fold
            train_labels_k = train_labels[j]
            valid_labels_k = valid_labels[j]

            # print(j, len(train_labels_k),len(valid_labels_k))

            k_low_individual_ps_cv_k = []
            k_low_individual_ps_cv_sizes_k = []
            k_low_individual_ps_cv_sizes_prop_k = []

            # print(len(np.concatenate(predictions_strengths_cv_per_samples[f][j])))

            for i in range(j + 1):
                index_in_class = np.where(np.asarray(
                    predictions_strengths_cv_per_samples[f][j][i]) < threshold)  # get position relative to class
                low_predictive_bursts = valid_fold[np.where(valid_labels_k == i)[0][index_in_class]]

                k_low_individual_ps_cv_k.append(low_predictive_bursts)  # get burst indices relative to overall data
                k_low_individual_ps_cv_sizes_k.append(len(low_predictive_bursts))
                k_low_individual_ps_cv_sizes_prop_k.append(
                    len(low_predictive_bursts) / len(valid_fold[np.where(valid_labels_k == i)[0]]))

            k_low_individual_ps_cv[j + 1].append(k_low_individual_ps_cv_k)
            k_low_individual_ps_cv_sizes[j + 1].append(k_low_individual_ps_cv_sizes_k)
            k_low_individual_ps_cv_sizes_prop[j + 1].append(k_low_individual_ps_cv_sizes_prop_k)

    return k_low_individual_ps_cv, k_low_individual_ps_cv_sizes, k_low_individual_ps_cv_sizes_prop


def get_low_and_high_ps_bursts_fold_with_labels(valid_folds, valid_fold_labels, k_low_individual_ps_bursts,
                                                k_low_individual_ps_cv_sizes, n_folds=2, n_clusters=range(1, 21)):
    k_low_ps_bursts_folds = {}
    k_high_ps_bursts_folds = {}
    low_ps_bursts_fold_labels = []
    high_ps_bursts_fold_labels = []

    for k in n_clusters:
        k_low_ps_bursts_folds[k] = [np.concatenate(k_low_individual_ps_bursts[k][f], axis=None) for f in range(n_folds)]
        high_ps_bursts_folds_k = [list(valid_fold) for valid_fold in valid_folds]
        for f in range(n_folds):
            for idx in k_low_ps_bursts_folds[k][f]:
                high_ps_bursts_folds_k[f].remove(idx)
        k_high_ps_bursts_folds[k] = high_ps_bursts_folds_k

    for f in range(n_folds):
        low_ps_bursts_labels_f = []
        high_ps_bursts_labels_f = []
        for k in n_clusters:
            high_ps_bursts_labels_f_k = valid_fold_labels[f][k - 1]
            remove_label_idx = []
            low_ps_bursts_labels_f_k = []
            for key in range(k):
                low_ps_bursts_labels_f_k += list(np.repeat(key, k_low_individual_ps_cv_sizes[k][f][key]))
            low_ps_bursts_labels_f.append(np.asarray(low_ps_bursts_labels_f_k))
            for idx in k_low_ps_bursts_folds[k][f]:
                remove_label_idx.append(np.where(valid_folds[f] == idx)[0][0])

            high_ps_bursts_labels_f_k = np.delete(high_ps_bursts_labels_f_k, remove_label_idx)
            high_ps_bursts_labels_f.append(np.asarray(high_ps_bursts_labels_f_k))

        low_ps_bursts_fold_labels.append(np.asarray(low_ps_bursts_labels_f))
        high_ps_bursts_fold_labels.append(np.asarray(high_ps_bursts_labels_f))

    return k_high_ps_bursts_folds, high_ps_bursts_fold_labels, k_low_ps_bursts_folds, low_ps_bursts_fold_labels



