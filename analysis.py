import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split
import prediction_strength
import os


def cluster_data(data, training_set, n_clusters,metric, k_conditions,reg_conditions, is_training, save_dir):
    for training in is_training:
        for k in k_conditions:
            for reg in reg_conditions:
                if training:
                    labels, eigvec, eigval = spectral_clustering(training_set, metric,n_clusters,  k=k, mutual = False, weighting = "distance", normalize = True, reg_lambda = reg, save_laplacian = False, save_eigenvalues_and_vectors = False)
                    np.save(save_dir + "labels/labels_k=%d_reg=%s_training" % (k, str(reg)),labels)
                    #np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=%s_training" % (k, str(reg)),eigvec)
                    np.save(save_dir + "eigenvalues/eigval_k=%d_reg=%s_training" % (k, str(reg)),eigval)
                else:
                    labels, eigvec, eigval = spectral_clustering(data, metric, n_clusters, k=k, mutual=False,weighting="distance", normalize=True, reg_lambda=reg,save_laplacian=False, save_eigenvalues_and_vectors=False)
                    np.save(save_dir + "labels/labels_k=%d_reg=%s" % (k, str(reg)), labels)
                    #np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=%s" % (k, str(reg)), eigvec)
                    np.save(save_dir + "eigenvalues/eigval_k=%d_reg=%s" % (k, str(reg)), eigval)

            if training:
                labels, eigvec, eigval = spectral_clustering(training_set, metric, n_clusters, k=k, mutual=False, weighting="distance", normalize=True,use_lambda_heuristic=True, reg_lambda=None,saving_lambda_file=save_dir + "k=%d_quin_rohe_heuristic_lambda_training" % k,save_laplacian=False, save_eigenvalues_and_vectors=False)
                np.save(save_dir + "labels/labels_k=%d_reg=heuristic_training" % k, labels)
                #np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=heuristic_training" % k, eigvec)
                np.save(save_dir + "eigenvalues/eigval_k=%d_reg=heuristic_training" % k, eigval)
            else:
                labels, eigvec, eigval = spectral_clustering(data, metric, n_clusters, k=k, mutual=False, weighting="distance", normalize=True,use_lambda_heuristic=True, reg_lambda=None,saving_lambda_file=save_dir + "k=%d_quin_rohe_heuristic_lambda" % k, save_laplacian=False, save_eigenvalues_and_vectors=False)
                np.save(save_dir + "labels/labels_k=%d_reg=heuristic" % k, labels)
                #np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=heuristic" % k,eigvec)
                np.save(save_dir + "eigenvalues/eigval_k=%d_reg=heuristic" % k, eigval)


def cluster_data_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    k_mean_labels = kmeans.labels_
    return k_mean_labels



data_dir = "data/"

data = np.load(data_dir + "ambiguous_data_tau_amplitude_F_signal_noise.npy")

amplitude_conditions = ["S", "S/M", "M", "M/L", "L"]
time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]

samples_per_condition = 1000
samples_per_ambiguous_condition = 400

cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)
true_labels_ambiguous = get_labels_by_layout(data,cluster_dict, list(cluster_dict.keys()), 9, layout_per_condition=(2,5))



n_clusters = range(1, 50)
k_conditions = [5, 10]
reg_conditions = [None, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
is_training = [False, True]
save_dir = "Toy_data/Ambiguous/Ambiguous_Tau_Amplitude/Prediction_Strength/"


train_fold_indices, train_fold_indices = training_set_split.get_training_folds(data, cluster_dict)

training_set = data[train_fold_indices[0]]
validation_set = data[train_fold_indices[1]]

cluster_data(validation_set, training_set, n_clusters,"euclidean", k_conditions,reg_conditions, is_training, save_dir)



######################################## Prediction Strength ##########################################################

data_dir = "data/"

data = np.load(data_dir + "clearly_separated_data_F_signal_noise.npy")

amplitude_conditions =["S", "M", "L"] # ["S", "S/M", "M", "M/L", "L"]
time_constant_conditions = ["equal_sharp", "equal_wide", "wide_sharp_negative_skew", "sharp_wide_positive_skew"]#["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
ambiguous_conditions = []#["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]

samples_per_condition = 1000
samples_per_ambiguous_condition = 400

cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)

import asymmetric_laplacian_distribution
importlib.reload(asymmetric_laplacian_distribution)

true_labels = asymmetric_laplacian_distribution.get_labels(data, cluster_dict)
clusters_ordered = list(range(0,13))
layout_label_mapping = labels_to_layout_mapping(clusters_ordered, 4, (1,4))



save_dir = "Toy_data/Clearly_Separated/Prediction_Strength/"
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


importlib.reload(functions_for_plotting)
functions_for_plotting.plot_clusters(training_set, true_labels[train_fold_indices[0]],train_labels[12], 3,4, layout_label_mapping,figsize=(20,20),n_bursts = 100,y_lim = (0,14))





prediction_strengths, cluster_sizes = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[0], train_fold_indices[1], train_labels, valid_labels, per_sample = False)
prediction_strength_per_sample, cluster_sizes_per_sample = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[0], train_fold_indices[1], train_labels, valid_labels, per_sample = True)

import importlib
importlib.reload(functions_for_plotting)

test = {}
test_per_sample = {}
test_sizes = {}
test_size_per_sample = {}
for key in prediction_strengths.keys():
    if key <= 20:
        test[key] = prediction_strengths[key]
        test_per_sample[key] = prediction_strength_per_sample[key]
        test_sizes[key] = cluster_sizes[key]
        test_size_per_sample[key] = cluster_sizes_per_sample[key]

importlib.reload(prediction_strength)
#low_clusters, low_cluster_sizes, low_cluster_sizes_percent = prediction_strength.get_clusters_below_threshold(test, test_sizes, 0.8)
low_points_in_clusters, low_points_in_clusters_sizes, low_points_in_clusters_percent, low_points_labels, low_ps_per_sample_per_k, high_points_in_clusters, high_points_labels, high_ps_per_sample_per_k = prediction_strength.get_points_in_clusters_below_and_above_threshold(data,test_per_sample, train_fold_indices[1], valid_labels, 0.9)







import matplotlib.pyplot as plt
plt.close("all")
#functions_for_plotting.plot_mean_prediction_strengths(test, test_sizes,threshold=None,title = "")
importlib.reload(functions_for_plotting)
plt.close("all")

functions_for_plotting.plot_number_burst_with_low_prediction_strength_per_k(low_points_in_clusters_sizes,low_ps_per_sample_per_k,6000,0.5, plot_proportion=False, plot_mean_low_ps = True)

#functions_for_plotting.plot_prediction_strength(test,test_sizes)

