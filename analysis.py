import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split
import os


SUBPLOT_ADJUSTMENTS = [0.05,0.95,0.03,0.9,0.4, 0.15]
FIGSIZE = (20,20)


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

def get_labels_by_layout(data,cluster_dict, cluster_order, clusters_per_condition_for_layout, layout_per_condition=(2,5)):
    labels = np.zeros(len(data))
    shift_needed_to_stay_in_layout = layout_per_condition[0]*layout_per_condition[1] - clusters_per_condition_for_layout
    label = 0
    for i, key in enumerate(cluster_order):
        if i == 0:
            labels[cluster_dict[key][0]:cluster_dict[key][1] + 1] = label
        else:
            if i % clusters_per_condition_for_layout == 0:
                label += (1 + shift_needed_to_stay_in_layout)
            else:
                label += 1

            labels[cluster_dict[key][0]:cluster_dict[key][1] + 1] = label
    return labels



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


