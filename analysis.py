import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split


def cluster_data(data, training_set, n_clusters, k_conditions,reg_conditions, is_training, save_dir):
    for training in is_training:
        for k in k_conditions:
            for reg in reg_conditions:
                if training:
                    labels, eigvec, eigval = spectral_clustering(training_set, "euclidean",n_clusters,  k=k, mutual = False, weighting = "distance", normalize = True, reg_lambda = reg, save_laplacian = False, save_eigenvalues_and_vectors = False)
                    np.save(save_dir + "labels/labels_k=%d_reg=%s_training" % (k, str(reg)),labels)
                    np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=%s_training" % (k, str(reg)),eigvec)
                    np.save(save_dir + "eigenvalues/eigval_k=%d_reg=%s_training" % (k, str(reg)),eigval)

                else:
                    labels, eigvec, eigval = spectral_clustering(data, "euclidean", n_clusters, k=k, mutual=False,
                                                                 weighting="distance", normalize=True, reg_lambda=reg,
                                                                 save_laplacian=False, save_eigenvalues_and_vectors=False)
                    np.save(save_dir + "labels/labels_k=%d_reg=%s" % (k, str(reg)), labels)
                    np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=%s" % (k, str(reg)), eigvec)
                    np.save(save_dir + "eigenvalues/eigval_k=%d_reg=%s" % (k, str(reg)), eigval)

           if training:
                labels, eigvec, eigval = spectral_clustering(training_set, "euclidean", n_clusters, k=k, mutual=False, weighting="distance", normalize=True,use_lambda_heuristic=True, reg_lambda=None,saving_lambda_file=save_dir + "k=%d_quin_rohe_heuristic_lambda_training" % k,save_laplacian=False, save_eigenvalues_and_vectors=False)
                np.save(save_dir + "labels/labels_k=%d_reg=heuristic_training" % k, labels)
                np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=heuristic_training" % k, eigvec)
                np.save(save_dir + "eigenvalues/eigval_k=%d_reg=heuristic_training" % k, eigval)
            else:
                labels, eigvec, eigval = spectral_clustering(data, "euclidean", n_clusters, k=k, mutual=False, weighting="distance", normalize=True,use_lambda_heuristic=True, reg_lambda=None,saving_lambda_file=save_dir + "k=%d_quin_rohe_heuristic_lambda" % k, save_laplacian=False, save_eigenvalues_and_vectors=False)
                np.save(save_dir + "labels/labels_k=%d_reg=heuristic" % k, labels)
                np.save(save_dir + "eigenvectors/eigvec_k=%d_reg=heuristic" % k,eigvec)
                np.save(save_dir + "eigenvalues/eigval_k=%d_reg=heuristic" % k, eigval)

def cluster_data_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    k_mean_labels = kmeans.labels_
    return k_mean_labels





data_dir = "data/"

data = np.load(data_dir + "F_signal_noise_ambiguous.npy")
#true_labels = np.repeat(range(12), 1000)


amplitude_conditions = ["S", "M", "L"] #["S", "S/M", "M", "M/L", "L"]
time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "medium_sharp_negative_skew", "wide_sharp_positive_skew", "medium_sharp_positive_skew"]
#["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","wide_sharp_positive_skew", "wide_medium_positive_skew" ,"medium_sharp_positive_skew"]
ambiguous_conditions = ["equal_medium", "medium_sharp_negative_skew", "medium_sharp_positive_skew"]
#["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "wide_medium_positive_skew", "medium_sharp_positive_skew"]
samples_per_condition = 1000
samples_per_ambiguous_condition = 400

cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)
true_labels_ambiguous = get_labels(data,cluster_dict, ambiguous_conditions, true_clusters_starting_point = 0, new_clusters_amplitude_starting_point = 27, new_clusters_tau_starting_point = 12)

#training_set, training_set_indices = training_set_split.get_training_set(data, 12, 1000, 0.5)
training_set, training_set_indices, training_set_conditions = training_set_split.get_training_set_for_ambiguous_data(data, cluster_dict, ambiguous_conditions,ambiguous_tau = True, ambiguous_amplitude = False, proportion=0.5)

#np.save("Toy_data/Ambiguous/training_set_conditions_ambiguous_tau", training_set_conditions)

#n_clusters = range(1,21)
n_clusters =  range(1, 30)
k_conditions = [5]#[5, 10]
reg_conditions = []#[None, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
is_training = [False, True]
save_dir = "Toy_data/Ambiguous/Ambiguous_Tau/"


cluster_data(data, training_set, n_clusters, k_conditions,reg_conditions, is_training, save_dir)



k = 5
heuristic = True
if heuristic:
    reg = float(np.load(save_dir + "/k=%d_quin_rohe_heuristic_lambda.npy" % k))
    # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % k)
    eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=heuristic.npy" % k)
    label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=heuristic.npy" % k)
    print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=heuristic.npy" % k)
else:
    reg = 100
    # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % (k,str(reg)))
    eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=%s.npy" % (k, str(reg)))
    label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=%s.npy" % (k, str(reg)))
    print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=%s.npy" % (k, str(reg)))



functions_for_plotting.plot_eigenvalues(eigval,true_cutoff=21, cutoff=13,eigenvalue_range=[0,50], figsize = None, configuration="k=%d, $\lambda$ = %s" % (k, str(reg)))



k_clusters = 2
labels_k = label_predictions[k_clusters-1]
n_bursts = None

columns = 4
rows = 3

training = False
mean = False

if heuristic:
    title = "Clusters k=%d,$\lambda=%.4f$" % (k, reg)
    title += "(Quin-Rohe heuristic)"
else:
    title = "Clusters k=%d,$\lambda=%s$" % (k, str(reg))
if mean:
    title += " (Mean)"
if training:
    title += " -T raining Set"
print(title)

functions_for_plotting.plot_clusters(data,true_labels, labels_k, k_clusters,rows,columns, figsize = (20,15), percent_true_cluster = False, n_bursts=n_bursts, y_lim = (0,16), plot_mean = mean, title = title)

