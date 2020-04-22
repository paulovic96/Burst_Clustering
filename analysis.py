import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split
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


def main():
    data_dir = "data/"

    data = np.load(data_dir + "F_signal_noise_ambiguous_tau_amplitude.npy")
    #true_labels = np.repeat(range(12), 1000)

    amplitude_conditions = ["S", "S/M", "M", "M/L", "L"] #["S", "M", "L"]
    time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
    #["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
    # ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "medium_sharp_negative_skew", "wide_sharp_positive_skew", "medium_sharp_positive_skew"]
    ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]
    #["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]
    #["equal_medium", "medium_sharp_negative_skew", "sharp_medium_positive_skew"]
    samples_per_condition = 1000
    samples_per_ambiguous_condition = 400

    cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)
    #true_labels_ambiguous = get_labels(data,cluster_dict, ambiguous_conditions, true_clusters_starting_point = 0, new_clusters_amplitude_starting_point = 27, new_clusters_tau_starting_point = 12)

    true_labels_ambiguous = get_labels_by_layout(data,cluster_dict, list(cluster_dict.keys()), 9, layout_per_condition=(2,5))


    #training_set, training_set_indices = training_set_split.get_training_set(data, 12, 1000, 0.5)
    #training_set, training_set_indices, training_set_conditions = training_set_split.get_training_set_for_ambiguous_data(data, cluster_dict, ambiguous_conditions, ambiguous_tau = True, ambiguous_amplitude = False, proportion=0.5)

    #np.save("Toy_data/Ambiguous/training_set_conditions_ambiguous_tau_amplitude", training_set_conditions)

    #n_clusters = range(1,21)
    n_clusters = range(1, 50)
    k_conditions = [5, 10]
    reg_conditions = [None, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
    is_training = [False, True]
    save_dir = "Toy_data/Ambiguous/Ambiguous_Tau_Amplitude/"


    train_fold_indices, train_fold_indices = training_set_split.get_training_folds(data, cluster_dict)

    training_set = data[train_fold_indices[0]]
    validation_set = data[train_fold_indices[1]]

    cluster_data(validation_set, training_set, n_clusters,"euclidean", k_conditions,reg_conditions, is_training, save_dir)

if __name__== "__main__":
  main()







def further_analysis():
    ks = [5,10]
    regs = [None, 0.01, 0.1, 1, 5, 10, 20, 50, 100, "heuristic", "true clusters"]
    n_clusters = [21]
    training = True
    save_dir = "Toy_data/Ambiguous/Ambiguous_Tau"

    for n_cluster in n_clusters:
        for k in ks:
            #k = 5
            for reg in regs:
                heuristic = False
                true_clusters = False

                if reg == "heuristic":
                    heuristic = True

                if training:
                    if reg == "true clusters":
                        print("True Clusters")
                        true_clusters = True

                    elif heuristic:
                        reg = float(np.load(save_dir + "/k=%d_quin_rohe_heuristic_lambda_training.npy" % k))
                        # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % k)
                        eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=heuristic_training.npy" % k)
                        label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=heuristic_training.npy" % k)
                        print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=heuristic_training.npy" % k)
                    else:
                        reg = reg
                        # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % (k,str(reg)))
                        eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=%s_training.npy" % (k, str(reg)))
                        label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=%s_training.npy" % (k, str(reg)))
                        print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=%s_training.npy" % (k, str(reg)))

                else:
                    if reg == "true clusters":
                        print("True Clusters")
                        true_clusters = True

                    elif heuristic:
                        reg = float(np.load(save_dir + "/k=%d_quin_rohe_heuristic_lambda.npy" % k))
                        # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % k)
                        eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=heuristic.npy" % k)
                        label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=heuristic.npy" % k)
                        print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=heuristic.npy" % k)
                    else:
                        reg = reg
                        # eigvec = np.load(save_dir + "/eigenvectors/eigvec_k=%d_reg=%s.npy" % (k,str(reg)))
                        eigval = np.load(save_dir + "/eigenvalues/eigval_k=%d_reg=%s.npy" % (k, str(reg)))
                        label_predictions = np.load(save_dir + "/labels/labels_k=%d_reg=%s.npy" % (k, str(reg)))
                        print("Load labels from: " + save_dir + "/labels/labels_k=%d_reg=%s.npy" % (k, str(reg)))


                functions_for_plotting.plot_eigenvalues(eigval,true_cutoff=10, cutoff=10,eigenvalue_range=[0,50], figsize = None, configuration="k=%d, $\lambda$ = %s" % (k, str(reg)))


                true_labels = true_labels_ambiguous[training_set_indices]
                k_clusters = n_cluster
                if reg == "true clusters":
                    labels_k = true_labels
                else:
                    labels_k = label_predictions[k_clusters-1]
                n_bursts = 100

                columns = 4
                rows = 6

                mean = False

                if heuristic:
                    title = "Clusters k=%d,$\lambda=%.4f$" % (k, reg)
                    title += "(Quin-Rohe heuristic)"
                else:
                    if true_clusters:
                        title = "True Clusters"
                    else:
                        title = "Clusters k=%d,$\lambda=%s$" % (k, str(reg))
                if mean:
                    title += " (Mean)"
                if training:
                    title += " - Training Set"

                if heuristic:
                    save_file = "clusters_k=%d_reg=%.2f_heuristic_n_clusters=%d.pdf" % (k, reg,n_cluster)
                else:
                    if true_clusters:
                        save_file = "clusters.pdf"
                    else:
                        save_file = "clusters_k=%d_reg=%s_n_clusters=%d_training.pdf" % (k,str(reg),n_cluster)


                print(title)

                subplot_adjustments = [0.05,0.95,0.03,0.9,0.4, 0.15] #[0.05, 0.95, 0.05, 0.9, 0.4, 0.15]
                figsize = (20,20) #(20,10)

                if training:
                    functions_for_plotting.plot_clusters(training_set, true_labels, labels_k, k_clusters, rows, columns,
                                                         figsize=figsize, percent_true_cluster=False, n_bursts=n_bursts,
                                                         y_lim=(0, 16), plot_mean=mean, title=title,
                                                         subplot_adjustments=subplot_adjustments, savefile=save_file)
                else:
                    functions_for_plotting.plot_clusters(data,true_labels, labels_k, k_clusters,rows,columns, figsize = figsize, percent_true_cluster = False, n_bursts=n_bursts, y_lim = (0,16), plot_mean = mean, title = title, subplot_adjustments = subplot_adjustments, savefile=save_file)



