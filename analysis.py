import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels, labels_to_layout_mapping
from sklearn.cluster import KMeans
import training_set_split
import seaborn as sns
import prediction_strength
import importlib
import matplotlib.pyplot as plt


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



#----------------------------------------------- DATA ------------------------------------------------------------------
data_dir = "data/"

data = np.load(data_dir + "clearly_separated_data_F_signal_noise.npy")#np.load(data_dir + "ambiguous_data_tau_amplitude_F_signal_noise.npy") #np.load(data_dir + "clearly_separated_data_F_signal_noise.npy")

amplitude_conditions = ["S", "M", "L"] #["S", "S/M", "M", "M/L", "L"] #["S", "M", "L"]
time_constant_conditions = ["equal_sharp", "equal_wide", "wide_sharp_negative_skew", "sharp_wide_positive_skew"]
#["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
#["equal_sharp", "equal_wide", "wide_sharp_negative_skew", "sharp_wide_positive_skew"]
ambiguous_conditions = []#["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]

samples_per_condition = 1000
samples_per_ambiguous_condition = 400

cluster_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)

true_labels = get_labels(data, cluster_dict)
clusters_ordered = list(range(0,len(cluster_dict)+1))
layout_label_mapping = labels_to_layout_mapping(clusters_ordered, 9, (2,5))


train_fold_indices, train_fold_indices = training_set_split.get_training_folds(data, cluster_dict,cluster_wise=True,folds = 2)
training_set = data[train_fold_indices[0]]
validation_set = data[train_fold_indices[1]]

#-------------------------------------------- Eigenvalues and Clusters -------------------------------------------------
"""
prediction_strength_dir = "Toy_data/Ambiguous/Ambiguous_Tau_Amplitude/Prediction_Strength/"


ks = [10]
regs = [None, 0.01, 0.1, 1, 5, 10, 20, 50, 100, "heuristic"]

for k in ks:
    for reg in regs:
        training_set_labels = np.load(prediction_strength_dir + "Labels/labels_k=%d_reg=%s_training.npy" % (k, str(reg)))
        validation_set_labels = np.load(prediction_strength_dir + "Labels/labels_k=%d_reg=%s_validation.npy" % (k, str(reg)))
        training_set_eigenvalues = np.load(prediction_strength_dir + "Eigenvalues/eigval_k=%d_reg=%s_training.npy" % (k, str(reg)))
        validation_set_eigenvalues = np.load(prediction_strength_dir + "Eigenvalues/eigval_k=%d_reg=%s_validation.npy" % (k, str(reg)))

        if reg == "heuristic":
            reg_train = np.round(np.load(prediction_strength_dir + "k=%d_quin_rohe_heuristic_lambda_training.npy" % k),2)
            reg_valid = np.round(np.load(prediction_strength_dir + "k=%d_quin_rohe_heuristic_lambda_validation.npy" % k),2)
        else:
            reg_train = reg
            reg_valid = reg

        train_labels = {}
        valid_labels = {}
        for i, labels in enumerate(training_set_labels):
            train_labels[i+1] = labels
            valid_labels[i+1] = validation_set_labels[i]


        subplot_adjustments = [0.05,0.93,0.02,0.92,0.9, 0.2] #[0.05, 0.95, 0.05, 0.87, 0.35, 0.18]#[0.05,0.95,0.03,0.9,0.4, 0.15]
        figsize = (40,30)#(30,20) #(20,20)

        functions_for_plotting.plot_eigenvalues(training_set_eigenvalues, true_cutoff=45, eigenvalue_range=[0,100], save_file="training_set_eigenvalues_k=%d_reg=%s.pdf" % (k,reg),configuration = "k=%d, $\lambda$=%s - Training Set" % (k,str(reg_train)))
        functions_for_plotting.plot_eigenvalues(validation_set_eigenvalues, true_cutoff=45, eigenvalue_range=[0,100], save_file ="validation_set_eigenvalues_k=%d_reg=%s.pdf" % (k,reg),configuration = "k=%d, $\lambda$=%s - Validation Set" % (k,str(reg_valid)))

        functions_for_plotting.plot_clusters(training_set, true_labels[train_fold_indices[0]],train_labels[45], 10,5, layout_label_mapping,figsize=figsize,n_bursts = 100,y_lim = (0,16),save_file="training_set_clusters_k=%d_reg=%s.pdf" % (k,reg),subplot_adjustments= subplot_adjustments, title= "Training Set Clusters \n k=%d, $\lambda$=%s" % (k,str(reg_train)))
        functions_for_plotting.plot_clusters(validation_set, true_labels[train_fold_indices[1]],valid_labels[45], 10,5, layout_label_mapping,figsize=figsize,n_bursts = 100,y_lim = (0,16),save_file="validation_set_clusters_k=%d_reg=%s.pdf" % (k,reg), subplot_adjustments= subplot_adjustments, title= "Validation Set Clusters \n k=%d, $\lambda$=%s" % (k,str(reg_valid)))


"""
#-------------------------------------------- Prediction Strength ------------------------------------------------------
def main():
    prediction_strength_dir = "Toy_data/Clearly_Separated/Prediction_Strength/"#Toy_data/Ambiguous/Ambiguous_Tau_Amplitude/Prediction_Strength/"

    ks = [5,10]
    regs = [None, 0.01, 0.1, 1, 5, 10, "heuristic"]


    for k in ks:
        for reg in regs:
            training_set_labels = np.load(prediction_strength_dir + "Labels/labels_k=%d_reg=%s_training.npy" % (k, str(reg)))
            validation_set_labels = np.load(prediction_strength_dir + "Labels/labels_k=%d_reg=%s_validation.npy" % (k, str(reg)))
            if reg == "heuristic":
                reg_train = np.round(np.load(prediction_strength_dir + "k=%d_quin_rohe_heuristic_lambda_training.npy" % k), 2)
                reg_valid = np.round(np.load(prediction_strength_dir + "k=%d_quin_rohe_heuristic_lambda_validation.npy" % k), 2)
            else:
                reg_train = reg
                reg_valid = reg

            train_labels = {}
            valid_labels = {}
            for i, labels in enumerate(training_set_labels):
                train_labels[i+1] = labels
                valid_labels[i+1] = validation_set_labels[i]


            validation_prediction_strengths, validation_cluster_sizes = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[0], train_fold_indices[1], train_labels, valid_labels, per_sample = False)
            training_prediction_strengths, training_cluster_sizes = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[1], train_fold_indices[0], valid_labels, train_labels, per_sample = False)

            np.save("Toy_data/Clearly_Separated/Prediction_Strength/validation_set_prediction_strength_k=%d_reg=%s.npy" % (k,reg_valid), validation_prediction_strengths)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/validation_set_cluster_sizes_k=%d_reg=%s.npy" % (k,reg_valid),validation_cluster_sizes)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/training_set_prediction_strength_k=%d_reg=%s.npy" % (k,reg_train), training_prediction_strengths)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/training_set_cluster_sizes_k=%d_reg=%s.npy" % (k,reg_train),training_cluster_sizes)

            threshold = 0.8
            n_total = 6000
            #color=sns.color_palette('Reds', 55, )[::-1]
            #figsize=(20,10)

            #functions_for_plotting.plot_prediction_strength(validation_prediction_strengths,validation_cluster_sizes,figsize=figsize, color=color,save_file="validation_set_prediction_strength_k=%d_reg=%s.pdf" % (k,reg_valid), title = "Prediction Strength per Cluster\n(Validation Set)")
            #functions_for_plotting.plot_prediction_strength(training_prediction_strengths,training_cluster_sizes,figsize=figsize, color=color,save_file="training_set_prediction_strength_k=%d_reg=%s.pdf" % (k,reg_train), title = "Prediction Strength per Cluster\n(Training Set)")

            #functions_for_plotting.plot_mean_prediction_strengths(validation_prediction_strengths,threshold=None,save_file="validation_set_mean_prediction_strength_k=%d_reg=%s.pdf" % (k,reg_valid), title = "Mean Prediction Strength\n(Validation Set)")
            #functions_for_plotting.plot_mean_prediction_strengths(training_prediction_strengths,threshold=None,save_file="training_set_mean_prediction_strength_k=%d_reg=%s.pdf" % (k,reg_train), title = "Mean Prediction Strength\n(Training Set)")

            validation_low_clusters, validation_low_cluster_sizes, validation_low_cluster_sizes_percent = prediction_strength.get_clusters_below_threshold(validation_prediction_strengths, validation_cluster_sizes, threshold=threshold)
            training_low_clusters, training_low_cluster_sizes, training_low_cluster_sizes_percent = prediction_strength.get_clusters_below_threshold(training_prediction_strengths, training_cluster_sizes, threshold=threshold)

            functions_for_plotting.plot_number_bursts_in_low_clusters_per_k(validation_low_cluster_sizes,k_clusters=list(range(1,21)),n_total=n_total,threshold=threshold,save_file="Toy_data/Clearly_Separated/Prediction_Strength/validation_clusters_below_threshold_k=%d_reg=%s.pdf" % (k,reg_valid))
            functions_for_plotting.plot_number_bursts_in_low_clusters_per_k(training_low_cluster_sizes,k_clusters=list(range(1,21)),n_total=n_total,threshold=threshold,save_file="Toy_data/Clearly_Separated/Prediction_Strength/training_clusters_below_threshold_k=%d_reg=%s.pdf" % (k,reg_train))


            validation_prediction_strength_per_sample, validation_cluster_sizes_per_sample = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[0], train_fold_indices[1], train_labels, valid_labels, per_sample = True)
            training_prediction_strength_per_sample, training_cluster_sizes_per_sample = prediction_strength.get_prediction_strength_per_k(data, train_fold_indices[1], train_fold_indices[0], valid_labels, train_labels, per_sample = True)
            
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/validation_set_prediction_strength_per_sample_k=%d_reg=%s.npy" % (k,reg_valid), validation_prediction_strength_per_sample)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/validation_set_cluster_sizes_per_sample_k=%d_reg=%s.npy" % (k,reg_valid),validation_cluster_sizes_per_sample)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/training_set_prediction_strength_per_sample_k=%d_reg=%s.npy" % (k,reg_train), training_prediction_strength_per_sample)
            np.save("Toy_data/Clearly_Separated/Prediction_Strength/training_set_cluster_sizes_per_sample_k=%d_reg=%s.npy" % (k,reg_train),training_cluster_sizes_per_sample)
            
            
            _, validation_low_points_in_clusters_sizes, _, _, validation_low_ps_per_sample_per_k, _, _, _ = prediction_strength.get_points_in_clusters_below_and_above_threshold(data,validation_prediction_strength_per_sample, train_fold_indices[1], valid_labels, threshold)
            _, training_low_points_in_clusters_sizes, _, _, training_low_ps_per_sample_per_k, _, _, _ = prediction_strength.get_points_in_clusters_below_and_above_threshold(data,training_prediction_strength_per_sample, train_fold_indices[0], train_labels, threshold)

            functions_for_plotting.plot_number_burst_with_low_prediction_strength_per_k(validation_low_points_in_clusters_sizes,validation_low_ps_per_sample_per_k,n_total=n_total,k_clusters=list(range(1,21)),threshold=threshold,plot_proportion=False, plot_mean_low_ps = True,save_file="Toy_data/Clearly_Separated/Prediction_Strength/validation_individual_points_below_threshold_k=%d_reg=%s.pdf" % (k,reg_valid))
            functions_for_plotting.plot_number_burst_with_low_prediction_strength_per_k(training_low_points_in_clusters_sizes,training_low_ps_per_sample_per_k,n_total=n_total,k_clusters=list(range(1,21)),threshold=threshold,plot_proportion=False, plot_mean_low_ps = True,save_file="Toy_data/Clearly_Separated/Prediction_Strength/training_individual_points_below_threshold_k=%d_reg=%s.pdf" % (k,reg_train))




if __name__== "__main__":
  main()

