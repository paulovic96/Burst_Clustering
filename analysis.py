import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
import matplotlib.pyplot as plt

data_dir = "data/"

data = np.load(data_dir + "F_signal_noise_ambiguous.npy")
true_labels = np.repeat(range(12), 1000)

seed = np.random.seed(42)
training_split = list(range(12))
np.random.shuffle(training_split)
training_split = np.sort(training_split[:6])

training_set_indices = []
for i in training_split:
    training_set_indices += list(range(i*1000,(i+1)*1000))
training_set_indices = np.asarray(training_set_indices)

training_set = data[training_set_indices]

n_clusters = range(1,21)

labels, eigvec, eigval = spectral_clustering(training_set, "euclidean",n_clusters,  k=10, mutual = False, weighting = "distance", normalize = True, reg_lambda = 0.1, save_laplacian = False, save_eigenvalues_and_vectors = False)

np.save("Toy_data/labels/labels_k_10_reg=0.1_training",labels)
np.save("Toy_data/eigenvectors/eigvec_k=10_reg=0.1_training",eigvec)
np.save("Toy_data/eigenvalues/eigval_k=10_reg=0.1_training",eigval)


labels = np.load("Toy_data/Labels/labels_k=10_reg=0.1.npy")


# check if all cluster captured correctly
labels_6 = labels[5]


for i in range(6):
    burst_indices = np.where(labels_6 == i)[0]
    wrongly_clustered_data = len(np.where(np.diff(burst_indices) > 1)[0])
    print("Cluster %d: %d wrongly clustered data points!" % ((i+1),wrongly_clustered_data))


_, idx = np.unique(labels_12, return_index=True)
arrangement = labels_12[np.sort(idx)]


functions_for_plotting.plot_cluster_examples(training_set, labels_6, 6, 2, 4, figsize = (30,25), burst_percent = False, n_bursts=None, y_lim = (0,16),plot_mean=False, arrangement = None, title = None)
functions_for_plotting.plot_cluster_distribution(data, true_labels, 12, 3, 4,figsize = (30,25), y_lim = None, arrangement = None)
plt.close()




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(training_set)
k_mean_labels = kmeans.labels_
