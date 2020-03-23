import numpy as np
from spectral_clustering import spectral_clustering

data_dir = "data/"

data = np.load(data_dir + "F_signal_noise.npy")
#np.random.shuffle(data)
n_clusters = range(1,21)

#labels, eigvec, eigval = spectral_clustering(data, "euclidean",n_clusters,  k=10, mutual = False, weighting = "distance", normalize = True, reg_lambda = 0.01, save_laplacian = False, save_eigenvalues_and_vectors = False)


labels = np.load("Toy_data/labels/labels_k=10_reg=0.1.npy")


# check if all cluster captured correctly
labels_12 = labels[11]


for i in range(12):
    burst_indices = np.where(labels_12 == i)[0]
    wrongly_clustered_data = len(np.where(np.diff(burst_indices) > 1)[0])
    print("Cluster %d: %d wrongly clustered data points!" % ((i+1),wrongly_clustered_data))



plot_cluster_examples(data, labels_12, 12,3,4, figsize = (30,25), burst_percent = False, n_bursts=50, y_lim = (0,16),plot_mean=True)
plt.close()

