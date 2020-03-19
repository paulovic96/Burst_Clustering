import numpy as np
from spectral_clustering import spectral_clustering

data_dir = "data/"

data = np.load(data_dir + "F_signal_noise.npy")
n_clusters = range(1,11)

labels = spectral_clustering(data, "euclidean",n_clusters,  k=5, mutual = False, weighting = "distance", normalize = True, reg_lambda = None, save_laplacian = False, save_eigenvalues_and_vectors = False)
