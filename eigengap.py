import numpy as np
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split



save_dir = "Toy_data/Ambiguous/Ambiguous_Tau"
eigval_dir = save_dir + "/eigenvalues/"

eigenvalues_per_reg_k_5 = {}
eigenvalues_per_reg_k_10 = {}

for file in os.listdir(eigval_dir):
    if file.endswith(".npy") and "training" not in file:
        reg = file.replace(".npy", "")
        reg = (reg.split("_")[2]).split("=")[1]
        if "k=5" in file:
            eigenvalues_per_reg_k_5[reg] = np.load(eigval_dir + file)
        else:
            eigenvalues_per_reg_k_10[reg] = np.load(eigval_dir + file)


eigval = eigenvalues_per_reg_k_10["None"][0:100]
eigval_diff = np.diff(eigval)
np.argmax(eigval_diff)


