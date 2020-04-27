import numpy as np
import os
import matplotlib.pyplot as plt
from spectral_clustering import spectral_clustering
import functions_for_plotting
from asymmetric_laplacian_distribution import get_index_per_class, get_labels
from sklearn.cluster import KMeans
import training_set_split



save_dir = "Toy_data/Clearly_Separated/prediction_strength"
eigval_dir = save_dir + "/eigenvalues/"

eigenvalues_per_reg_k_5_training = {}
eigenvalues_per_reg_k_5_validation = {}
eigenvalues_per_reg_k_10_training = {}
eigenvalues_per_reg_k_10_validation = {}

for file in os.listdir(eigval_dir):
    if file.endswith(".npy") and "validation" in file:
        reg = file.replace(".npy", "")
        reg = (reg.split("_")[2]).split("=")[1]
        if "k=5" in file:
            eigenvalues_per_reg_k_5_validation[reg] = np.load(eigval_dir + file)
        else:
            eigenvalues_per_reg_k_10_validation[reg] = np.load(eigval_dir + file)
    elif file.endswith(".npy") and "training" in file:
        reg = file.replace(".npy", "")
        reg = (reg.split("_")[2]).split("=")[1]
        if "k=5" in file:
            eigenvalues_per_reg_k_5_training[reg] = np.load(eigval_dir + file)
        else:
            eigenvalues_per_reg_k_10_training[reg] = np.load(eigval_dir + file)


eigenvalues_per_reg_k_10_training.keys()


eigval = eigenvalues_per_reg_k_10_training["1"][0:101]
eigval_diff = np.diff(eigval)

plt.close("all")
fig,ax = plt.subplots()
ax.plot(range(1,len(eigval)), eigval_diff)
ax.set_xticks(range(1,len(eigval)))
ax.tick_params(axis='x', rotation=90)
np.argmax(eigval_diff)


