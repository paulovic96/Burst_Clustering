import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

def plot_cluster_examples(data, labels, k_clusters,rows,columns, figsize = (30,25), burst_percent = False, n_bursts=None, y_lim = None, plot_mean = False):
    if k_clusters < 10:
        colors = ["C" + str(i) for i in range(k_clusters)]
    else:
        colors = cm.rainbow(np.linspace(0, 1, k_clusters))

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    rows = rows
    for i in range(k_clusters):
        class_i = data[np.where(labels == i)]
        #class_i = np.random.permutation(class_i)

        if i < k_clusters:
            count = i + 1
        ax = fig.add_subplot(rows, columns, count)
        ax.set_xlabel("Time", fontsize = 12)
        if plot_mean:
            if burst_percent:
                ax.set_title("Class %i Mean (%i Bursts = %.2f %%)" % ((i + 1), len(class_i), len(class_i) / len(data) * 100),
                             fontsize=12)
            else:
                ax.set_title("Class %i Mean (%i Bursts )" % ((i + 1), len(class_i)), fontsize=12)

            ax.plot(np.mean(class_i, axis = 0))

        else:
            if burst_percent:
                ax.set_title("Class %i (%i Bursts = %.2f %%)" % ((i+1),len(class_i), len(class_i)/len(data) * 100), fontsize = 12)
            else:
                ax.set_title("Class %i (%i Bursts )" % ((i+1),len(class_i)), fontsize = 12)

            if not n_bursts:
                n_bursts = len(class_i)

            for burst in class_i[0:n_bursts]:
                ax.plot(burst)

        if y_lim:
            ax.set_ylim(y_lim)




def plot_cluster_distribution(data, labels, k_clusters,rows,columns, figsize = (30,25), y_lim = None):
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(k_clusters):
        if i < k_clusters:
            count = i + 1
        ax = fig.add_subplot(rows, columns, count)
        class_i = data[np.where(labels == i)]
        sns.distplot(np.concatenate(class_i, axis=0), ax=ax)

        if y_lim:
            ax.set_ylim(y_lim)
