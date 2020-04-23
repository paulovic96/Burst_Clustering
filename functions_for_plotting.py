import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns


def plot_cluster_distribution(data, labels, k_clusters,rows, columns, figsize = (30,25), y_lim = None, arrangement = None):
    fig = plt.figure(figsize=figsize)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(k_clusters):
        if arrangement is None:
            class_i = data[np.where(labels == i)]
        else:
            class_i = data[np.where(labels == arrangement[i])]

        if i > 0:
            ax = fig.add_subplot(rows, columns, i + 1, sharey=ax)
        else:
            ax = fig.add_subplot(rows, columns, i + 1)
            if y_lim:
                ax.set_ylim(y_lim)

        sns.distplot(np.concatenate(class_i, axis=0), ax=ax)

        if i > 3:
            ax_inset = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax_inset)
            if i < 8:
                x1, x2, y1, y2 = -0.2, 2, 0, 3
            else:
                x1, x2, y1, y2 = -0.2, 5, 0, 2.5
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)

            ax.indicate_inset_zoom(ax_inset)



def plot_cluster_distribution_comparison(data, labels, k_clusters, figsize = (30,25), y_lim = None, arrangement = None):
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    ax1_inset = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
    ax2_inset = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    ax3_inset = ax3.inset_axes([0.5, 0.5, 0.47, 0.47])
    ax4_inset = ax4.inset_axes([0.5, 0.5, 0.47, 0.47])

    for i in range(k_clusters):
        if arrangement is None:
            class_i = data[np.where(labels == i)]
        else:
            class_i = data[np.where(labels == arrangement[i])]

        if i%4 == 0:
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax1, norm_hist=True)
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax1_inset)
        elif i%4 == 1:
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax2, norm_hist=True)
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax2_inset)
        elif i%4 == 2:
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax3, norm_hist=True)
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax3_inset)
        else:
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax4, norm_hist=True)
            sns.distplot(np.concatenate(class_i, axis=0), ax=ax4_inset)

        if y_lim:
            ax1.set_ylim(y_lim)
            ax2.set_ylim(y_lim)
            ax3.set_ylim(y_lim)
            ax4.set_ylim(y_lim)

        x1, x2, y1, y2 = -0.2, 3, 0, 6
        ax1_inset.set_xlim(x1, x2)
        ax1_inset.set_ylim(y1, y2)

        x1, x2, y1, y2 = -0.2, 8, 0, 2
        ax2_inset.set_xlim(x1, x2)
        ax2_inset.set_ylim(y1, y2)

        x1, x2, y1, y2 = -0.2, 2, 0, 2.5
        ax3_inset.set_xlim(x1, x2)
        ax3_inset.set_ylim(y1, y2)

        x1, x2, y1, y2 = -0.2, 2, 0, 2.5
        ax4_inset.set_xlim(x1, x2)
        ax4_inset.set_ylim(y1, y2)

        ax1.indicate_inset_zoom(ax1_inset)
        ax2.indicate_inset_zoom(ax2_inset)
        ax3.indicate_inset_zoom(ax3_inset)
        ax4.indicate_inset_zoom(ax4_inset)


def get_true_label_mapping(true_labels, label_predictions):
    corresponding_true_classes_for_prediction = {}
    highest_overlap_class_for_prediction = {}

    for label in np.unique(label_predictions):
        class_i = np.where(label_predictions == label)[0]
        corresponding_true_labels = true_labels[class_i]
        true_classes, overlap_count = np.unique(corresponding_true_labels, return_counts=True)
        corresponding_true_classes_for_prediction[label] = [true_classes, overlap_count]

        highest_overlap_class_for_prediction[label] = [true_classes[np.argmax(overlap_count)], np.amax(overlap_count)]
    return corresponding_true_classes_for_prediction,highest_overlap_class_for_prediction


def get_splitted_clusters(highest_overlap_class_for_prediction):
    splitted_classes, count = np.unique(np.asarray(list(highest_overlap_class_for_prediction.values()))[:,0], return_counts=True)
    splitted_classes = splitted_classes[np.where(count > 1)[0]]
    count = count[np.where(count > 1)[0]]
    new_clusters = np.asarray(list(highest_overlap_class_for_prediction.keys()))
    splitted_into_new_clusters = {}
    for splitted_class in splitted_classes:
        splitted_into_new_clusters[splitted_class] = list(new_clusters[np.where(np.asarray(list(highest_overlap_class_for_prediction.values()))[:,0] == splitted_class)])

    return splitted_classes, count, splitted_into_new_clusters

def get_merged_clusters(corresponding_true_class_for_prediction):
    merged_clusters = np.asarray(list(corresponding_true_class_for_prediction.values()))[:, 0]
    count = np.asarray(list(map(len, merged_clusters)))
    new_clusters = np.asarray(list(corresponding_true_class_for_prediction.keys()))
    return merged_clusters[np.where(count > 1)[0]], new_clusters[np.where(count > 1)[0]]

def get_new_layout(k_clusters, rows, columns, splitted_classes, split_count):
    layout = np.arange(k_clusters).reshape((rows,columns))
    rows_per_column = np.repeat(rows, columns)
    for i,splitted_class in enumerate(splitted_classes):
        position_in_layout = np.where(layout == splitted_class)
        rows_per_column[position_in_layout[1][0]] = max(rows_per_column[position_in_layout[1][0]], len(layout)*split_count[i])
    return max(rows_per_column), rows_per_column




def plot_clusters(data, true_labels,labels_k, k_clusters,rows,columns, figsize = (20,20), percent_true_cluster = False, n_bursts=None, y_lim = None, plot_mean = False, title = None, subplot_adjustments = [0.05,0.95,0.03,0.9,0.4, 0.15], savefile = "test.pdf"):
    corresponding_true_class_for_prediction, highest_overlap_class_for_prediction = get_true_label_mapping(true_labels, labels_k)
    splitted_classes, split_count, new_clusters_splitted = get_splitted_clusters(highest_overlap_class_for_prediction)
    merged_classes, new_clusters_merged = get_merged_clusters(corresponding_true_class_for_prediction)
    position_count_for_splitted_classes = np.zeros(len(split_count))
    empty_true_clusters = list(np.unique(true_labels))

    normal_layout = np.arange(rows*columns).reshape((rows,columns))

    # Figure adjustments
    plt.close("all")
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=25)

    fig.subplots_adjust(left=subplot_adjustments[0], right=subplot_adjustments[1], bottom=subplot_adjustments[2],top=subplot_adjustments[3], hspace=subplot_adjustments[4], wspace=subplot_adjustments[5])
    # Outer Grid
    outer_grid = matplotlib.gridspec.GridSpec(rows, columns)

    # iterate over predicted labels
    for i in np.unique(labels_k):
        class_i = data[np.where(labels_k == i)] # get corresponding data points
        corresponding_true_label = highest_overlap_class_for_prediction[i][0] # get true label (= position)
        corresponding_column = int(np.where(normal_layout == corresponding_true_label)[1][0]) # column position
        corresponding_row = int(np.where(normal_layout == corresponding_true_label)[0][0])  # row position

        if corresponding_true_label in splitted_classes:
            true_class_size = len(data[np.where(true_labels == corresponding_true_label)])
            splitted_into = split_count[np.where(splitted_classes==corresponding_true_label)[0]][0]
            inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(splitted_into, 1, subplot_spec=outer_grid[corresponding_row,corresponding_column], wspace=0.0, hspace=0.0) # inner grid for splitted clusters
            position_count = position_count_for_splitted_classes[np.where(splitted_classes==corresponding_true_label)[0]] # get stacked position in inner grid
            row_start = int(position_count)


            if position_count == 0: # first plot of splitted cluster
                true_clusters = []
                true_cluster_counts = []
                true_cluster_percents = []

                for new_cluster_splitted in new_clusters_splitted[corresponding_true_label]:
                    true_clusters.append(list(corresponding_true_class_for_prediction[new_cluster_splitted][0]))
                    true_cluster_count = corresponding_true_class_for_prediction[new_cluster_splitted][1]
                    true_cluster_percent = list(np.round(true_cluster_count/true_class_size * 100, decimals=1))
                    true_cluster_percents += [true_cluster_percent]
                    true_cluster_counts += [list(true_cluster_count)]


                if percent_true_cluster:
                    cluster_title = ("Cluster [" + ', '.join(['%d'] * len(new_clusters_splitted[corresponding_true_label])) + "]" +
                                    "\nTrue Cluster: " + ', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_clusters]) +
                                    "\n%%: " + ', '.join(['[' + ', '.join(['%.1f'] * len(x)) +']' for x in true_cluster_percents])) % tuple(new_clusters_splitted[corresponding_true_label] + sum(true_clusters,[]) + sum(true_cluster_percents,[]))
                else:
                    cluster_title = ("Cluster [" + ', '.join(['%d'] * len(new_clusters_splitted[corresponding_true_label])) + "]" +
                                    "\nTrue Cluster: " + ', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_clusters]) +
                                    "\n#:  " + ', '.join(['[' + ', '.join(['%.1f'] * len(x)) +']' for x in true_cluster_counts])) % tuple(new_clusters_splitted[corresponding_true_label] + sum(true_clusters,[]) + sum(true_cluster_counts,[]))

            row_end = int(row_start + 1)



            if position_count == 0:
                topax = fig.add_subplot(inner_grid[row_start:row_end, 0])
                topax.set_title(cluster_title,loc="left", fontsize = 12) # only set title on top of split

                if plot_mean:
                    topax.plot(np.mean(class_i, axis=0))
                else:
                    if not n_bursts:
                        step = 1

                    else:
                        step = int(np.ceil(len(class_i)/n_bursts))

                    for burst in class_i[::step]:
                        topax.plot(burst,rasterized=True)

                if y_lim:
                    topax.set_ylim(y_lim)

                plt.setp(topax.get_xticklabels(), visible=False)

            else:
                if position_count < splitted_into-1:
                    ax = fig.add_subplot(inner_grid[row_start:row_end, 0], sharex=topax)
                    ax.set_xticklabels([])
                    plt.setp(ax.get_xticklabels(), visible=False)

                else:# position_count == splitted_into-1:
                    ax = fig.add_subplot(inner_grid[row_start:row_end, 0], sharex=topax)
                    ax.set_xlabel("Time")

                if plot_mean:
                    ax.plot(np.mean(class_i, axis=0),rasterized=True)
                else:
                    if not n_bursts:
                        step = 1
                    else:
                        step = int(np.ceil(len(class_i)/n_bursts))

                    for burst in class_i[::step]:
                        ax.plot(burst,rasterized=True)

                if y_lim:
                    ax.set_ylim(y_lim)
            position_count_for_splitted_classes[np.where(splitted_classes == corresponding_true_label)[0]] += 1
            if corresponding_true_label in empty_true_clusters:
                empty_true_clusters.remove(corresponding_true_label)

        else:
            row_start = corresponding_row
            row_end = int(corresponding_row + 1)

            if i in new_clusters_merged:
                corresponding_merged_true_clusters = list(merged_classes[np.where(new_clusters_merged == i)[0]][0])
                corresponding_merged_true_clusters.remove(corresponding_true_label)
                for merged_class in corresponding_merged_true_clusters:
                    if merged_class in empty_true_clusters:
                        row_i = int(np.where(normal_layout == merged_class)[0][0])
                        column_i =  int(np.where(normal_layout == merged_class)[1][0])
                        ax = fig.add_subplot(outer_grid[row_i:(row_i + 1), column_i])
                        ax.plot()
                        ax.set_xticklabels([])
                        ax.set_title("True Cluster: [%d]" % merged_class,loc="left", fontsize = 12)
                        if y_lim:
                            ax.set_ylim(y_lim)

            ax = fig.add_subplot(outer_grid[row_start:row_end, corresponding_column])

            true_clusters = corresponding_true_class_for_prediction[i][0]
            true_cluster_counts = corresponding_true_class_for_prediction[i][1]
            true_class_size = [len(data[np.where(true_labels == true_cluster)[0]]) for i,true_cluster in enumerate(true_clusters)]
            true_cluster_percents = [true_cluster_counts[i]/true_class_size[i] for i,true_cluster in enumerate(true_clusters)]

            if len(true_clusters) > 1:
                if percent_true_cluster:
                    cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(
                                            ['%d'] * len(true_clusters)) + "]" + "\n%%: [" + ', '.join(
                                            ['%.1f'] * len(true_cluster_percents)) + "]") % tuple([i] + list(true_clusters) + list(np.round(np.asarray(true_cluster_percents) * 100, decimals=1)))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
                                     #"\n%%: [" + ', '.join(['%d/%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + sum(list(map(list, zip(true_cluster_counts, true_class_size))),[]))
                else:
                    cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(
                        ['%d'] * len(true_clusters)) + "]" + "\n#: [" + ', '.join(
                        ['%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + list(
                        true_cluster_counts))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
            else:
                if percent_true_cluster:
                    cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(
                        ['%d'] * len(true_clusters)) + "] " + "%%: [" + ', '.join(
                        ['%.1f'] * len(true_cluster_percents)) + "]") % tuple([i] + list(true_clusters) + list(np.round(np.asarray(true_cluster_percents) * 100, decimals=1)) ) # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
                        #"%%: [" + ', '.join(['%d/%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + sum(list(map(list, zip(true_cluster_counts, true_class_size))), []))

                else:
                    cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(
                        ['%d'] * len(true_clusters)) + "]" + " #: [" + ', '.join(
                        ['%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + list(
                        true_cluster_counts))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
            ax.set_title(cluster_title, loc="left", fontsize = 12)
            ax.set_xlabel("Time")



            if plot_mean:
                ax.plot(np.mean(class_i, axis = 0), rasterized=True)
            else:
                if not n_bursts:
                    step = 1
                else:
                    step = int(np.ceil(len(class_i)/n_bursts))

                for burst in class_i[::step]:
                    ax.plot(burst,rasterized=True)

            if y_lim:
                ax.set_ylim(y_lim)

            if corresponding_true_label in empty_true_clusters:
                empty_true_clusters.remove(corresponding_true_label)
    plt.savefig(savefile)
    plt.close()
    #outer_grid.tight_layout(fig)


def plot_eigenvalues(eigenvalues,true_cutoff=None,cutoff=None,eigenvalue_range=None, figsize = None, configuration = None):
    plt.close("all")
    plt.figure(figsize=figsize)
    if eigenvalue_range:
        plt.scatter(range(eigenvalue_range[0]+1,eigenvalue_range[1]+1), eigenvalues[eigenvalue_range[0]:eigenvalue_range[1]])
        plt.xlim(eigenvalue_range[0],eigenvalue_range[1]+1)
    else:
        plt.scatter(range(len(eigenvalues)), eigenvalues)

    if true_cutoff:
        plt.axhline(eigenvalues[true_cutoff - 1] + (eigenvalues[true_cutoff] - eigenvalues[true_cutoff - 1]) / 2, c="red", linestyle="--",
                    label="True Eigenvalue Gap: " + str(true_cutoff))
        plt.xticks([1] + list(plt.xticks()[0][1:]) + [true_cutoff])
    if cutoff:
        plt.axhline(eigenvalues[cutoff - 1] + (eigenvalues[cutoff] - eigenvalues[cutoff - 1]) / 2, c="blue", linestyle="--", label="Eigenvalue Gap: " + str(cutoff))
        plt.xticks([1] + list(plt.xticks()[0][1:]) + [cutoff])


    plt.xlabel("Index", fontsize=16, labelpad=10)
    plt.ylabel("Eigenvalue$_i$", fontsize=16,labelpad=10)
    if configuration:
        plt.title("Eigenvalues of Graph Laplacian in ascending order\n\n" + configuration, fontsize=18, pad=15)
    else:
        plt.title("Eigenvalues of Graph Laplacian in ascending order", fontsize=20, pad=20)
    plt.legend(fontsize=14)


#def plot_eigenvalue_gap_sizes




def plot_prediction_strength(k_predictions_strengths, cluster_sizes_per_k, color=sns.color_palette('Reds', 25, )[::-1], k_clusters=None, threshold=None, figsize = (30, 16), title = "Prediction Strength"):
    """Plot prediction strength per cluster found by clusterings with differnt number of clusters k
    Args:
        k_predictions_strengths (dict): dictionary containing the prediction strength for each cluster by clustering with k-clusters.
                                          key = k (number of clusters in clustering)
                                          value =   _clusters (prediction strenght of each individual cluster)
        cluster_sizes_per_k (dict): dictionary containing the size (number of bursts) for each cluster found by clustering with k-clusters.
                                          key = k (number of clusters in clustering)
                                          value = k_clusters (cluster size of each individual cluster)
        color (cm): color map

        k_clusters (list): list of keys for clusterings to plot the prediction strengths

        threshold (float): prediction strength threshold to mark in plot

    Returns:
        A plot of the prediction strength of each cluster found by clustering with k clusters. Point size indicates the size of each cluster in percent of the whole data used to calculate the prediction strength.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax_clusters = 1  # maximal number of k clusters with prediction strength to plot on x axis
    if not k_clusters:
        k_clusters = list(k_predictions_strengths.keys())  # if not specified plot PS for all clusterings in dict

    for k in k_clusters:
        clusters = range(1, k + 1)  # clusters in clustering_k
        ax_clusters = max(ax_clusters, k +1)  # update maximal number of clusters for plotting
        if len(k_predictions_strengths[k]) > 0:
            ax.plot(clusters, k_predictions_strengths[k], color=color[k], label=k)  # plot prediction strenght per cluster for clustering with k clusters
            ax.scatter(clusters, k_predictions_strengths[k], color=color[k], s=10000 * np.asarray(cluster_sizes_per_k[k])/ np.sum(cluster_sizes_per_k[k]))  # add points with size corresponding to percent of bursts falling into cluster (cluster size)
            ax.annotate(str(k), (clusters[k-1] - 0.1, k_predictions_strengths[k][-1] - 0.04),fontsize=12)  # annotate line with k used in clustering


    leg1 = ax.legend(fontsize=12, loc="lower left")  # add legend for lines

    color_index = len(color) // 2

    # add legend for point size indicating cluster size

    l1 = ax.scatter([], [], s=10000 * 0.01, edgecolors='none',color=color[color_index])  # point size corresponding to 1%
    l2 = ax.scatter([], [], s=10000 * 0.03, edgecolors='none', color=color[color_index])  # 3%
    l3 = ax.scatter([], [], s=10000 * 0.05, edgecolors='none', color=color[color_index])  # 5%
    l4 = ax.scatter([], [], s=10000 * 0.10, edgecolors='none', color=color[color_index])  # 10%
    l5 = ax.scatter([], [], s=10000 * 0.25, edgecolors='none', color=color[color_index])  # 25%
    l6 = ax.scatter([], [], s=10000 * 0.50, edgecolors='none', color=color[color_index])  # 50
    l7 = ax.scatter([], [], s=10000 * 0.75, edgecolors='none', color=color[color_index])  # 75%
    l8 = ax.scatter([], [], s=10000 * 0.90, edgecolors='none', color=color[color_index])  # 90%

    labels = ["1%", "3%", "5%", "10%", "25%", "50%", "75%", "90%"]
    leg2 = ax.legend([l1, l2, l3, l4, l5, l6, l7, l8], labels, ncol=4, title="Cluster Size (%) ", fontsize=15,
                     loc="upper right", borderpad=3, scatterpoints=1, columnspacing=2.5, labelspacing=5.5,
                     handletextpad=2.5)


    ax.set_title(title, fontsize=25)

    ax.set_xticks(range(1, ax_clusters + 1))
    ax.set_xlabel("Number of clusters", fontsize=14)
    ax.set_ylabel("Prediction Strength", fontsize=14)
    if threshold:
        ax.axhline(threshold)

    ax.set_xlim((1, ax_clusters + 2))
    ax.set_yticks(np.arange(0, 1.2, 0.1),)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[-1].label1.set_visible(False)

    ax.add_artist(leg1)
    # ax.set_ylim ((-19, 0.5))
    # ax.set_yticks(range(0, -19, -1))
    # ax.set_yticklabels(range(2,21))



def plot_mean_prediction_strengths(k_prediction_strengths,cluster_sizes_per_k,threshold, size_weighted=False,figsize = (50, 20), title=""):
    """

    k_prediction_strengths (dict): dictionary containing the prediction strength for each cluster by clustering with k-clusters.
                                          key = k (number of clusters in clustering)
                                          value =   _clusters (prediction strenght of each individual cluster)
    threshold (float): prediction strength threshold to mark in plot

    returns:
    """

    fig, ax = plt.subplots(figsize=figsize)

    k_clusters = list(k_prediction_strengths.keys())
    mean_prediction_strengths = []
    err_prediction_strengths = []

    for k in k_clusters:
        if size_weighted:
            mean_prediction_strengths.append(np.mean(np.asarray(k_prediction_strengths[k]) * np.asarray(cluster_sizes_per_k[k])/np.asarray(cluster_sizes_per_k[k])))
            err_prediction_strengths.append(np.std(np.asarray(k_prediction_strengths[k]) * np.asarray(cluster_sizes_per_k[k])/np.asarray(cluster_sizes_per_k[k])))
        else:
            mean_prediction_strengths.append(np.mean(k_prediction_strengths[k]))
            err_prediction_strengths.append(np.std(k_prediction_strengths[k]))

    ax.errorbar(k_clusters, mean_prediction_strengths, yerr=err_prediction_strengths)

    for i, k in enumerate(k_clusters):
        ax.annotate("%.2f" % (mean_prediction_strengths[i]), (k + 0.01, mean_prediction_strengths[i] + 0.01))

    if threshold:
        ax.axhline(threshold, color="red", label="Threshold")
        plt.yticks(list(plt.yticks()[0]) + [threshold])

    if title:
        ax.set_title(title, fontsize=25, pad=10)
    else:
        if size_weighted:
            title = "Mean Prediction Strength for Clustering with k Clusters weighted by Cluster Size"
        else:
            title = "Mean Prediction Strength for Clustering with k Clusters"
        ax.set_title(title, fontsize=25, pad=10)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=14)
    ax.set_ylabel("prediction strength", fontsize=14)
    ax.set_ylim((0, 1.1))
    #ax.legend(fontsize = 14)


def plot_number_bursts_in_low_clusters_per_k(k_low_cluster_sizes, n_total,threshold, plot_proportion = False,plot_mean=False,figsize = (30, 16)):
    """ Plot proportion of bursts falling into clusters with low prediction strength (below a threshold):
    Args:
        k_low_cluster_sizes (dict): dictionary containing bursts falling into cluster with low prediction strenght for clustering with k clusters.
                                    key = k (number of clusters in clustering)
                                    value = k_clusters (cluster size of each individual cluster)
        n_total (int): total number of bursts clustered
        plot_proportion (bool): boolean for plotting the mean proportion of bursts
        plot_mean (bool): boolean for plotting the mean or the total of all clusters in k clustering (mean over k-clusters)

    Returns:
        For each clustering plot of the average proportion of bursts falling into clusters with low prediction strength (below a threshold).
    """

    fig, ax = plt.subplots(figsize=figsize)
    k_clusters = list(k_low_cluster_sizes.keys())
    k_bad_bursts = []
    k_bad_bursts_err = []
    for k in k_clusters:
        if plot_mean:
            bad_bursts = np.mean(k_low_cluster_sizes[k])
            bad_bursts_err = np.std(k_low_cluster_sizes[k])
        else:
            bad_bursts = np.sum(k_low_cluster_sizes[k])
            bad_bursts_err = 0
        if plot_proportion:
            bad_bursts = bad_bursts/n_total * 100
            bad_bursts_err = bad_bursts_err/n_total * 100
        k_bad_bursts.append(bad_bursts)
        k_bad_bursts_err.append(bad_bursts_err)

    if plot_mean:
        ax.errorbar(k_clusters, k_bad_bursts, yerr=k_bad_bursts_err, marker="o", markersize=15)
    else:
        ax.plot(k_clusters, k_bad_bursts, marker="o", markersize=15)

    for i, p in enumerate(k_bad_bursts):
        if plot_proportion:
            ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] + 0.1, k_bad_bursts[i] + 0.1),)
        else:
            ax.annotate("#%d%" % int(p), (k_clusters[i] + 0.1, k_bad_bursts[i] + 0.1),)

    title = ""
    if plot_mean:
        title += "Mean "
    if plot_proportion:
        title += "Proportion of Bursts in Clusters below prediction Strength threshold= %.2f" % threshold
    else:
        if plot_mean:
            title += "Number of Bursts in Clusters below prediction Strength threshold= %.2f" % threshold
        else:
            title += "Total Number of Bursts in Clusters below prediction Strength threshold= %.2f" % threshold

    ax.set_title(title, fontsize=40)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=14)
    if plot_proportion:
        ax.set_ylabel("% Bursts", fontsize=14)
    else:
        ax.set_ylabel("# Bursts", fontsize=14)
    ax.set_xlim((k_clusters[0], k_clusters[-1] + 2))


def plot_number_burst_with_low_prediction_strength_per_k(k_low_individual_per_cluster,n_total,threshold, plot_proportion=False,plot_mean=False, figsize=(30, 16)):
    fig, ax = plt.subplots(figsize=figsize)

    k_clusters = list(k_low_individual_per_cluster.keys())

    k_low_bursts = []
    k_low_bursts_err = []

    for k in k_clusters:
        if plot_mean:
            low_bursts = np.mean(k_low_individual_per_cluster[k])
            k_low_bursts_err = np.std(k_low_individual_per_cluster[k])
        else:
            low_bursts = np.sum(k_low_individual_per_cluster[k])
            k_low_bursts_err = 0
        if plot_proportion:
            low_bursts = low_bursts/n_total * 100
            k_low_bursts_err = k_low_bursts_err/n_total * 100

        k_low_bursts.append(low_bursts)
        k_low_bursts_err.append(k_low_bursts_err)


    if plot_mean:
        ax.errorbar(k_clusters, k_low_bursts, yerr=k_low_bursts_err, marker="o", markersize=15)
    else:
        ax.plot(k_clusters, k_low_bursts, marker="o", markersize=15)

    for i, p in enumerate(k_low_bursts):
        if plot_proportion:
            ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] + 0.1, k_low_bursts[i] + 0.1), fontsize=15)
        else:
            ax.annotate("#%d%" % int(p), (k_clusters[i] + 0.1, k_low_bursts[i] + 0.1), fontsize=15)

    title = ""
    if plot_mean:
        title += "Mean "
    if plot_proportion:
        title += "Proportion of Bursts with individual Prediction Strength below threshold= %.2f" % threshold
    else:
        if plot_mean:
            title += "Number of Bursts with individual Prediction Strength below threshold= %.2f" % threshold
        else:
            title += "Total Number of Bursts with individual Prediction Strength below threshold= %.2f" % threshold

    ax.set_title(title, fontsize=40)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=14)
    if plot_proportion:
        ax.set_ylabel("% Bursts", fontsize=14)
    else:
        ax.set_ylabel("# Bursts", fontsize=14)
    ax.set_xlim((k_clusters[0], k_clusters[-1] + 2))


def plot_parameter_space(parameter_df, color_dict = None, time_condition_labels_dict = None):
    condition_groups = parameter_df.groupby(['amplitude_condition', 'time_constant_condition'])

    amplitude_conditions = np.unique(parameter_df.amplitude_condition)
    time_constant_conditions = np.unique(parameter_df.time_constant_condition)


    if color_dict:
        print("Use provided color Pallets!")
    else:
        color1 = [(151, 218, 114), (121, 207, 74), (95, 181, 48), (74, 141, 37), (53, 101, 27)]  # greens
        color2 = [(193, 183, 139), (175, 162, 105), (150, 137, 80), (116, 106, 62), (83, 76, 44)]  # greybrowns
        color3 = [(118, 210, 213), (79, 197, 202), (53, 172, 176), (42, 133, 137), (30, 95, 98)]  # türkis
        color4 = [(233, 98, 157), (227, 53, 129), (202, 28, 103), (157, 22, 80), (112, 15, 57)]  # magenta
        color5 = [(146, 97, 234), (115, 52, 229), (89, 26, 203), (69, 21, 158), (50, 15, 113)]  # purple
        color6 = [(86, 124, 246), (37, 86, 243), (12, 61, 218), (9, 47, 169), (6, 34, 121)]  # blues
        color7 = [(166, 166, 166), (140, 140, 140), (115, 115, 115), (89, 89, 89), (64, 64, 64)]  # greys
        color8 = [(233, 248, 84), (226, 246, 35), (201, 220, 9), (156, 171, 7), (112, 122, 5)]  # yellow green
        color9 = [(255, 192, 77), (255, 174, 26), (230, 149, 0), (179, 115, 0), (128, 83, 0)]  # orange

        color_pallets = [color1,color2,color3,color4,color5,color6,color7,color8,color9]
        color_dict = {}
        for i,key in enumerate(time_constant_conditions):
            color_dict[key] = color_pallets[i]

    if time_condition_labels_dict:
        print("Use provided label for time conditions!")
    else:
        time_condition_labels_dict = {"equal_sharp": "tau1≈tau2 (large)", "equal_medium": "tau1≈tau2 (medium)", "equal_wide":"tau1≈tau2 (small)", "wide_sharp_negative_skew": "tau1<<tau2 (small<<large)", "wide_medium_negative_skew": "tau1<tau2 (small<<medium)", "medium_sharp_negative_skew": "tau1<tau2 (medium<large)", "sharp_wide_positive_skew":"tau1>>tau2 (large>>small)", "medium_wide_positive_skew":"tau1>tau2 (medium>small)", "sharp_medium_positive_skew":"tau1>tau2 (large>medium)"}

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for name, group in condition_groups:
        color_palette = color_dict[name[0]]
        amplitude_idx = np.where(np.asarray(amplitude_conditions) == name[1])[0][0]
        color = color_palette[amplitude_idx]
        color = np.array(color)/255
        if amplitude_idx == 3:
            ax.scatter(np.log(group['Tau1']), np.log(group['Tau2']), group['Lambda'], label=time_condition_labels_dict[name[0]], color = color)
        else:
            ax.scatter(np.log(group['Tau1']), np.log(group['Tau2']), group['Lambda'], color=color)

    handles, labels = ax.get_legend_handles_labels()
    sorted_handles = []
    for i in list(time_condition_labels_dict.values()):
        np.where(np.asarray(labels) == i)
        sorted_handles.append(handles[np.where(np.asarray(labels) == i)[0][0]])
    ax.legend(sorted_handles,time_condition_labels_dict.values(),ncol=3)
    ax.set_xlabel("Log(tau1)")
    ax.set_ylabel("Log(tau2)")
    ax.set_zlabel("Amplitude")

