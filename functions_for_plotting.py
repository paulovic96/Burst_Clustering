import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import textwrap

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




def plot_clusters(data, reference_labels,labels_k,rows,columns, layout_label_mapping, figsize = (30,20), reference_clustering="True", percent_true_cluster = False, scores=None, n_bursts=None, y_lim = None, plot_mean = False, title = None, subplot_adjustments = [0.05,0.95,0.03,0.9,0.4, 0.15], save_file = "test.pdf"):

    corresponding_true_class_for_prediction, highest_overlap_class_for_prediction = get_true_label_mapping(reference_labels, labels_k)
    splitted_classes, split_count, new_clusters_splitted = get_splitted_clusters(highest_overlap_class_for_prediction)
    merged_classes, new_clusters_merged = get_merged_clusters(corresponding_true_class_for_prediction)
    position_count_for_splitted_classes = np.zeros(len(split_count))
    empty_true_clusters = list(np.unique(reference_labels))

    if reference_labels is None:
        reference_labels = labels_k

    normal_layout = np.arange(rows*columns).reshape((rows,columns))

    if not layout_label_mapping:
        layout_label_mapping = {}
        for i in np.unique(reference_labels):
            layout_label_mapping[i] = i

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
        layout_label = layout_label_mapping[corresponding_true_label]
        corresponding_column = int(np.where(normal_layout == layout_label)[1][0]) # column position
        corresponding_row = int(np.where(normal_layout == layout_label)[0][0])  # row position

        if corresponding_true_label in splitted_classes:
            true_class_size = len(data[np.where(reference_labels == corresponding_true_label)])
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
                                    "\n" +reference_clustering+" Cluster: " + "\n".join(textwrap.wrap(', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_clusters]), 25)) +
                                    "\n%%: " + "\n".join(textwrap.wrap(', '.join(['[' + ', '.join(['%.1f'] * len(x)) +']' for x in true_cluster_percents]), 25))) % tuple(new_clusters_splitted[corresponding_true_label] + sum(true_clusters,[]) + sum(true_cluster_percents,[]))
                else:
                    if "Score" in reference_clustering:
                        cluster_title = ("Cluster [" + ', '.join(['%d'] * len(new_clusters_splitted[corresponding_true_label])) + "]" +
                                         "\n" + reference_clustering + ": " + "[" + ', '.join(['%.2f'] * len(new_clusters_splitted[corresponding_true_label])) + "]" +
                                        "\n#:  " + "\n".join(textwrap.wrap(', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_cluster_counts]), 25))) % tuple(
                            new_clusters_splitted[corresponding_true_label] + [scores[k] for k in new_clusters_splitted[corresponding_true_label]] + sum(true_cluster_counts,[]))
                    else:
                        cluster_title = ("Cluster [" + ', '.join(['%d'] * len(new_clusters_splitted[corresponding_true_label])) + "]" +
                                        "\n" + reference_clustering+" Cluster: " + "\n".join(textwrap.wrap(', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_clusters]), 25)) +
                                        "\n#:  " + "\n".join(textwrap.wrap(', '.join(['[' + ', '.join(['%d'] * len(x)) +']' for x in true_cluster_counts]), 25))) % tuple(new_clusters_splitted[corresponding_true_label] + sum(true_clusters,[]) + sum(true_cluster_counts,[]))

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
                    if merged_class in empty_true_clusters and merged_class not in splitted_classes:
                        layout_label_i = layout_label_mapping[merged_class]
                        row_i = int(np.where(normal_layout == layout_label_i)[0][0])
                        column_i =  int(np.where(normal_layout == layout_label_i)[1][0])
                        ax = fig.add_subplot(outer_grid[row_i:(row_i + 1), column_i])
                        ax.plot()
                        ax.set_title(reference_clustering+ " Cluster: [%d]" % merged_class,loc="left", fontsize = 12)
                        if y_lim:
                            ax.set_ylim(y_lim)

            ax = fig.add_subplot(outer_grid[row_start:row_end, corresponding_column])

            true_clusters = corresponding_true_class_for_prediction[i][0]
            true_cluster_counts = corresponding_true_class_for_prediction[i][1]
            true_class_size = [len(data[np.where(reference_labels == true_cluster)[0]]) for i,true_cluster in enumerate(true_clusters)]
            true_cluster_percents = [true_cluster_counts[i]/true_class_size[i] for i,true_cluster in enumerate(true_clusters)]

            if len(true_clusters) > 1:
                if percent_true_cluster:
                    cluster_title = ("Cluster %i "
                                     "\n"+reference_clustering+" Cluster: [" + "\n".join(textwrap.wrap(', '.join(['%d'] * len(true_clusters)) + "]", 25)) +
                                     "\n%%: [" + "\n".join(textwrap.wrap(', '.join(['%.1f'] * len(true_cluster_percents)) + "]", 25))) % tuple([i] + list(true_clusters) + list(np.round(np.asarray(true_cluster_percents) * 100, decimals=1)))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
                                     #"\n%%: [" + ', '.join(['%d/%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + sum(list(map(list, zip(true_cluster_counts, true_class_size))),[]))
                else:
                    if "Score" in reference_clustering:
                        cluster_title = ("Cluster %i "
                                         "\n" + reference_clustering + ": " + "[%.2f]"+
                                         "\n#: [" + "\n".join(textwrap.wrap(', '.join(['%d'] * len(true_cluster_counts)) + "]", 25))) % tuple([i] + [scores[i]] + list(true_cluster_counts))
                    else:
                        cluster_title = ("Cluster %i "
                                         "\n"+reference_clustering+" Cluster: [" + "\n".join(textwrap.wrap(', '.join(['%d'] * len(true_clusters)) + "]", 25)) +
                                         "\n#: [" + "\n".join(textwrap.wrap(', '.join(['%d'] * len(true_cluster_counts)) + "]", 25))) % tuple([i] + list(true_clusters) + list(true_cluster_counts))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
            else:
                if percent_true_cluster:
                    cluster_title = ("Cluster %i "
                                     "\n"+reference_clustering+" Cluster: [" + ', '.join(['%d'] * len(true_clusters)) + "] " +
                                     "%%: [" + ', '.join(['%.1f'] * len(true_cluster_percents)) + "]") % tuple([i] + list(true_clusters) + list(np.round(np.asarray(true_cluster_percents) * 100, decimals=1)) ) # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
                        #"%%: [" + ', '.join(['%d/%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + sum(list(map(list, zip(true_cluster_counts, true_class_size))), []))

                else:
                    if "Score" in reference_clustering:
                        cluster_title = ("Cluster %i \n" + reference_clustering + ": " + "[%.2f]" +
                                         " #: [" + ', '.join(['%d'] * len(true_cluster_counts)) + "]"
                                         ) % tuple([i] + [scores[i]] + list(true_cluster_counts))
                    else:
                        cluster_title = ("Cluster %i \n"+reference_clustering+" Cluster: [" + ', '.join(
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


    for empty_true_cluster in empty_true_clusters:
        layout_label_i = layout_label_mapping[empty_true_cluster]
        row_i = int(np.where(normal_layout == layout_label_i)[0][0])
        column_i = int(np.where(normal_layout == layout_label_i)[1][0])
        ax = fig.add_subplot(outer_grid[row_i:(row_i + 1), column_i])
        ax.set_title(reference_clustering+" Cluster: [%d]" % empty_true_cluster, loc="left", fontsize=12)
        ax.plot()
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.savefig(save_file)
    plt.close()
    #outer_grid.tight_layout(fig)


def plot_eigenvalues(eigenvalues,true_cutoff=None,cutoff=None,eigenvalue_range=None, figsize = (20,10), configuration = None, save_file="test.pdf"):
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

    plt.savefig(save_file)
    plt.close()

#def plot_eigenvalue_gap_sizes




def plot_prediction_strength(k_predictions_strengths, cluster_sizes_per_k,strength_sorted=True,color=sns.color_palette('Reds', 25, )[::-1], k_clusters=None, threshold=None, figsize=(20,8.5), title = "Prediction Strength",plot_adjustments = [0.05,0.08,0.95, 0.91], save_file = "test.pdf"):
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
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)
    ax_clusters = 1  # maximal number of k clusters with prediction strength to plot on x axis
    if not k_clusters:
        k_clusters = list(k_predictions_strengths.keys())  # if not specified plot PS for all clusterings in dict

    for k in k_clusters:
        clusters = range(1, k + 1)  # clusters in clustering_k
        ax_clusters = max(ax_clusters, k +1)  # update maximal number of clusters for plotting


        if len(k_predictions_strengths[k]) > 0:
            prediction_strength = k_predictions_strengths[k]
            cluster_sizes = cluster_sizes_per_k[k]

            if strength_sorted:
                strength_and_size = list(zip(np.round(prediction_strength, 2), cluster_sizes))
                strength_and_size = sorted(strength_and_size, key=lambda e: (e[0], e[1]), reverse=True)

                prediction_strength = np.asarray(strength_and_size)[:, 0]
                cluster_sizes = np.asarray(strength_and_size)[:, 1]


            ax.plot(clusters, prediction_strength, color=color[k], label=k, zorder = 0)  # plot prediction strenght per cluster for clustering with k clusters
            ax.scatter(clusters, prediction_strength, color=color[k], s=10000 * np.asarray(cluster_sizes)/ np.sum(cluster_sizes),zorder = 1)  # add points with size corresponding to percent of bursts falling into cluster (cluster size)
            ax.annotate(str(k), (clusters[k-1] - 0.1, prediction_strength[-1] - 0.04),fontsize=10,zorder = 3)  # annotate line with k used in clustering
            ax.scatter(clusters, prediction_strength,facecolors='red', edgecolors='black',zorder = 2)

    leg1 = ax.legend(fontsize=12, ncol=int(len(k_clusters)/2), loc="lower left")  # add legend for lines

    color_index = len(color) // 2

    # add legend for point size indicating cluster size

    l1 = ax.scatter([], [], s=10000 * 0.01, edgecolors='none',color=color[color_index])  # point size corresponding to 1%
    #l2 = ax.scatter([], [], s=10000 * 0.03, edgecolors='none', color=color[color_index])  # 3%
    l3 = ax.scatter([], [], s=10000 * 0.05, edgecolors='none', color=color[color_index])  # 5%
    l4 = ax.scatter([], [], s=10000 * 0.10, edgecolors='none', color=color[color_index])  # 10%
    l5 = ax.scatter([], [], s=10000 * 0.25, edgecolors='none', color=color[color_index])  # 25%
    l6 = ax.scatter([], [], s=10000 * 0.50, edgecolors='none', color=color[color_index])  # 50
    #l7 = ax.scatter([], [], s=10000 * 0.75, edgecolors='none', color=color[color_index])  # 75%
    l8 = ax.scatter([], [], s=10000 * 0.90, edgecolors='none', color=color[color_index])  # 90%

    labels = ["1%", "5%", "10%", "25%", "50%", "90%"]
    leg2 = ax.legend([l1, l3, l4], labels[0:3], ncol=1, title="Cluster Size (%) ", fontsize=12,
                     loc = "upper right", borderpad=1, scatterpoints=1, columnspacing=0, labelspacing=2,
                     handletextpad=3.5,frameon=False)
    plt.setp(leg2.get_title(), fontsize=15)
    leg3 = ax.legend([l5, l6, l8], labels[3:], ncol=1, title="", fontsize=12,
                    loc = "lower right", borderpad=4, scatterpoints=1, columnspacing=0,labelspacing=6,
                     handletextpad=3.5,frameon=False)

    bb2 = leg2.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb3 = leg3.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

    # Change to location of the legend.

    bb2.x0 += 0.075#0.065
    bb2.x1 += 0.075#0.065
    bb3.x0 += 0.095#0.085
    bb3.x1 += 0.095#0.085

    bb2.y0 -= -0.08#0.07
    bb2.y1 -= -0.08#0.07
    bb3.y0 += 0.39#0.14
    bb3.y1 += 0.39#0.14

    leg2.set_bbox_to_anchor(bb2, transform=ax.transAxes)
    leg3.set_bbox_to_anchor(bb3, transform=ax.transAxes)


    ax.set_title(title, fontsize=22, pad=10)

    ax.set_xticks(range(1, ax_clusters + 1))
    ax.set_xlabel("Number of clusters", fontsize=14,labelpad=10)
    ax.set_ylabel("Prediction Strength", fontsize=14,labelpad=10)
    if threshold:
        ax.axhline(threshold)

    ax.set_xlim((1, ax_clusters + 2))
    ax.set_yticks(np.arange(0, 1.2, 0.1),)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[-1].label1.set_visible(False)

    ax.add_artist(leg1)
    ax.add_artist(leg2)
    # ax.set_ylim ((-19, 0.5))
    # ax.set_yticks(range(0, -19, -1))
    # ax.set_yticklabels(range(2,21))
    left = plot_adjustments[0]
    bottom = plot_adjustments[1]
    right = plot_adjustments[2]
    top = plot_adjustments[3]

    plt.subplots_adjust(left, bottom, right, top)
    ax.spines["right"].set_visible(False)

    plt.savefig(save_file)
    plt.close()


def plot_mean_prediction_strengths(k_prediction_strengths,threshold=None,k_clusters=None,plot_min = True,figsize=(20,10), title="",plot_adjustments = [0.05,0.08,0.95, 0.91], save_file = "test.pdf"):
    """
    k_prediction_strengths (dict): dictionary containing the prediction strength for each cluster by clustering with k-clusters.
                                          key = k (number of clusters in clustering)
                                          value =   _clusters (prediction strength of each individual cluster)
    threshold (float): prediction strength threshold to mark in plot

    returns:
    """
    plt.close("all")

    fig, ax = plt.subplots(figsize=figsize)

    if not k_clusters:
        k_clusters = list(k_prediction_strengths.keys())

    mean_prediction_strengths = []
    err_prediction_strengths = []
    min_prediction_strengths = []

    for k in k_clusters:
        mean_prediction_strengths.append(np.mean(k_prediction_strengths[k]))
        err_prediction_strengths.append(np.std(k_prediction_strengths[k]))
        min_prediction_strengths.append(np.amin(k_prediction_strengths[k]))


    upper_err = np.asarray(err_prediction_strengths) - np.maximum(0,(np.asarray(err_prediction_strengths)+np.asarray(mean_prediction_strengths)-1))
    lower_err = np.asarray(err_prediction_strengths)
    err = np.stack((lower_err,upper_err), axis=0)

    ax.plot(k_clusters, mean_prediction_strengths, "o",label="Mean PS")
    ax.errorbar(k_clusters, mean_prediction_strengths, yerr=err,fmt='-',ecolor='lightgray', elinewidth=3,)
    if plot_min:
        ax.plot(k_clusters, min_prediction_strengths, "o-", color = "C01", label="Min PS")
        for i, k in enumerate(k_clusters):
            if i % 2 == 0:
                ax.annotate("%.2f" % (min_prediction_strengths[i]), (k - 0.25, 0.1), fontsize=12)
            else:
                ax.annotate("%.2f" % (min_prediction_strengths[i]), (k - 0.25, 0.05), fontsize=12)

    for i, k in enumerate(k_clusters):
        if i%2==0:
            ax.annotate("%.2f" % (mean_prediction_strengths[i]), (k-0.25, 1.05),fontsize = 12)
        else:
            ax.annotate("%.2f" % (mean_prediction_strengths[i]), (k - 0.25, 1.0), fontsize=12)

    if threshold:
        ax.axhline(threshold, color="C03", label="Threshold")
        plt.yticks(list(plt.yticks()[0]) + [threshold])

    if title:
        ax.set_title(title, fontsize=22, pad=20)
    else:
        title = "Mean Prediction Strength for Clustering with k Clusters"
        ax.set_title(title, fontsize=22, pad=20)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=16, labelpad=10)
    ax.set_ylabel("Mean Prediction Strength", fontsize=16, labelpad=10),
    ax.set_ylim((0, 1.1))

    ax.set_yticks(np.arange(0, 1.1,0.1))
    left = plot_adjustments[0]
    bottom = plot_adjustments[1]
    right = plot_adjustments[2]
    top = plot_adjustments[3]

    plt.subplots_adjust(left,bottom,right, top)

    ax.legend(fontsize = 14, loc="upper right",bbox_to_anchor=(1, 1.1))
    plt.savefig(save_file)
    plt.close()


def plot_number_bursts_in_low_clusters_per_k(k_low_cluster_sizes, n_total,threshold, k_clusters = None, plot_proportion = False,figsize=(20,10), plot_adjustments = [0.075,0.08,0.95, 0.93], save_file = "test.pdf"):
    """ Plot proportion of bursts falling into clusters with low prediction strength (below a threshold):
    Args:
        k_low_cluster_sizes (dict): dictionary containing bursts falling into cluster with low prediction strenght for clustering with k clusters.
                                    key = k (number of clusters in clustering)
                                    value = k_clusters (cluster size of each individual cluster)
        n_total (int): total number of bursts clustered
        plot_proportion (bool): boolean for plotting the mean proportion of bursts

    Returns:
        For each clustering plot of the average proportion of bursts falling into clusters with low prediction strength (below a threshold).
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    if not k_clusters:
        k_clusters = list(k_low_cluster_sizes.keys())

    k_bad_bursts = []
    for k in k_clusters:
        bad_bursts = np.sum(k_low_cluster_sizes[k])
        if plot_proportion:
            bad_bursts = bad_bursts/n_total * 100
        k_bad_bursts.append(bad_bursts)


    ax.plot(k_clusters, k_bad_bursts, marker="o", linestyle='dashed', markersize=15,color = "C0", zorder=-1)


    for i, p in enumerate(k_bad_bursts):
        if plot_proportion:
            if i%2 == 0:
                ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] - 0.5, k_bad_bursts[i] + 100*0.04),color = "C0",zorder = 1)
            else:
                ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] - 0.5, k_bad_bursts[i] - 100 * 0.04),color="C0", zorder=1)
        else:
            if i%2==0:
                ax.annotate("#%d" % int(p), (k_clusters[i] - 0.4, k_bad_bursts[i] + n_total*0.033),color="C0",zorder = 1) #0.8
            else:
                ax.annotate("#%d" % int(p), (k_clusters[i] - 0.4, k_bad_bursts[i] + n_total * 0.033), color="C0",zorder= 1) #- #0.8

    title = ""
    if plot_proportion:
        title += "Proportion of bursts in clusters below prediction strength threshold= %.2f" % threshold
    else:
        title += "Total number of bursts in clusters below prediction strength threshold= %.2f" % threshold

    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=16, labelpad=10)
    if plot_proportion:
        ax.set_ylabel("% Bursts", fontsize=16,color="C0",labelpad=10)
        ax.set_ylim((0,100))
    else:
        ax.set_ylabel("# Bursts", fontsize=16,color="C0",labelpad=10)
        ax.set_ylim((0,n_total+n_total*0.1))
    ax.set_xlim((k_clusters[0], k_clusters[-1] + 1))
    ax.set_yticks(np.arange(0,n_total+n_total*0.1,1000))
    ax.tick_params(axis='y', labelcolor="C0",labelsize=12)
    ax.tick_params(axis="x", labelsize=12)

    ax2 = ax.twinx()
    ax2.set_ylabel('#Clusters below threshold', fontsize=16,color="C3",labelpad=10)
    n_low_clusters = [len(k_low_cluster_sizes[k]) for k in k_clusters]
    ax2.plot(k_clusters, n_low_clusters,marker="v", linestyle='dashed', markersize=15, color = "C3")
    ax2.tick_params(axis='y', labelcolor="C3",labelsize=12)
    ax2.set_ylim((0, max(n_low_clusters) +1))
    ax2.set_yticks(np.arange(0,  max(n_low_clusters) +1, 1))

    left = plot_adjustments[0]
    bottom = plot_adjustments[1]
    right = plot_adjustments[2]
    top = plot_adjustments[3]
    plt.subplots_adjust(left, bottom, right, top)

    plt.savefig(save_file)
    plt.close()


def plot_number_burst_with_low_prediction_strength_per_k(k_low_individual_per_cluster,k_low_individual_ps_per_cluster,n_total,threshold, k_clusters=None,plot_proportion=False,plot_mean_low_ps = True,figsize=(20,10),plot_adjustments = [0.075,0.08,0.94, 0.93], save_file = "test.pdf"):
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    if not k_clusters:
        k_clusters = list(k_low_individual_per_cluster.keys())

    k_low_bursts = []

    for k in k_clusters:
        low_bursts = np.sum(k_low_individual_per_cluster[k])
        k_low_bursts_err = 0
        if plot_proportion:
            low_bursts = low_bursts/n_total * 100

        k_low_bursts.append(low_bursts)



    ax.plot(k_clusters, k_low_bursts, marker="o", linestyle='dashed', markersize=15,color = "C0",zorder = 0)

    for i, p in enumerate(k_low_bursts):
        if plot_proportion:
            if i % 2 == 0:
                ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] - 0.5, k_low_bursts[i] + 100 * 0.04), color="C0", zorder=1)
            else:
                ax.annotate("%.2f%%" % np.around(p, decimals=2), (k_clusters[i] - 0.5, k_low_bursts[i] - 100 * 0.04),color="C0", zorder=1)
        else:
            if i % 2 == 0:
                ax.annotate("#%d" % int(p), (k_clusters[i] - 0.4, k_low_bursts[i] + n_total * 0.033), color="C0",zorder=1) #0.8
            else:
                ax.annotate("#%d" % int(p), (k_clusters[i] - 0.4, k_low_bursts[i] + n_total * 0.033), color="C0",zorder=1) #0.8 -


    title = ""
    if plot_proportion:
        title += "Proportion of Bursts with individual Prediction Strength below threshold= %.2f" % threshold
    else:
        title += "Total Number of Bursts with individual Prediction Strength below threshold= %.2f" % threshold

    ax.set_title(title, fontsize=22, pad=20)
    ax.set_xticks(k_clusters)
    ax.set_xlabel("Number of clusters", fontsize=16,labelpad=10)
    if plot_proportion:
        ax.set_ylabel("% Bursts",color = "C0", fontsize=16,labelpad=10)
        ax.set_ylim((0, 110))
    else:
        ax.set_ylabel("# Bursts",color = "C0", fontsize=16,labelpad=10)
        ax.set_ylim((0,n_total+n_total*0.1))
    ax.set_xlim((k_clusters[0], k_clusters[-1] + 1))
    ax.set_yticks(np.arange(0, n_total + n_total * 0.1, 1000))
    ax.tick_params(axis='y', labelcolor="C0",labelsize=12)
    ax.tick_params(axis="x", labelsize=12)

    if plot_mean_low_ps:
        ax2 = ax.twinx()
        ax2.set_ylabel('Mean PS for bursts below threshold', fontsize=16, color="C3",labelpad=10)
        mean_ps = [np.nan_to_num(np.mean(np.asarray(k_low_individual_ps_per_cluster[k]))) for k in k_clusters]
        ax2.plot(k_clusters, mean_ps, marker="v", linestyle='dashed', markersize=15, color="C3",zorder = 1)
        for i,p in enumerate(mean_ps):
            if i%2 ==0:
                ax2.annotate("%.2f" % np.around(p, decimals=2), (k_clusters[i] - 0.3, mean_ps[i] + 1 * 0.04), color="C3", zorder = 2) #0.4
            else:
                ax2.annotate("%.2f" % np.around(p, decimals=2), (k_clusters[i] - 0.3, mean_ps[i] + 1 * 0.04), color="C3", zorder=2) #0.4 -

        ax2.tick_params(axis='y', labelcolor="C3", labelsize=12)
        ax2.set_ylim((0, 1))
    else:
        ax2 = ax.twinx()
        ax2.set_ylabel('#Clusters with bursts below threshold', fontsize=16, color="C3",labelpad=10)
        n_low_clusters = [np.sum(np.asarray(k_low_individual_per_cluster[k])>0) for k in k_clusters]
        ax2.plot(k_clusters, n_low_clusters, marker="v", linestyle='dashed', markersize=15, color="C3")
        ax2.tick_params(axis='y', labelcolor="C3",labelsize=12)
        ax2.set_ylim((0, max(n_low_clusters) + 1))


    left = plot_adjustments[0]
    bottom = plot_adjustments[1]
    right = plot_adjustments[2]
    top = plot_adjustments[3]
    plt.subplots_adjust(left, bottom, right, top)

    plt.savefig(save_file)
    plt.close()

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


def grouped_heatmap_plot(shape,
                         within_distances_mms_dict,
                         between_distances_mms_dict,
                         max_value,
                         row_conditions,
                         col_conditions,
                         row_conditions_occurance_in_data_interval,
                         col_conditions_occurance_in_data_interval,
                         condition = "mean",
                         normalized = True,
                         clusters_row=None,
                         clusters_col=None,
                         title="",
                         figsize=(20, 20),
                         subplotadjustments=None):
    if clusters_row is None:
        clusters_row = list(range(len(within_distances_mms_dict.keys())))
    if clusters_col is None:
        clusters_col = list(range(len(within_distances_mms_dict.keys())))

    mean_distance_matrix = np.zeros((shape[0], shape[1]))

    # cluster_positions = list(range(n_clusters))

    if condition == "mean":
        condition_index = 0
    elif condition == "median":
        condition_index = 1
    elif condition == "std":
        condition_index = 2
    else:
        print("please provide a valid condition: ['mean', 'median', 'std'] !!!")
        return

    for c, cluster in enumerate(clusters_row):
        for i, j in enumerate(clusters_col):
            if cluster == j:
                mean_distance_matrix[c, i] = within_distances_mms_dict[cluster][condition_index]
            else:
                mean_distance_matrix[c, i] = between_distances_mms_dict[cluster][j][condition_index]

        # other_clusters = clusters.copy()
        # other_cluster_positions = cluster_positions.copy()
        # other_clusters.remove(cluster)
        # other_cluster_positions.remove(c)
        # for i,j in enumerate(other_clusters):
        # mean_distance_matrix[c,other_cluster_positions[i]] = between_distances_mms_dict[cluster][j][1]

    if normalized:
        mean_distance_matrix = mean_distance_matrix / np.round(max_value)
        vmax = 1
    else:
        vmax = np.round(max_value)

    rows = len(row_conditions)
    cols = len(col_conditions)

    fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)

    cmap = plt.cm.magma
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, vmin=0, vmax=vmax, annot_kws={"fontsize": 15})

    row_labels = row_conditions  # ["Small Amplitude", "Medium Amplitude", "Large Amplitude"]
    col_labels = col_conditions  # ["Small Amplitude", "Medium Amplitude", "Large Amplitude"]

    cluster_in_rows = sum([list(range(i, shape[0], row_conditions_occurance_in_data_interval)) for i in
                           range(row_conditions_occurance_in_data_interval)], [])
    # print(cluster_in_rows)

    for row in range(rows):
        cluster_in_columns = sum([list(range(i, shape[1], col_conditions_occurance_in_data_interval)) for i in
                                  range(col_conditions_occurance_in_data_interval)], [])
        # print(cluster_in_columns)

        data_rows = cluster_in_rows[:int(shape[0] / rows)]
        for i in data_rows:
            cluster_in_rows.remove(i)

        for col in range(cols):
            data_cols = cluster_in_columns[:int(shape[1] / cols)]
            for i in data_cols:
                cluster_in_columns.remove(i)

            yticklabels = col == 0
            xticklabels = (row == 0 or row == len(row_conditions) - 1)

            sns.heatmap(mean_distance_matrix[np.ix_(data_rows, data_cols)], ax=ax[row, col], annot=True, fmt=".2f",
                        **heatmapkws, yticklabels=yticklabels, xticklabels=xticklabels)

            if col == 0:
                ax[row, col].set_yticklabels(np.asarray(clusters_row)[data_rows])
                ax[row, col].set_ylabel(row_labels[row], fontsize=24, labelpad=20)
                ax[row, col].tick_params(axis='y', labelsize=18)

            if row == 0 or row == len(row_conditions) - 1:
                ax[row, col].set_xticklabels(np.asarray(clusters_col)[data_cols])
                ax[row, col].set_xlabel(col_labels[col], fontsize=24, labelpad=20)
                if row == 0:
                    ax[row, col].xaxis.tick_top()
                    ax[row, col].xaxis.set_label_position('top')

                ax[row, col].tick_params(axis="x", labelsize=18)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    cax = fig.add_axes([0.93, 0.1, 0.03, 0.8])

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=24)
    if not subplotadjustments is None:
        left = subplotadjustments[0]
        right = subplotadjustments[1]
        bottom = subplotadjustments[2]
        top = subplotadjustments[3]
        wspace = subplotadjustments[4]
        hspace = subplotadjustments[5]

        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    else:
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig.suptitle(title, fontsize=28)


""" Example for plotting Clusters
data_dir = "data/"
ambig_data = np.load(data_dir + "ambiguous_data_equal_noise=[0,0.2]_F_signal_noise.npy")
ambig_amplitude_conditions = ["S", "S/M", "M", "M/L", "L"]
ambig_time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]

ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]

samples_per_condition = 1000
samples_per_ambiguous_condition = 400

ambig_cluster_dict = get_index_per_class(ambig_amplitude_conditions,
                                         ambig_time_constant_conditions, 
                                         ambiguous_conditions, 
                                         samples_per_condition, 
                                         samples_per_ambiguous_condition)

ambig_true_labels = get_labels(ambig_data, ambig_cluster_dict)

# Clusters in our dataset
ambig_clusters_ordered = list(range(0,len(ambig_cluster_dict)+1))


# We have 9 clusters for each amplitude and we want them to be plotted in a nice grid format with different
# not overlapping in rows thats why we allocate 2 rows and 5 columns for each amplitude 
ambig_layout_label_mapping = labels_to_layout_mapping(ambig_clusters_ordered, 9, (2,5))

k = "Full"
labels = np.load("labels_ambig_equal_noise=[0,0.2]_SSIM_RAW_k=%s_reg=None_weighting=True.npy" % str(k))
  
clustered_labels = {}
for i, labels_i in enumerate(labels):
    clustered_labels[i+1] = labels_i
    


For Clear Dataset:
rows = 3
columns = 4
figsize = (20,20)
clear_layout_label_mapping = 4, (1,4)
subplot_adjustments = [0.05,0.95,0.03,0.9,0.4, 0.15]



k_clusters = 14

save_file_clusters = "F1_clusters_k=%s_ambig_equal_noise=[0,0.2]_SSIM_RAW_kclusters=%d.pdf" % (str(k),k_clusters)
title = r"SSIM-RAW Clusters"+ "\n k=%s" % str(k)

functions_for_plotting.plot_clusters(ambig_data, # the dataset 
                                     ambig_true_labels, # the true labels for the dataset 
                                     clustered_labels[k_clusters],  # the clustered labels 
                                     10, # the number of rows in the grid 
                                     5, # the number of columns in the grid 
                                     ambig_layout_label_mapping, # our layout mapping 
                                     figsize=(40,30), # the figsize
                                     n_bursts = 100, # the number of bursts you want to plot for each cluster 
                                     y_lim = (0,16), # the y_lim
                                     save_file=save_file_clusters, # the file you want to save the plot 
                                     subplot_adjustments= [0.05,0.93,0.02,0.92,0.9, 0.2], # adjustments for suplots and overall spacing (tricky) 
                                     plot_mean=False, # plot the mean of each cluster ? 
                                     title= title) # title of the plot


clusters_from_ambig_dataset, counts = np.unique(ambig_true_labels, return_counts = True)
clear_clusters_from_ambig = clusters_from_ambig_dataset[np.where(counts!= 400)]
clear_clusters_from_ambig_idx = np.where(np.isin(ambig_true_labels,clear_clusters_from_ambig) == True)[0]
ambig_clear_inidices = np.asarray(list(range(len(ambig_data))))[clear_clusters_from_ambig_idx]
#ambig_clear_valid_inidices = ambig_train_fold_indices[1][clear_clusters_from_ambig_idx_validation]
ambig_clear_labels = {}
for i, labels_i in enumerate(labels):
    ambig_clear_labels[i+1] = labels_i[clear_clusters_from_ambig_idx]

ambig_clear_true_labels = ambig_true_labels[clear_clusters_from_ambig_idx]    


#k = 50
#reg = None
#k_clusters = 14

save_file_clusters = "F1_clusters_k=%s_ambig_equal_noise=[0,0.2]_SSIM_RAW_kclusters=%d_clear_clusters.pdf" % (str(k),k_clusters)
title = r"SSIM-RAW Clear Clusters" + "\n k=%s" % str(k)


functions_for_plotting.plot_clusters(ambig_data[clear_clusters_from_ambig_idx], # the dataset 
                                     ambig_true_labels[ambig_clear_inidices], # the true labels for the dataset 
                                     ambig_clear_labels[k_clusters],  # the clustered labels 
                                     10, # the number of rows in the grid 
                                     5, # the number of columns in the grid 
                                     ambig_layout_label_mapping, # our layout mapping 
                                     figsize=(40,30), # the figsize
                                     n_bursts = 100, # the number of bursts you want to plot for each cluster 
                                     y_lim = (0,16), # the y_lim
                                     save_file=save_file_clusters, # the file you want to save the plot 
                                     subplot_adjustments= [0.05,0.93,0.02,0.92,0.9, 0.2], # adjustments for suplots and overall spacing (tricky) 
                                     plot_mean=False, # plot the mean of each cluster ? 
                                     title= title )# title of the plot    
"""


"""
Wagenaar Example:
data_dir = "data/raw_data/daily_spontanous_dense/day20/"
data = np.load(data_dir + "data_burst_by_time_day_20.npy").T

data_burst_batches_files = [x for x in os.listdir(data_dir) if x.find("burst_data_batch_") >= 0 and x.find("tiny") <0]
tiny_burst_indices_files = [x for x in os.listdir(data_dir) if x.find("burst_data_batch_tiny_index") >= 0]
data_batch_names = [x.split('.')[0] for x in data_burst_batches_files]
data_batch_names_for_tiny_indices = [x.split('.')[0] for x in tiny_burst_indices_files]

data_burst_batches = wagenaar_dataset.load_batch_files_with_number_bursts(data_dir, data_batch_names)
_, culture_count_dict = wagenaar_dataset.merge_data_batches_ordered(data_burst_batches, day_wise = False) # get number of bursts per culture

tiny_burst_count_dict, tiny_bursts_indices_dict = wagenaar_dataset.load_tiny_indices_per_culture(data_dir, data_batch_names_for_tiny_indices, day_wise = False) # get number of tiny bursts per culture and indices within culture
tiny_bursts_in_data_indices = wagenaar_dataset.get_tiny_burst_indices_for_merged_data(culture_count_dict, tiny_bursts_indices_dict) # get indices with respect to the whole dataset

culture_dict = wagenaar_dataset.get_culture_dict(culture_count_dict)

labels = np.load("labels_day20_Euclidean_k=10_reg=None_100clusters.npy",allow_pickle=True)
clustered_labels = {}
for i, labels_i in enumerate(labels):
    clustered_labels[i+1] = labels_i
    

save_file_clusters = "test.pdf"
k_clusters = 11
reference_clusters = 11
title = "Clusters Day 20 \n reg=1"

functions_for_plotting.plot_clusters(data, # the dataset 
                                     clustered_labels[reference_clusters], # the reference labels for the dataset 
                                     clustered_labels[k_clusters],  # the clustered labels 
                                     4, # the number of rows in the grid 
                                     4, # the number of columns in the grid 
                                     None, # our layout mapping 
                                     figsize=(20,20), # the figsize
                                     #reference_clustering="k-clusters=14", 
                                     n_bursts = 100, # the number of bursts you want to plot for each cluster 
                                     y_lim = (0,16), # the y_lim for zoomed plot (0,1)
                                     save_file=save_file_clusters, # the file you want to save the plot 
                                     subplot_adjustments= [0.05,0.95,0.03,0.9,0.4, 0.15], # adjustments for suplots and overall spacing (tricky) 
                                     plot_mean=False, # plot the mean of each cluster ? 
                                     title= title )# title of the plot 
                                     
# Displaying F1-Score:

condition = "5_fold_random"
reg = "None"
folds = 5

train_fold_indices, valid_fold_indices = get_training_folds(data_no_tiny,culture_dict_no_tiny, cluster_split = "random",folds = 5)

labels_valid = np.load("labels_day20_Euclidean_k=10_reg=%s_%s_valid_100clusters.npy" % (reg,condition),allow_pickle=True)
labels_train = np.load("labels_day20_Euclidean_k=10_reg=%s_%s_train_100clusters.npy" % (reg,condition),allow_pickle=True)
clustered_labels_valid = {}
clustered_labels_train = {}
for i in range(len(labels_valid[0])):
    clustered_labels_valid[i+1] = labels_valid[:,i]
    clustered_labels_train[i+1] = labels_train[:,i]

F1_scores = {}
for i in range(1,101):
    F1_scores[i] = []

for f in range(1,folds+1):
    F1_score = np.load("F1_day20_Euclidean_k=10_reg=%s_%s_%d_100clusters_clusterwise.npy" % (reg,condition,f),allow_pickle=True).item()
    for k in range(1,101):
        F1_scores[k].append(F1_score[k])
        
save_file_clusters = "test.pdf"
k_clusters = 14
reference_clusters = 14
title = "Clusters Day 20 (no-tiny, Validation 1) \n reg=%s" % reg

     
                                     
functions_for_plotting.plot_clusters(data[train_fold_indices[0]], # the dataset 
                                     clustered_labels_train[reference_clusters][0], # the reference labels for the dataset 
                                     clustered_labels_train[k_clusters][0],  # the clustered labels 
                                     4, # the number of rows in the grid 
                                     4, # the number of columns in the grid 
                                     None, # our layout mapping 
                                     figsize=(20,20), # the figsize
                                     reference_clustering="True",
                                     scores = None,
                                     n_bursts = 100, # the number of bursts you want to plot for each cluster 
                                     y_lim = (0,16), # the y_lim for zoomed plot (0,1) normal (0,16)
                                     save_file=save_file_clusters, # the file you want to save the plot 
                                     subplot_adjustments= [0.05,0.95,0.03,0.9,0.4, 0.15], # adjustments for suplots and overall spacing (tricky) 
                                     plot_mean=False, # plot the mean of each cluster ? 
                                     title= title )# title of the plot                                      
                                                                 
                                     
"""


