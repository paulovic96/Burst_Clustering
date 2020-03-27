import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

def plot_cluster_examples(data, labels, k_clusters,rows,columns, figsize = (30,25), burst_percent = False, n_bursts=None, y_lim = None, plot_mean = False, arrangement = None, title = None):
    if k_clusters < 10:
        colors = ["C" + str(i) for i in range(k_clusters)]
    else:
        colors = cm.rainbow(np.linspace(0, 1, k_clusters))

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if title:
        fig.suptitle(title, fontsize=16)

    for i in range(k_clusters):
        if arrangement is None:
            class_i = data[np.where(labels == i)]
        else:
            class_i = data[np.where(labels == arrangement[i])]
        #class_i = np.random.permutation(class_i)
        ax = fig.add_subplot(rows, columns, i+1)
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



data_dir = "data/"

data = np.load(data_dir + "F_signal_noise.npy")
true_labels = np.repeat(range(12), 1000)


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
    return splitted_classes[np.where(count > 1)[0]], count[np.where(count > 1)[0]]

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

label_predictions = np.load("Toy_data/Labels/labels_k=5_reg=0.1.npy")
k = 12
labels_k = label_predictions[k-1]


columns = 4
rows = 3
k_clusters = 12


def plot_clusters(data, true_labels,labels_k, k_clusters,rows,columns, figsize = None, burst_percent = False, n_bursts=None, y_lim = None, plot_mean = False, title = None):
    corresponding_true_class_for_prediction, highest_overlap_class_for_prediction = get_true_label_mapping(true_labels, labels_k)
    splitted_classes, split_count = get_splitted_clusters(highest_overlap_class_for_prediction)
    merged_classes, new_clusters = get_merged_clusters(corresponding_true_class_for_prediction)
    position_count_for_splitted_classes = np.zeros(len(split_count))

    normal_layout = np.arange(k_clusters).reshape((rows,columns))
    normal_rows = rows

    if len(splitted_classes) > 0:
        rows, rows_per_column = get_new_layout(k_clusters, rows, columns, splitted_classes, split_count)

    fig = plt.figure(figsize=figsize)
    #if title:
    #    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=4, wspace=0.2)
    gs = fig.add_gridspec(rows, columns)
    rows_per_layout_rows = int(rows/normal_rows)

    for i in range(k_clusters):
        class_i = data[np.where(labels_k == i)]
        corresponding_true_label = highest_overlap_class_for_prediction[i][0] # get true label and also position
        corresponding_column = int(np.where(normal_layout == corresponding_true_label)[1][0])
        corresponding_row = int(np.where(normal_layout == corresponding_true_label)[0][0] * rows_per_layout_rows)


        if corresponding_true_label in splitted_classes:
            splitted_into = split_count[np.where(splitted_classes==corresponding_true_label)[0]][0]
            rows_per_plot = rows_per_layout_rows/splitted_into
            position_count = position_count_for_splitted_classes[np.where(splitted_classes==corresponding_true_label)[0]]
            row_start = int(corresponding_row + position_count*np.floor(rows_per_plot))

            if position_count < splitted_into-1: # first plot of splitted cluster
                row_end = int(row_start + np.floor(rows_per_plot))
            else:
                row_end = int(row_start + np.ceil(rows_per_plot))

            ax = fig.add_subplot(gs[row_start:row_end, corresponding_column])
            position_count_for_splitted_classes[np.where(splitted_classes == corresponding_true_label)[0]] += 1
        else:
            row_start = corresponding_row
            row_end = corresponding_row+rows_per_layout_rows

            if i in new_clusters:
                corresponding_merged_true_clusters = list(merged_classes[np.where(new_clusters == i)[0]][0])
                corresponding_merged_true_clusters.remove(corresponding_true_label)
                for merged_class in corresponding_merged_true_clusters:
                    row_i = int(np.where(normal_layout == merged_class)[0][0] * rows_per_layout_rows)
                    column_i =  int(np.where(normal_layout == merged_class)[1][0])
                    ax = fig.add_subplot(gs[row_i:(row_i + rows_per_layout_rows), column_i])
                    ax.plot()
                    ax.set_xlabel("Time")
                    if y_lim:
                        ax.set_ylim(y_lim)


            ax = fig.add_subplot(gs[row_start:row_end, corresponding_column])

        ax.set_xlabel("Time")

        if plot_mean:
            ax.plot(np.mean(class_i, axis = 0))
        else:
            if not n_bursts:
                n_bursts = len(class_i)

            for burst in class_i[0:n_bursts]:
                ax.plot(burst)

        if y_lim:
            ax.set_ylim(y_lim)

        true_clusters = corresponding_true_class_for_prediction[i][0]
        true_cluster_counts = corresponding_true_class_for_prediction[i][1]

        if len(true_clusters)>1:
            cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(['%d']*len(true_clusters)) + "]" + "\n#: [" + ', '.join(['%d']*len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + list(true_cluster_counts))#sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
        else:
            cluster_title = ("Cluster %i \nTrue Cluster: [" + ', '.join(
                ['%d'] * len(true_clusters)) + "]" + " #: [" + ', '.join(
                ['%d'] * len(true_cluster_counts)) + "]") % tuple([i] + list(true_clusters) + list(
                true_cluster_counts))  # sum(list(map(list,zip(true_clusters,true_cluster_counts))),[]))
        ax.set_title(cluster_title,loc="left")

        #if burst_percent:
            #ax.set_title(
            #    "Class %i Mean (%i Bursts = %.2f %%)" % ((i + 1), len(class_i), len(class_i) / len(data) * 100),
            #    fontsize=12)
        #else:
            #ax.set_title("Class %i Mean (%i Bursts )" % ((i + 1), len(class_i)), fontsize=12)











plot_clusters(data, true_labels,labels_k, k_clusters,3,4, figsize = (30,15), burst_percent = False, n_bursts=10, y_lim = (0,16), plot_mean = False, title = None)

_, idx = np.unique(labels_k, return_index=True)
arrangement = labels_k[np.sort(idx)]




plot_cluster_examples(data, labels_k, k, 3, 4, figsize = (30,25), burst_percent = False, n_bursts=None, y_lim = (0,16),plot_mean=False, arrangement = None, title = None)

plt.close()
