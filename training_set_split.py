import numpy as np

def get_training_set(data,true_n_clusters,n_samples_per_clusters,proportion=0.5):

    seed = np.random.seed(42)
    training_split = list(range(true_n_clusters))
    np.random.shuffle(training_split)
    training_split = np.sort(training_split[:int(np.floor(true_n_clusters*proportion))])

    training_set_indices = []
    for i in training_split:
        training_set_indices += list(range(i*n_samples_per_clusters,(i+1)*n_samples_per_clusters))
    training_set_indices = np.asarray(training_set_indices)

    training_set = data[training_set_indices]
    return training_set, training_set_indices



def get_training_set_for_ambiguous_data(data, cluster_dict, ambiguous_conditions,ambiguous_tau = True, ambiguous_amplitude = False,proportion=0.5):
    true_clusters = []
    ambiguous_clusters = []

    for key in cluster_dict.keys():
        amplitude, time_course = key.split("-")
        if amplitude in ambiguous_conditions:
            if ambiguous_amplitude:
                if time_course in ambiguous_conditions:
                    if ambiguous_tau:
                        ambiguous_clusters.append((key, cluster_dict[key]))
                else:
                    ambiguous_clusters.append((key, cluster_dict[key]))
        else:
            if time_course in ambiguous_conditions:
                if ambiguous_tau:
                    ambiguous_clusters.append((key, cluster_dict[key]))
            else:
                true_clusters.append((key, cluster_dict[key]))

    seed = np.random.seed(42)
    true_clusters_training_split = list(range(len(true_clusters)))
    ambiguous_clusters_training_split = list(range(len(ambiguous_clusters)))

    np.random.shuffle(true_clusters_training_split)
    np.random.shuffle(ambiguous_clusters_training_split)

    true_clusters_training_split = true_clusters_training_split[:int(np.floor(len(true_clusters_training_split) * proportion))]
    ambiguous_clusters_training_split = ambiguous_clusters_training_split[:int(np.floor(len(ambiguous_clusters_training_split) * proportion))]

    training_clusters = list(np.asarray(true_clusters)[true_clusters_training_split]) + list(np.asarray(ambiguous_clusters)[ambiguous_clusters_training_split])

    training_set_indices = []
    training_set_conditions = []
    for i in training_clusters:
        training_set_indices += list(range(i[1][0],i[1][1]))
        training_set_conditions += [i[0]]

    training_set_indices = np.asarray(training_set_indices)

    training_set = data[training_set_indices]

    return training_set, training_set_indices, training_set_conditions


def get_training_folds(data, cluster_dict=None, cluster_split="random",only_training_clusters = None, seed=42, folds = 2):
    seed = np.random.seed(seed)
    train_fold_indices = []
    valid_fold_indices = []
    if cluster_split == "balanced":
        if not cluster_dict is None:
            if only_training_clusters is None:
                for i in range(folds - 1):
                    valid_fold = []
                    train_fold = []
                    for cluster, start_end_point in cluster_dict.items():
                        if isinstance(start_end_point, tuple):
                            cluster_indices = np.arange(start_end_point[0], start_end_point[1] + 1)
                        else:
                            cluster_indices = start_end_point
                        np.random.shuffle(cluster_indices)
                        fold_length =int(len(cluster_indices)/folds)

                        valid_fold += [cluster_indices[i * fold_length:(i + 1) * fold_length]]
                        train_fold += [np.delete(cluster_indices, range(i * fold_length, (i + 1) * fold_length), axis=0)]

                    valid_fold = np.concatenate(valid_fold).ravel().astype(int)
                    train_fold = np.concatenate(train_fold).ravel().astype(int)

                    train_fold_indices += [train_fold]
                    valid_fold_indices += [valid_fold]

                valid_fold = []
                train_fold = []
                for cluster, start_end_point in cluster_dict.items():
                    if isinstance(start_end_point, tuple):
                        cluster_indices = np.arange(start_end_point[0], start_end_point[1] + 1)
                    else:
                        cluster_indices = start_end_point
                    np.random.shuffle(cluster_indices)
                    fold_length = int(len(cluster_indices) / folds)

                    valid_fold += [cluster_indices[(folds - 1) * fold_length:]]
                    train_fold += [np.delete(cluster_indices, range((folds - 1) * fold_length, len(cluster_indices)), axis=0)]

                valid_fold = np.concatenate(valid_fold).ravel().astype(int)
                train_fold = np.concatenate(train_fold).ravel().astype(int)

                train_fold_indices += [train_fold]
                valid_fold_indices += [valid_fold]

        else:
            print("Please provide a valid cluster dictionary!")

    elif cluster_split == "unbalanced":
        if not cluster_dict is None:
            if not only_training_clusters is None:
                for i in range(folds - 1):
                    valid_fold = []
                    train_fold = []
                    for k, item in enumerate(cluster_dict.items()):
                        cluster = item[0]
                        start_end_point = item[1]
                        if isinstance(start_end_point, tuple):
                            cluster_indices = np.arange(start_end_point[0], start_end_point[1] + 1)
                        else:
                            cluster_indices = start_end_point
                        fold_length =int(len(cluster_indices)/folds)

                        if k in only_training_clusters:
                            valid_fold += [[]]
                            train_fold += [cluster_indices]
                        else:
                            valid_fold += [cluster_indices[i * fold_length:(i + 1) * fold_length]]
                            train_fold += [np.delete(cluster_indices, range(i * fold_length, (i + 1) * fold_length), axis=0)]

                    valid_fold = np.concatenate(valid_fold).ravel().astype(int)
                    train_fold = np.concatenate(train_fold).ravel().astype(int)

                    train_fold_indices += [train_fold]
                    valid_fold_indices += [valid_fold]

                valid_fold = []
                train_fold = []
                for k, item in enumerate(cluster_dict.items()):
                    cluster = item[0]
                    start_end_point = item[1]

                    if isinstance(start_end_point, tuple):
                        cluster_indices = np.arange(start_end_point[0], start_end_point[1] + 1)
                    else:
                        cluster_indices = start_end_point
                    fold_length = int(len(cluster_indices) / folds)

                    if k in only_training_clusters:
                        valid_fold += [[]]
                        train_fold += [cluster_indices]
                    else:
                        valid_fold += [cluster_indices[(folds - 1) * fold_length:]]
                        train_fold += [np.delete(cluster_indices, range((folds - 1) * fold_length, len(cluster_indices)), axis=0)]

                valid_fold = np.concatenate(valid_fold).ravel().astype(int)
                train_fold = np.concatenate(train_fold).ravel().astype(int)

                train_fold_indices += [train_fold]
                valid_fold_indices += [valid_fold]
            else:
                print("Please provide a valid list of clusters for your training set!")
        else:
            print("Please provide a valid cluster dictionary!")

    elif cluster_split == "random":
        n_datapoints = np.arange(len(data))
        np.random.shuffle(n_datapoints)
        fold_length = int(len(n_datapoints)/folds)

        for i in range(folds - 1):
            valid_fold = n_datapoints[i * fold_length:(i + 1) * fold_length]
            train_fold = np.delete(n_datapoints, range(i * fold_length, (i + 1) * fold_length), axis=0)

            train_fold_indices += [train_fold]
            valid_fold_indices += [valid_fold]

        valid_fold_indices += [n_datapoints[(folds - 1) * fold_length:]]
        train_fold_indices += [np.delete(n_datapoints, range((folds - 1) * fold_length, len(n_datapoints)), axis=0)]

    return train_fold_indices, valid_fold_indices
