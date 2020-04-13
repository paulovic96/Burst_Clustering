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


