import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "data"
SAVE_FILE_NAME = "clearly_separated_data"


# Amplitudes
S = [1,2]
S_M = [3,4]
M = [5,7]
M_L = [8,11]
L = [12,14]

# Time Constants
SMALL_TAU = [0.001, 0.003]
MEDIUM_TAU = [0.004, 0.019]
LARGE_TAU = [0.02, 0.5]

# Noises
SMALL_NOISE = [0, 0.02]
SMALL_MEDIUM_NOISE = [0, 0.05]
MEDIUM_NOISE = [0, 0.07]
MEDIUM_LARGE_NOISE = [0, 0.11]
LARGE_NOISE = [0, 0.14]


# mu
MU = 1750
TIME_RANGE = [0, 3500]


# Conditions
AMPLITUDE_CONDITIONS = ["S", "S/M", "M", "M/L", "L"]

TIME_CONSTANT_CONDITIONS = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew",
                            "wide_medium_negative_skew", "medium_sharp_negative_skew", "sharp_wide_positive_skew",
                            "medium_wide_positive_skew", "sharp_medium_positive_skew"]

AMBIGUOUS_CONDITIONS = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew",
                        "medium_wide_positive_skew", "sharp_medium_positive_skew"]


SAMPLES_PER_CONDITION = 1000
SAMPLES_PER_AMBIGUOUS_CONDITION = 400



def asymmetric_laplace_distribution(x,mu,lam,tau1,tau2):
    if x < mu:
        f = lam*np.exp(tau1*(x-mu))
    else:
        f = lam*np.exp(-tau2*(x-mu))
    return f


def sample_parameter(lower, upper):
    return np.random.uniform(lower, upper)


def generate_ALD(X, mu, amplitude_condition, time_constant_condition):
    V_ald = np.vectorize(asymmetric_laplace_distribution)

    if amplitude_condition == "S":
        lam = sample_parameter(S[0], S[1])

    elif amplitude_condition =="S/M":
        lam = sample_parameter(S_M[0], S_M[1])

    elif amplitude_condition == "M":
        lam = sample_parameter(M[0], M[1])

    elif amplitude_condition == "M/L":
        lam = sample_parameter(M_L[0], M_L[1])

    elif amplitude_condition == "L":
        lam = sample_parameter(L[0], L[1])
    else:
        print("Invalid amplitude condition: %s ..." % amplitude_condition)

    if time_constant_condition == "equal_sharp":
        tau1 = sample_parameter(LARGE_TAU[0], LARGE_TAU[1]) # tau_large = tau_large
        tau2 = sample_parameter(LARGE_TAU[0], LARGE_TAU[1])

    elif time_constant_condition == "equal_medium":
        tau1 = sample_parameter(MEDIUM_TAU[0], MEDIUM_TAU[1]) # tau_medium = tau_medium
        tau2 = sample_parameter(MEDIUM_TAU[0], MEDIUM_TAU[1])

    elif time_constant_condition == "equal_wide":
        tau1 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1]) # tau_small = tau_small
        tau2 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1])

    elif time_constant_condition == "wide_sharp_negative_skew": # tau_small << tau_large
        tau1 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1])
        tau2 = sample_parameter(LARGE_TAU[0], LARGE_TAU[1])

    elif time_constant_condition == "wide_medium_negative_skew": # tau_small < tau_medium
        tau1 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1])
        tau2 = sample_parameter(MEDIUM_TAU[0], MEDIUM_TAU[1])

    elif time_constant_condition == "medium_sharp_negative_skew": # tau_medium < tau_large
        tau1 = sample_parameter(MEDIUM_TAU[0], MEDIUM_TAU[1])
        tau2 = sample_parameter(LARGE_TAU[0], LARGE_TAU[1])

    elif time_constant_condition == "sharp_wide_positive_skew": # tau_large >> tau_small
        tau1 = sample_parameter(LARGE_TAU[0], LARGE_TAU[1])
        tau2 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1])

    elif time_constant_condition == "medium_wide_positive_skew": # tau_medium > tau_small
        tau1 = sample_parameter(MEDIUM_TAU[0], MEDIUM_TAU[1])
        tau2 = sample_parameter(SMALL_TAU[0], SMALL_TAU[1])

    elif time_constant_condition == "sharp_medium_positive_skew": # tau_large > tau_medium
        tau1 = sample_parameter(LARGE_TAU[0],LARGE_TAU[1])
        tau2 = sample_parameter(MEDIUM_TAU[0],MEDIUM_TAU[1])

    else:
        print("Invalid time_constant condition: %s ..." % time_constant_condition)

    f = V_ald(X, mu=mu, lam=lam, tau1=tau1, tau2=tau2)

    return f, tau1, tau2, lam



def generate_ALD_data(X, amplitude_conditions, time_constant_conditions, ambiguous_conditions, samples_per_condition=1000,samples_per_ambiguous_condition=100, mu=1750):
    # Random Seed
    np.random.seed(42)

    A_conditions = []
    tau_conditions = []

    tau1s = []
    tau2s = []
    amplitudes = []
    F_signal = []
    F_signal_noise = []
    noises = []

    for amplitude_condition in amplitude_conditions:
        for time_constant_condition in time_constant_conditions:
            if amplitude_condition in ambiguous_conditions or time_constant_condition in ambiguous_conditions:
                n_samples = samples_per_ambiguous_condition
            else:
                n_samples = samples_per_condition

            for i in range(n_samples):
                A_conditions.append(amplitude_condition)
                tau_conditions.append(time_constant_condition)

                f_i, tau1, tau2, lam = generate_ALD(X,mu=mu,amplitude_condition=amplitude_condition, time_constant_condition=time_constant_condition)
                if amplitude_condition == "S":
                    noise = np.random.normal(SMALL_NOISE[0], SMALL_NOISE[0], f_i.shape)
                elif amplitude_condition == "S/M":
                    noise = np.random.normal(SMALL_MEDIUM_NOISE[0], SMALL_MEDIUM_NOISE[1], f_i.shape)
                elif amplitude_condition == "M":
                    noise = np.random.normal(MEDIUM_NOISE[0],MEDIUM_NOISE[1], f_i.shape)
                elif amplitude_condition == "M/L":
                    noise = np.random.normal(MEDIUM_LARGE_NOISE[0], MEDIUM_LARGE_NOISE[1], f_i.shape)
                elif amplitude_condition == "L":
                    noise = np.random.normal(LARGE_NOISE[0], LARGE_NOISE[1], f_i.shape)


                F_signal.append(f_i)
                F_signal_noise.append((f_i+noise).clip(0)) # no negative spike counts
                noises.append(noise)
                tau1s.append(tau1)
                tau2s.append(tau2)
                amplitudes.append(lam)

    param_data = pd.DataFrame(
    {'amplitude_condition': A_conditions,
     'time_constant_condition': tau_conditions,
     'ampltiude': amplitudes,
     'tau1': tau1s,
     'tau2': tau2s
    })
    return F_signal, F_signal_noise, noises, param_data


def get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_for_ambiguous):
    class_dict = {}
    current_index = 0
    for amplitude_condition in amplitude_conditions:
        for time_constant_condition in time_constant_conditions:
            condition = amplitude_condition + "-" + time_constant_condition
            if amplitude_condition in ambiguous_conditions or time_constant_condition in ambiguous_conditions:
                class_dict[condition] = [current_index, current_index + samples_for_ambiguous-1]
                current_index += samples_for_ambiguous
            else:
                class_dict[condition] = [current_index, current_index + samples_per_condition-1]
                current_index += samples_per_condition
    return class_dict


def get_labels(data,cluster_dict, cluster_order = None):
    if not cluster_order:
        cluster_order = list(cluster_dict.keys())

    labels = np.zeros(len(data))
    for i, key in enumerate(cluster_order):
        labels[cluster_dict[key][0]:cluster_dict[key][1] + 1] = i
    return labels

def labels_to_layout_mapping(k_cluster_labels_ordered, clusters_per_condition_for_layout, layout_per_condition):
    layout_label_mapping = {}
    shift_needed_to_stay_in_layout = layout_per_condition[0] * layout_per_condition[1] - clusters_per_condition_for_layout

    label = 0
    for i in k_cluster_labels_ordered:
        if i == 0:
            layout_label_mapping[i] = label
        else:
            if i % clusters_per_condition_for_layout == 0:
                label += (1 + shift_needed_to_stay_in_layout)
            else:
                label += 1

            layout_label_mapping[i] = label
    return layout_label_mapping



def main():
    print("Generating asymmetric laplacian Data!!!")
    X = np.round(np.linspace(TIME_RANGE[0],TIME_RANGE[1],TIME_RANGE[1] + 1))
    np.random.seed(42)

    F_signal, F_signal_noise, noises, param_data = generate_ALD_data(X, AMPLITUDE_CONDITIONS, TIME_CONSTANT_CONDITIONS, AMBIGUOUS_CONDITIONS, SAMPLES_PER_CONDITION,SAMPLES_PER_AMBIGUOUS_CONDITION,MU)
    print("Done!")

    param_data.to_csv(SAVE_DIR +"/" + SAVE_FILE_NAME + "_parameter" + ".csv",index=False)

    np.save(SAVE_DIR + "/" + SAVE_FILE_NAME + "_F_signal",F_signal)
    np.save(SAVE_DIR + "/" + SAVE_FILE_NAME + "_F_signal_noise", F_signal_noise)

    """
    class_dict = get_index_per_class(AMPLITUDE_CONDITIONS, TIME_CONSTANT_CONDITIONS, AMBIGUOUS_CONDITIONS, SAMPLES_PER_CONDITION, SAMPLES_PER_AMBIGUOUS_CONDITION)
    amplitude_dict = {"S": "Small", "M": "Medium", "S/M": "Small/Medium", "L": "Large", "M/L": "Medium/Large"}

    data = F_signal_noise
    for key in list(class_dict.keys()):
        start = class_dict[key][0]
        end = class_dict[key][1]
        class_i = data[start:end]

        fig, ax = plt.subplots()
        for i in class_i[0:100]:
            ax.plot(i)
        ax.set_xlabel("Time", fontsize=10, labelpad=8)

        amplitude, time_course = str(key).split("-")
        time_course_title = [x.capitalize() for x in time_course.split("_")]

        if amplitude in AMBIGUOUS_CONDITIONS or time_course in AMBIGUOUS_CONDITIONS:
            title = ("Generated Clusters \n Ambiguous Data n = %d \n Amplitude: %s - Time Course: " + (" ").join(
                ['%s'] * len(time_course_title))) % tuple([SAMPLES_PER_AMBIGUOUS_CONDITION] + [amplitude_dict[amplitude]] + time_course_title)

        else:
            title = ("Generated Clusters \n n = %d \n Amplitude: %s - Time Course: " + (" ").join(
                ['%s'] * len(time_course_title))) % tuple([SAMPLES_PER_CONDITION] + [amplitude_dict[amplitude]] + time_course_title)

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylim((0, 16))

        fig.savefig(key.replace("/", "_") + ".png", bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(np.mean(class_i, axis=0))
        ax.set_xlabel("Time", fontsize=10, labelpad=8)
        title_splitted = title.split("\n")
        title_splitted[0] = title_splitted[0] + "(Mean)"
        ax.set_title("\n".join(title_splitted), fontsize=12, pad=10)
        ax.set_ylim((0, 16))

        fig.savefig(key.replace("/", "_") + "_mean.png", bbox_inches="tight")
        plt.close()
        """

if __name__== "__main__":
  main()

