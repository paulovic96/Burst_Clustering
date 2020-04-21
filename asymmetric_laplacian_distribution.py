import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def asymmetric_laplace_distribution(x,mu,lam,tau1,tau2):
    """
    :param x:
    :param mu:
    :param tau1:
    :param tau2:
    :return:
    """
    if x < mu:
        f = lam*np.exp(tau1*(x-mu))
    else:
        f = lam*np.exp(-tau2*(x-mu))

    return f



# small amplitudes: 1 - 2
# medium amplitudes: 5 - 7
# large amplitudes: 12 - 14

# tau_small: 0.001 - 0.003 --> wide
# tau_large: 0.02 - 0.5 --> sharp
# tau_medium: 0.004 - 0.019 --> ambiguous



def sample_tau(lower, upper):
    return np.random.uniform(lower, upper)

def sample_amplitude(lower, upper):
    return np.random.uniform(lower, upper)

def generate_ALD(X, mu, amplitude_condition, time_constant_condition):
    V_ald = np.vectorize(asymmetric_laplace_distribution)

    if amplitude_condition == "S":
        lam = sample_amplitude(1, 2)

    elif amplitude_condition =="S/M":
        lam = sample_amplitude(3, 4)

    elif amplitude_condition == "M":
        lam = sample_amplitude(5, 7)

    elif amplitude_condition == "M/L":
        lam = sample_amplitude(8, 11)

    elif amplitude_condition == "L":
        lam = sample_amplitude(12, 14)
    else:
        print("Invalid amplitude condition: %s ..." % amplitude_condition)

    if time_constant_condition == "equal_sharp":
        tau1 = sample_tau(lower = 0.02, upper=0.5) # tau_large = tau_large
        tau2 = sample_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "equal_medium":
        tau1 = sample_tau(lower = 0.004, upper=0.019) # tau_medium = tau_medium #sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower = 0.004, upper=0.019)

    elif time_constant_condition == "equal_wide":
        tau1 = sample_tau(lower=0.001, upper=0.003) # tau_small = tau_small
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "wide_sharp_negative_skew": # tau_small << tau_large
        tau1 = sample_tau(lower=0.001, upper=0.003)
        tau2 = sample_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "wide_medium_negative_skew": # tau_small < tau_medium
        tau1 = sample_tau(lower=0.001, upper=0.003)
        tau2 = sample_tau(lower = 0.004, upper=0.019) #sample_tau(lower=0.007, upper=0.014)

    elif time_constant_condition == "medium_sharp_negative_skew": # tau_medium < tau_large
        tau1 = sample_tau(lower = 0.004, upper=0.019) #sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower=0.02, upper=0.5)

    elif time_constant_condition == "sharp_wide_positive_skew": # tau_large >> tau_small
        tau1 = sample_tau(lower = 0.02, upper=0.5)
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "medium_wide_positive_skew": # tau_medium > tau_small
        tau1 = sample_tau(lower = 0.004, upper=0.019)#sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "sharp_medium_positive_skew": # tau_large > tau_medium
        tau1 = sample_tau(lower=0.02, upper=0.5)
        tau2 = sample_tau(lower = 0.004, upper=0.019) #sample_tau(lower=0.007, upper=0.014)

    else:
        print("Invalid time_constant condition: %s ..." % time_constant_condition)

    f = V_ald(X, mu=mu, lam=lam, tau1=tau1, tau2=tau2)

    return f, tau1, tau2, lam


#def generate_ALD(X, mu, amplitude_conditions, time_constant_conditions):
#    V_ald = np.vectorize(asymmetric_laplace_distribution)
#
#    lam = np.random.choice(amplitude_conditions)
#    tau1,tau2 = np.random.choice(time_constant_conditions,2)
#
#    f = V_ald(X, mu=mu, lam=lam, tau1=tau1, tau2=tau2)
#
#    return f, tau1, tau2, lam



def generate_ALD_data(X, amplitude_conditions, time_constant_conditions, ambiguous_conditions, samples_per_condition=1000,samples_per_ambiguous_condition=100, mu=1750):
    conditions = []
    tau1s = []
    tau2s = []
    lams = []
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
                conditions.append(amplitude_condition + "/" + time_constant_condition)
                f_i, tau1, tau2, lam = generate_ALD(X,mu=mu,amplitude_condition=amplitude_condition, time_constant_condition=time_constant_condition)
                if amplitude_condition == "S":
                    noise = np.random.normal(0, 0.02, f_i.shape)
                elif amplitude_condition == "S/M":
                    noise = np.random.normal(0, 0.05, f_i.shape)
                elif amplitude_condition == "M":
                    noise = np.random.normal(0, 0.07, f_i.shape)
                elif amplitude_condition == "M/L":
                    noise = np.random.normal(0, 0.11, f_i.shape)
                elif amplitude_condition == "M":
                    noise = np.random.normal(0, 0.14, f_i.shape)


                F_signal.append(f_i)
                F_signal_noise.append((f_i+noise).clip(0)) # no negative spike counts
                noises.append(noise)
                tau1s.append(tau1)
                tau2s.append(tau2)
                lams.append(lam)

    param_data = pd.DataFrame(
    {'Condition': conditions,
     'Lambda': lams,
     'Tau1': tau1s,
     'Tau2': tau2s
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

def get_labels(data, cluster_dict,ambiguous_conditions,true_clusters_starting_point = 0, new_clusters_amplitude_starting_point = 27, new_clusters_tau_starting_point = 12):
    true_labels_ambiguous = np.zeros(len(data))

    true_cluster = true_clusters_starting_point -1
    new_clusters_amplitude = new_clusters_amplitude_starting_point - 1 # new amplitude + normal and new tau: 18
    new_clusters_tau = new_clusters_tau_starting_point - 1 # new tau + amplitude: 15
    for key in cluster_dict.keys():
        amplitude, time_course = key.split("-")
        if amplitude in ambiguous_conditions:
            new_clusters_amplitude += 1
            true_labels_ambiguous[cluster_dict[key][0]:cluster_dict[key][1]] = new_clusters_amplitude

        else:
            if time_course in ambiguous_conditions:
                new_clusters_tau += 1
                true_labels_ambiguous[cluster_dict[key][0]:cluster_dict[key][1]] = new_clusters_tau
            else:
                true_cluster += 1
                true_labels_ambiguous[cluster_dict[key][0]:cluster_dict[key][1]] = true_cluster
    return true_labels_ambiguous

def get_amplitude_and_time_condtion_from_condition_string(condition_string):
    conditions = condition_string.split("/")
    if len(conditions) > 2:
        amplitude = "/".join(conditions[0:2])
    else:
        amplitude = conditions[0]

    time_condition = conditions[-1]
    return [amplitude, time_condition]




def main():
    print("Generating asymmetric laplacian Data!!!")
    X = np.round(np.linspace(0,3500,3501))
    np.random.seed(42)

    amplitude_conditions = ["S", "S/M", "M", "M/L", "L"] #["S", "M", "L"]
    time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","sharp_wide_positive_skew", "medium_wide_positive_skew" ,"sharp_medium_positive_skew"]
    # ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "medium_sharp_negative_skew", "wide_sharp_positive_skew", "medium_sharp_positive_skew"]
    ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "medium_wide_positive_skew", "sharp_medium_positive_skew"]
    #["equal_medium", "medium_sharp_negative_skew", "sharp_medium_positive_skew"]
    samples_per_condition = 1000
    samples_per_ambiguous_condition = 400
    mu = 1750

    F_signal, F_signal_noise, noises, param_data = generate_ALD_data(X, amplitude_conditions, time_constant_conditions, ambiguous_conditions,samples_per_condition=samples_per_condition,samples_per_ambiguous_condition=samples_per_ambiguous_condition,mu=mu)
    print("Done!")


    param_data.to_csv("data/parameter_ambiguous_data_tau_amplitude",index=False)

    np.save('data/F_signal_ambiguous_tau_amplitude',F_signal)
    np.save('data/F_signal_noise_ambiguous_tau_amplitude', F_signal_noise)

    class_dict = get_index_per_class(amplitude_conditions, time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)
    amplitude_dict = {"S": "Small", "M": "Medium", "S/M": "Small/Medium", "L": "Large", "M/L": "Medium/Large"}

    data = F_signal_noise
    for key in list(class_dict.keys()):
        start = class_dict[key][0]
        end = class_dict[key][1]
        class_i = data[start:end]

        fig, ax = plt.subplots()
        for i in class_i:
            ax.plot(i)
        ax.set_xlabel("Time", fontsize=10, labelpad=8)

        amplitude, time_course = str(key).split("-")
        time_course_title = [x.capitalize() for x in time_course.split("_")]

        if amplitude in ambiguous_conditions or time_course in ambiguous_conditions:
            title = ("Generated Clusters \n Ambiguous Data n = %d \n Amplitude: %s - Time Course: " + (" ").join(
                ['%s'] * len(time_course_title))) % tuple([samples_per_ambiguous_condition] + [amplitude_dict[amplitude]] + time_course_title)

        else:
            title = ("Generated Clusters \n n = %d \n Amplitude: %s - Time Course: " + (" ").join(
                ['%s'] * len(time_course_title))) % tuple([samples_per_condition] + [amplitude_dict[amplitude]] + time_course_title)

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


if __name__== "__main__":
  main()





#amplitude_conditions = ["S", "S/M", "M", "M/L", "L"]
#time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","wide_sharp_positive_skew", "wide_medium_positive_skew" ,"medium_sharp_positive_skew"]
#ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "wide_medium_positive_skew", "medium_sharp_positive_skew"]
#samples_per_condition = 1000
#samples_per_ambiguous_condition = 200

#class_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_per_ambiguous_condition)


#import matplotlib.pyplot as plt
# tau_medium: 0.003 - 0.014 #0.007 - 0.014
# tau_small: 0.001 - 0.003 --> wide
# tau_large: 0.02 - 0.5 --> sharp


#X = np.round(np.linspace(0,3500,3501))
#mu = 1750
#V_ald = np.vectorize(asymmetric_laplace_distribution)
#lam = sample_amplitude(5, 7)


#tau1 = sample_tau(lower = 0.003, upper=0.014)
#tau2 = sample_tau(lower = 0.003, upper=0.014)

#f = V_ald(X, mu=mu, lam=lam, tau1=tau1, tau2=tau2)
#noise = np.random.normal(0, 0.12, f.shape)

#plt.plot(X,f+noise)




