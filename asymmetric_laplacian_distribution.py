import numpy as np
import pandas as pd

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
# tau_medium: 0.007 - 0.014 --> ambiguous



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
        lam = sample_amplitude(8, 10)

    elif amplitude_condition == "L":
        lam = sample_amplitude(12, 14)
    else:
        print("Invalid amplitude condition: %s ..." % amplitude_condition)

    if time_constant_condition == "equal_sharp":
        tau1 = sample_tau(lower = 0.02, upper=0.5)
        tau2 = sample_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "equal_medium":
        tau1 = sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower=0.007, upper=0.014)

    elif time_constant_condition == "equal_wide":
        tau1 = sample_tau(lower=0.001, upper=0.003)
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "wide_sharp_negative_skew":
        tau1 = sample_tau(lower=0.001, upper=0.003)
        tau2 = sample_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "wide_medium_negative_skew":
        tau1 = sample_tau(lower=0.001, upper=0.003)
        tau2 = sample_tau(lower=0.007, upper=0.014)

    elif time_constant_condition == "medium_sharp_negative_skew":
        tau1 = sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower=0.02, upper=0.5)

    elif time_constant_condition == "wide_sharp_positive_skew":
        tau1 = sample_tau(lower = 0.02, upper=0.5)
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "wide_medium_positive_skew":
        tau1 = sample_tau(lower=0.007, upper=0.014)
        tau2 = sample_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "medium_sharp_positive_skew":
        tau1 = sample_tau(lower=0.02, upper=0.5)
        tau2 = sample_tau(lower=0.007, upper=0.014)

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



def generate_ALD_data(X, amplitude_conditions, time_constant_conditions, ambiguous_conditions, samples_per_condition=1000, mu=1750):
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
                n_samples = 100
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
                F_signal_noise.append(f_i+noise)
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



def main():
    print("Generating asymmetric laplacian Data!!!")
    X = np.round(np.linspace(0,3500,3501))
    np.random.seed(42)

    amplitude_conditions = ["S", "S/M", "M", "M/L", "L"]
    time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","wide_sharp_positive_skew", "wide_medium_positive_skew" ,"medium_sharp_positive_skew"]
    ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "wide_medium_positive_skew", "medium_sharp_positive_skew"]
    samples_per_condition = 1000
    mu = 1750

    F_signal, F_signal_noise, noises, param_data = generate_ALD_data(X, amplitude_conditions, time_constant_conditions, ambiguous_conditions,samples_per_condition=samples_per_condition, mu=mu)
    print("Done!")


    param_data.to_csv("data/parameter_ambiguous_data",index=False)

    np.save('data/F_signal_ambiguous',F_signal)
    np.save('data/F_signal_noise_ambiguous', F_signal_noise)


if __name__== "__main__":
  main()

def get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_for_ambiguous):
    class_dict = {}
    current_index = 0
    for amplitude_condition in amplitude_conditions:
        for time_constant_condition in time_constant_conditions:
            condition = amplitude_condition + "-" + time_constant_condition
            if amplitude_condition in ambiguous_conditions or time_constant_condition in ambiguous_conditions:
                class_dict[condition] = [current_index, current_index + samples_for_ambiguous]
                current_index += samples_for_ambiguous
            else:
                class_dict[condition] = [current_index, current_index + samples_per_condition]
                current_index += samples_per_condition
    return class_dict



amplitude_conditions = ["S", "S/M", "M", "M/L", "L"]
time_constant_conditions = ["equal_sharp", "equal_medium", "equal_wide", "wide_sharp_negative_skew", "wide_medium_negative_skew","medium_sharp_negative_skew","wide_sharp_positive_skew", "wide_medium_positive_skew" ,"medium_sharp_positive_skew"]
ambiguous_conditions = ["S/M", "M/L", "equal_medium", "wide_medium_negative_skew", "medium_sharp_negative_skew", "wide_medium_positive_skew", "medium_sharp_positive_skew"]
samples_per_condition = 1000
samples_for_ambiguous = 100

class_dict = get_index_per_class(amplitude_conditions,time_constant_conditions, ambiguous_conditions, samples_per_condition, samples_for_ambiguous)


import matplotlib.pyplot as plt
# tau_medium: 0.007 - 0.014
# tau_small: 0.001 - 0.003 --> wide
# tau_large: 0.02 - 0.5 --> sharp


#X = np.round(np.linspace(0,3500,3501))
#mu = 1750
#V_ald = np.vectorize(asymmetric_laplace_distribution)
#lam = sample_amplitude(8, 10)


#tau1 = sample_tau(lower = 0.007, upper=0.014)
#tau2 = sample_tau(lower = 0.007, upper=0.014)

#f = V_ald(X, mu=mu, lam=lam, tau1=tau1, tau2=tau2)
#noise = np.random.normal(0, 0.12, f.shape)

amplitude_dict = {"S": "Small", "M":"Medium", "S/M": "Small/Medium", "L":"Large", "M/L": "Medium/Large"}

for key in list(class_dict.keys()):
    start = class_dict[key][0]
    end =  class_dict[key][1]
    class_i = data[start:end]

    fig,ax = plt.subplots()
    for i in class_i:
        ax.plot(i)
    ax.set_xlabel("Time", fontsize=10, labelpad=8)

    amplitude, time_course = str(key).split("-")
    time_course_title = [x.capitalize() for x in time_course.split("_")]

    if amplitude in ambiguous_conditions or time_course in ambiguous_conditions:
        title = ("Generated Clusters \n Ambiguous Data n = 100 \n Amplitude: %s - Time Course: " + (" ").join(['%s'] * len(time_course_title))) % tuple([amplitude_dict[amplitude]] + time_course_title)

    else:
        title = ("Generated Clusters \n n = 1000 \n Amplitude: %s - Time Course: " + (" ").join(['%s'] * len(time_course_title))) % tuple([amplitude_dict[amplitude]] + time_course_title)


    ax.set_title(title, fontsize = 12, pad=10)
    ax.set_ylim((0,16))

    fig.savefig(key.replace("/","_")+".png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(np.mean(class_i, axis=0))
    ax.set_xlabel("Time", fontsize=10, labelpad=8)
    title_splitted = title.split("\n")
    title_splitted[0] = title_splitted[0] + "(Mean)"
    ax.set_title("\n".join(title_splitted), fontsize=12, pad=10)
    ax.set_ylim((0, 16))

    fig.savefig(key.replace("/","_") + "_mean.png", bbox_inches="tight")
    plt.close()





