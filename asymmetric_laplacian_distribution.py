import numpy as np
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

# tau_small: 0.001 - 0.003
# tau_large: 0.02 - 0.5



def sample_small_tau(lower=0.001, upper=0.003):
    return np.random.uniform(lower, upper)


def sample_large_tau(lower = 0.02, upper = 0.5):
    return np.random.uniform(lower, upper)


def sample_amplitude(lower, upper):
    return np.random.uniform(lower, upper)


def generate_ALD(X, mu, amplitude_condition, time_constant_condition):
    V_ald = np.vectorize(asymmetric_laplace_distribution)

    if amplitude_condition == "S":
        lam = sample_amplitude(1, 2)

    elif amplitude_condition == "M":
        lam = sample_amplitude(5, 7)

    elif amplitude_condition == "L":
        lam = sample_amplitude(12, 14)
    else:
        print("Invalid amplitude condition, please select one of [S,M,L]...")

    if time_constant_condition == "equal_sharp":
        tau1 = sample_large_tau(lower = 0.02, upper=0.5)
        tau2 = sample_large_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "equal_wide":
        tau1 = sample_small_tau(lower=0.001, upper=0.003)
        tau2 = sample_small_tau(lower=0.001, upper=0.003)

    elif time_constant_condition == "negative_skew":
        tau1 = sample_small_tau(lower=0.001, upper=0.003)
        tau2 = sample_large_tau(lower = 0.02, upper=0.5)

    elif time_constant_condition == "positive_skew":
        tau1 = sample_large_tau(lower = 0.02, upper=0.5)
        tau2 = sample_small_tau(lower=0.001, upper=0.003)
    else:
        print("Invalid time_constant condition, please select one of [equal,negative_skew,positive_skew]...")

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



V_ald = np.vectorize(asymmetric_laplace_distribution)
X = np.round(np.linspace(0,3500,3501))
np.random.seed(42)

amplitude_conditions = ["S", "M", "L"]
time_constant_conditions = ["equal_sharp","equal_wide", "negative_skew", "positive_skew"]
samples_per_condition = 1000
mu = 1750


conditions = []
tau1s = []
tau2s = []
lams = []
F_signal = []
F_signal_noise = []
noises = []

for amplitude_condition in amplitude_conditions:
    for time_constant_condition in  time_constant_conditions:
        for i in range(samples_per_condition):
            conditions.append(amplitude_condition + "/" + time_constant_condition)
            f_i, tau1, tau2, lam = generate_ALD(X,mu=mu,amplitude_condition=amplitude_condition, time_constant_condition=time_constant_condition)
            if amplitude_condition == "S":
                noise = np.random.normal(0, 0.02, f_i.shape)
            elif amplitude_condition == "M":
                noise = np.random.normal(0, 0.07, f_i.shape)
            elif amplitude_condition == "M":
                noise = np.random.normal(0, 0.14, f_i.shape)


            F_signal.append(f_i)
            F_signal_noise.append(f_i+noise)
            noises.append(noise)
            tau1s.append(tau1)
            tau2s.append(tau2)
            lams.append(lam)
