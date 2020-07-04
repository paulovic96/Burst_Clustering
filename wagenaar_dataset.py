import os
#os.environ['OPENBLAS_NUM_THREADS'] ='2'
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
import pandas as pd
#import nest
import struct
import json
import sys
import seaborn as sns
from scipy import signal, stats
import subprocess

WAGENAAR_FILE_NAME = "http://neurodatasharing.bme.gatech.edu/development-data/html/wget/%s.%s.%s.%s.0.0.%d.list"

#Full Info: http://neurodatasharing.bme.gatech.edu/development-data/html/wget/daily.spont.dense.full.0.0.20.list
#Compact Matlab: http://neurodatasharing.bme.gatech.edu/development-data/html/wget/daily.spont.dense.mat.0.0.20.list
#Compact txt: http://neurodatasharing.bme.gatech.edu/development-data/html/wget/daily.spont.dense.text.0.0.20.list

def download_wagenaar_data(day=20, time="daily", stimulus = "spont", density="dense", format="full"):
    """
    :param day: day of recording
    :param time: daily or overnight observation
    :param stimulus: spont (spontaneuous) or stim (stimulus-evoked) recordings
    :param density: on of dense • small • sparse • small and sparse • ultra sparse
    :param format: full (full info), mat (compact (matlab)), text (compact (text))
    :return:
    """
    path = os.getcwd()
    print("The current working directory is %s" % path)

    path = "Data/raw_Data"
    if os.path.isdir(path):
        print("%s directory already exists!" % path)

    else:
        if os.path.isdir("Data"):
            os.mkdir(path)
        else:
            os.makedirs(path)
        print("Successfully created the directory %s" % path)

    url = WAGENAAR_FILE_NAME % (time, stimulus, density, format, day)
    print("start downloading data from...\n %s" % url)


    bashCommand = "wget -i %s" % url
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = "mv *.spike.gz %s" % path
    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    bashCommand = "rm %s.%s.%s.%s*" % (time,stimulus,density,format)
    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    bashCommand = "gunzip %s/*.gz" % path
    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    print("Done!!")

def calc_spike_counts(spike_train, interval, starting_zero=True):
    """
    Produces the spike counts array of this spike train, given the interval
    :param interval: counting interval, in seconds
    :return: an array of spike counts
    """
    if (len(spike_train) > 0):
        latest_spike = max(spike_train)
        if starting_zero:
            earliest_spike = 0
        else:
            earliest_spike = min(spike_train)
        counts = np.diff([np.count_nonzero(spike_train < t) for t in np.arange(earliest_spike, latest_spike + interval,
                                                                               interval)])  # count_nonzero counts all spikes occuring in intervals prev to current one --> difference to get spike counts per interval
    else:
        counts = []
    return counts


def calc_inter_spike_interval(spike_train):
    isi = np.diff(spike_train)
    return isi

def sqrt_transformation(burst_batch):
    transformed_batch = []
    for burst in burst_batch:
        transformed_burst = []
        for channel in range(burst.shape[0]):
            transformed_burst.append(np.sqrt(burst[channel]))
        transformed_batch.append(np.asarray(transformed_burst))
    return transformed_batch

def load_full_spike_info(filename, n=None):
    """ Load full-info files from  D. A. Wagenaar, J. Pine, and S. M. Potter, 2006 data set

    Args:
        filename (string): path to filename.spike
        n (int): Info to the first n spikes

    Returns:
        Spikes (dict): A dictonary of arrays with spike information
    """
    spike_data = {}

    """
    Extract spike information stored as struct of type:

    typedef struct {
        unsigned long long time; /* 8-byte integer, little-endian */
        short channel; /* 2-byte integer, little-endian */
        short height;
        short width;
        short context[74];
        short threshold;
    } Spikeinfo;
    """

    file_length_in_bytes = os.path.getsize(filename)
    max_n = int(file_length_in_bytes / 164)  # spike information stored in 164 bytes
    # print(max_n)
    # print(n is None)
    if n is None:
        n = max_n
    elif n > max_n:
        print("File contains info on %d spikes max" % max_n)
        n = max_n

    # information of culture and day
    info = filename.split("/")[-1].split(".")[0]
    culture = info.split("-")[0] + "." + info.split("-")[1]
    day = info.split("-")[2]

    spike_data["culture"] = [culture] * n
    spike_data["day"] = [day] * n

    with open(filename, 'rb') as file:
        raw = file.read()
        file.close()

        # print(n)
    for i in range(n):
        unpacked1 = np.asarray(struct.unpack('<Q', raw[i * 164:i * 164 + 8]))  # 8-byte integer, little-endian
        unpacked2 = np.asarray(
            struct.unpack('<hhh74hh', raw[i * 164 + 8:i * 164 + 8 + 156]))  # 156-byte integer, little-endian
        time = unpacked1[0]
        channel = unpacked2[0]
        height = unpacked2[1]
        width = unpacked2[2]
        context = unpacked2[3:77]  # context[74]
        threshold = unpacked2[77]
        if i == 0:
            spike_data["time"] = [time]
            spike_data["channel"] = [channel]
            spike_data["height"] = [height]
            spike_data["width"] = [width]
            # spike_data["context"] = [context]
            spike_data["threshold"] = [threshold]
        else:
            spike_data["time"].append(time)
            spike_data["channel"].append(channel)
            spike_data["height"].append(height)
            spike_data["width"].append(width)
            # spikes["context"].append(context)
            spike_data["threshold"].append(threshold)

    spike_data["time"] = np.asarray(spike_data["time"])
    spike_data["channel"] = np.asarray(spike_data["channel"])
    spike_data["height"] = np.asarray(spike_data["height"])
    spike_data["width"] = np.asarray(spike_data["width"])
    # spike_data["context"] = np.asarray(spike_data["context"])
    spike_data["threshold"] = np.asarray(spike_data["threshold"])

    # convert to units
    # Time is the time of the spike, in units of the sample period (40 μs) --> seconds .
    # Channel is the electrode channel on which the spike occurred. Channels 0..59 are actual electrodes; channel 60 is used to mark the occurrence of stimuli.
    # Height is the amplitude of the detected spike, in digital units, i.e. 0.33 μV --> μV.
    # Width is the duration of the spike, in units of the sample period (40 μs) --> ms.
    # Context is 74 samples worth of context, from 1 ms before the peak to 2 ms after the peak of the spike, in digital units, 0.33 μV per step. Sampling frequency is 25 kHz.
    # Threshold is the spike detection threshold in force at the time of this detection.
    # spikes["time"] = np.asarray(spikes["time"])/25000
    # spikes["height"] = np.asarray(spikes["time"])
    r = 683
    auxrange = r * 1.2
    freq = 25.0

    # isaux = np.where(np.asarray(spikes["channel"])>=0)
    iselc = np.where(spike_data["channel"] < 60)

    spike_data["height"][iselc] = spike_data["height"][iselc] * r / 2048
    spike_data["threshold"][iselc] = spike_data["threshold"][iselc] * r / 2048
    # spike_data["context"][iselc] = spike_data["context"][iselc] * r/2048

    # spike_data["height"][isaux] = np.asarray(spike_data["height"])[isaux] * auxrange/2048
    # spike_data["threshold"][isaux] = np.asarray(spike_data["threshold"])[isaux] * auxrange/2048
    # spike_data["context"][isaux] = np.asarray(spike_data["context"])[isaux] * auxrange/2048

    spike_data["time"] = spike_data["time"] / (freq * 1000)
    spike_data["width"] = spike_data["width"] / (freq)

    return pd.DataFrame.from_dict(spike_data)



def burstlets_detection(spike_data):
    """ Burst Detection
        - ≥ 2 channels active at the same time with overlapping burstlets
        - sequence of one or more burstlets with non-zero temporal overlap
        Burstlets:
        - sequences ≥ 4 spikes
        - inter-spike intervals < threshold
        - threshold = 1/4 * 1/average spike detection rate, or 100 ms if the average spike detection rate < 10 Hz
        - 'core' burstlet + past and future spikes ISIs < min(200,1/3 * 1/average spike detection rate)

    """
    # burstlets: sequences of at least four spikes
    # - inter-spike intervals < threshold
    # - threshold = 1/4 * 1/average spike detection rate,
    #               or 100 ms if the average spike detection rate < 10 Hz

    # channel spike-trains, inter spike intervals and thresholds
    channels = spike_data["channel"].unique()
    burstlets_detection_thresholds = []
    burstlets_context_thresholds = []
    inter_spike_intervals = []
    spike_trains = []
    for i in channels:
        # spike train for each channel
        spike_train = spike_data[spike_data["channel"] == i]['time']
        spike_trains.append(np.asarray(spike_train))

        # spike count for active time
        firing_rate = len(spike_train) / (max(spike_train) - min(spike_train))
        # threshold
        threshold = 1 / firing_rate * 0.25 # ensures suceeding spikes faster 4 * average firing rate considered burstlets
        threshold = min(0.1, threshold)

        context_threshold = 1 / (3 * firing_rate)
        context_threshold = min(0.2, context_threshold)

        burstlets_detection_thresholds.append(threshold)
        burstlets_context_thresholds.append(context_threshold)

        # inter spike interval for each channel
        isi = calc_inter_spike_interval(spike_train)
        inter_spike_intervals.append(isi)

    # spikes sequences with inter-spike < threshold
    channel_burstlet_candidates = []  # spike sequences with inter-spike < threshold for each channel
    channel_burstlet_candidates_with_context = []

    for i, isi in enumerate(inter_spike_intervals):  # for inter spike intervals of each channel
        burstlet_candidates = []  # seen sequences
        burstlet_candidates_with_context = []

        current_core_seq = []  # current sequence to check
        current_context_seq = []
        for j, inter in enumerate(isi):  # go through isi of spike train in channel

            if inter <= burstlets_detection_thresholds[i]:
                current_core_seq.append(spike_trains[i][j])  # spike time in current sequence

            if inter <= burstlets_context_thresholds[i]:
                current_context_seq.append(spike_trains[i][j])

            else:
                current_context_seq.append(spike_trains[i][j])
                burstlet_candidates_with_context.append(current_context_seq)
                current_context_seq = []

                current_core_seq.append(spike_trains[i][j])  # last spike time in current sequence
                burstlet_candidates.append(current_core_seq)
                current_core_seq = []  # go on with and collect next sequence

        channel_burstlet_candidates.append(burstlet_candidates)
        channel_burstlet_candidates_with_context.append(burstlet_candidates_with_context)

        # print(len(burstlet_candidates),len(burstlet_candidates_with_context))

    # sequences with at least four spikes
    burstlets = []
    burstlets_with_context = []

    for i, sequences in enumerate(channel_burstlet_candidates):
        candidates_with_context = channel_burstlet_candidates_with_context[i]
        for j, seq in enumerate(sequences):  # go thourgh each sequence and check wether more than 4 spikes included
            if len(seq) >= 4:
                burstlets.append([seq[0], seq[-1]])  # start and end point
                burstlet_seq_with_context = candidates_with_context[j]
                burstlets_with_context.append([burstlet_seq_with_context[0], burstlet_seq_with_context[-1]])

    burstlets_in_time = np.sort(np.asarray(burstlets), axis=0)
    burstlets_with_context_in_time = np.sort(np.asarray(burstlets_with_context), axis=0)

    return burstlets_in_time, burstlets_with_context_in_time


def burst_detection(burstlets):
    bursts = []
    j = 1
    count_burstlets = 1

    current_burstlet = burstlets[0]
    next_burstlet = burstlets[j]
    start_time = current_burstlet[0]
    end_time = current_burstlet[1]

    while (j < len(burstlets) - 1):
        if next_burstlet[0] <= end_time:  # burstlet starts before current ends --> ongoing
            count_burstlets += 1
            end_time = max(next_burstlet[1], end_time)  # set new end
            j += 1
            next_burstlet = burstlets[j]
        else:
            bursts.append([start_time, end_time, count_burstlets])  # end of burstlet sequence
            count_burstlets = 1  # restet count
            current_burstlet = next_burstlet  # set current burstlet
            j += 1
            next_burstlet = burstlets[j]  # set next burstlet
            start_time = current_burstlet[0]  # set new starting time of potential burst
            end_time = current_burstlet[1]  # set new end time

    bursts = np.asarray(bursts)
    detected_bursts = bursts[
        np.where(bursts[:, 2] > 1)]  # more than 2 channels active at the same time with overlapping burstlets
    return detected_bursts


def verify_detected_bursts(spike_data, detected_bursts):
    bins = np.arange(0, max(spike_data['time']) + 1, 1)
    sc, _ = np.histogram(spike_data['time'], bins=bins)

    plt.figure(figsize=(15, 5))
    plt.plot(detected_bursts[:, 0], detected_bursts[:, 2], "x", markersize=10)
    plt.plot(sc)
    plt.xlim([0, 1000])
    #plt.yscale("log")


def extract_burst_data(spike_data, detected_bursts, include_context=True):
    # add start and end point of the recording for inter burst interval calculation
    flattened = np.insert(detected_bursts, 0, [0, 0, 0], axis=0)
    flattened = np.insert(flattened, len(flattened), [max(spike_data['time']), max(spike_data['time']), 0], axis=0)
    flattened = flattened[:, 0:2].flatten()

    interburst_interval = np.diff(flattened)[1::2]
    if include_context:
        min_interburst_interval = min(interburst_interval)
        print("Minimum Interburst Interval: ", min_interburst_interval)
    else:
        min_interburst_interval = 0

    for i, detected_burst in enumerate(detected_bursts):
        interval_before = detected_burst[0] - min_interburst_interval  # min(0.2, interburst_interval[i])
        interval_after = detected_burst[1] + min_interburst_interval  # min(0.2, interburst_interval[i+1])
        if i == 0:
            burst_data = spike_data[
                (interval_before <= spike_data['time']) & (spike_data['time'] <= interval_after)]
            burst_data = burst_data.assign(burst=i)
        else:
            data_i = spike_data[(interval_before <= spike_data['time']) & (spike_data['time'] <= interval_after)]
            data_i = data_i.assign(burst=i)
            burst_data = burst_data.append(data_i, ignore_index=True)
    return burst_data


def bin_burst_data(burst_data, channels=60, binsize=0.01):
    burst_batch = []
    too_short_burst_indices = []
    counter = 0

    for i, burst in enumerate(burst_data["burst"].unique()):
        spike_train = burst_data[burst_data["burst"] == burst][["time", "channel"]]
        burst_beginning = min(spike_train["time"])
        burst_ending = max(spike_train["time"])
        if np.abs(burst_ending - burst_beginning) < binsize:
            counter += 1
            print(counter, "Found burst too short for binning (<%d msec): Duration = " % (binsize * 1000), np.abs(burst_ending - burst_beginning))
            too_short_burst_indices.append(i)
            continue
        length = np.ceil((burst_ending - burst_beginning) / binsize) + 1 # add another interval to include spike if it occures right at the end of the burst
        # test = np.arange(min(spike_train["time"]),max(spike_train["time"]), 0.02)
        # print(length, len(test))

        burst_array = np.zeros(shape=(channels, int(length)))
        # print(burst_array.shape)
        for channel in spike_train["channel"].unique():
            current_channel = spike_train[spike_train["channel"] == channel]["time"]
            start = int(np.ceil((min(current_channel) - burst_beginning) / binsize))
            spike_count = calc_spike_counts(current_channel, binsize, starting_zero=False)
            burst_array[int(channel), start:start + len(spike_count)] = spike_count
        if np.count_nonzero(burst_array) > 0:
            burst_batch.append(burst_array)
        else:
            print(current_channel)
            print(min(current_channel))
            print(start,burst_beginning,np.ceil((min(current_channel) - burst_beginning)))
            print(spike_count)
            counter += 1
            print(counter, "Found flat burst: Duration = ", np.abs(burst_ending - burst_beginning))
            too_short_burst_indices.append(i)
    return burst_batch, too_short_burst_indices


def burst_batch_padding(burst_batch, channels=60, padding="onset"):
    durations = []
    # channel_averaged_bursts = []
    burst_peaks = []
    for burst in burst_batch:
        durations.append(burst.shape[1])
        channel_averaged_burst = np.mean(burst, axis=0)
        # channel_averaged_bursts.append(channel_averaged_burst)
        burst_peak = np.argmax(channel_averaged_burst)
        burst_peaks.append(burst_peak)

    max_duration = max(durations)
    longest_burst = np.argmax(durations)

    if padding == "onset":
        burst_batch_pad = np.zeros((len(burst_batch), channels, max_duration))
        for i, burst in enumerate(burst_batch):
            padding = max_duration - burst.shape[1]
            burst_batch_pad[i, :, :] = np.pad(burst, ((0, 0), (0, padding)), 'constant')
        return burst_batch_pad

    elif padding == "peak":
        latest_peak = max(burst_peaks)
        shifts = latest_peak - np.asarray(burst_peaks) + durations
        # print(latest_peak, np.argmax(burst_peaks))
        # print(max_duration, longest_burst)
        max_shifted = max(shifts)
        shifts = None
        durations = None

        burst_batch_pad = np.zeros((len(burst_batch), channels, max_shifted))
        center = latest_peak
        print("Peak centering at: ", center)
        print("Padding to length: ", max_shifted)
        for i, burst in enumerate(burst_batch):
            peak_position = burst_peaks[i]
            peak_shift = center - peak_position

            end_padding = max_shifted - (burst.shape[1] + peak_shift)
            # print(peak_shift,burst.shape[1], end_padding)
            burst_batch_pad[i, :, :] = np.pad(burst, ((0, 0), (peak_shift, end_padding)), 'constant')
        return burst_batch_pad, center
    else:
        print("No padding applied")
        return burst_batch


def save_burst_data_as_json(df, data_dir, filename):
    df.to_json(data_dir + "%s.JSON" % filename)

def save_burst_batch(batch, data_dir, filename):
    if isinstance(batch, np.ndarray):
        list_of_lists = batch.tolist()
    else:
        list_of_lists = []
        for i, burst in enumerate(batch):
            list_of_lists.append(burst.tolist())

    with open(data_dir + "%s.JSON" % filename, 'w') as filehandle:
        filehandle.write(json.dumps(list_of_lists))
    filehandle.close()

def load_burst_data_from_json(data_dir, filename):
    return pd.read_json(data_dir + "%s.JSON" % filename)

def load_burst_data_from_json(data_dir, filename):
    return pd.read_json(data_dir + "%s.JSON" % filename)

def load_burst_batch(data_dir, filename):
    with open(data_dir + '%s.JSON' % filename, 'r') as filehandle:
        return json.loads(filehandle.read())
    filehandle.close()






def data_preprocessing(spike_files, data_dir):
    culture_names = [x.split(".")[0].replace("-", "_") for x in spike_files]
    for i, file in enumerate(spike_files):
        spike_data = load_full_spike_info(data_dir + file)

        burstlets, burstlets_with_context = burstlets_detection(spike_data)
        detected_bursts_with_context = burst_detection(burstlets_with_context)

        # verify_detected_bursts(data_culture_full,detected_bursts_with_context)

        if len(detected_bursts_with_context) > 0:
            burst_data = extract_burst_data(spike_data, detected_bursts_with_context)
            # burst_data_no_tiny = extract_burst_data(spike_data, detected_bursts_no_tiny)
            # burst_data_tiny = extract_burst_data(spike_data, detected_bursts_tiny)
            # burst_data.head()

            burst_batch, too_short_burst_indices = bin_burst_data(burst_data, binsize=0.01)  # 10ms


            valid_binned_detected_bursts_with_context = np.delete(detected_bursts_with_context, too_short_burst_indices,axis=0)
            detected_bursts_tiny_index = np.where(valid_binned_detected_bursts_with_context[:, 2] < 5)
            #detected_bursts_no_tiny_index = np.where(valid_binned_detected_bursts_with_context[:, 2] >= 5)

            #print("Active Channels in Bursts: ", valid_binned_detected_bursts_with_context[:, 2])
            print("#Bursts: ", len(valid_binned_detected_bursts_with_context))
            print("Mean burst duration: ", np.mean(
                valid_binned_detected_bursts_with_context[:, 1] - valid_binned_detected_bursts_with_context[:, 0]))
            print("Variance of burst duration: ", np.var(
                valid_binned_detected_bursts_with_context[:, 1] - valid_binned_detected_bursts_with_context[:, 0]))

            #print(len(valid_binned_detected_bursts_with_context), np.amax(detected_bursts_tiny_index))

            save_burst_data_as_json(burst_data, data_dir, "burst_data_" + culture_names[i])
            save_burst_batch(burst_batch, data_dir, "burst_data_batch_" + culture_names[i])
            np.save(data_dir + 'burst_data_batch_tiny_index_' + culture_names[i], detected_bursts_tiny_index)


        else:
            print("No bursts found in %s" % culture_names[i])



def nested_ndarrays(data):
    return [np.asarray(x) for x in data]

def load_batch_files_with_number_bursts(data_dir, data_batch_names):
    data_batches = {}
    for data_name in data_batch_names:
        burst_data = load_burst_batch(data_dir,data_name)
        burst_data = nested_ndarrays(burst_data)
        #print(data_name,len(burst_data))
        data_batches[data_name] = burst_data
    return data_batches


def merge_data_batches_ordered(data_batches,day_wise = False):
    keys = np.sort(list(data_batches.keys()))
    if day_wise:
        culture_names = []
        days = np.sort(np.unique([x.split("_")[-1] for x in keys]))
        for day in days:
            day_i = keys[[day in culture for culture in keys]]
            culture_names += list(np.sort(day_i))
    else:
        culture_names = keys
    data = []
    burst_counts = []
    culture_bursts = {}
    for i,key in enumerate(culture_names):
        burst_count = len(data_batches[key])
        burst_counts.append(burst_count)
        culture_bursts[key] = burst_count
        print(key, burst_count)
        data += data_batches[key]
    print("#Cultures: ", len(culture_names), " #Bursts (total): ", np.sum(burst_counts))
    print(np.cumsum(burst_counts))
    return data, culture_bursts


def load_tiny_indices_per_culture(data_dir, tiny_indices_names, day_wise=False, nd_position=True):
    keys = np.sort(list(tiny_indices_names))
    if day_wise:
        culture_names = []
        days = np.sort(np.unique([x.split("_")[-1] for x in keys]))
        for day in days:
            day_i = keys[[day in culture for culture in keys]]
            culture_names += list(np.sort(day_i))
    else:
        culture_names = keys

    tiny_burst_counts = {}
    tiny_bursts_indices = {}

    for i, key in enumerate(culture_names):
        if nd_position:
            indices = np.load(data_dir + key + ".npy")[0]
        else:
            indices = np.load(data_dir + key + ".npy")
        burst_count = len(indices)
        tiny_burst_counts[key] = burst_count
        tiny_bursts_indices[key] = indices

        print(key, burst_count)

    print("#Cultures: ", len(culture_names), " #Tiny Bursts (total): ", np.sum(list(tiny_burst_counts.values())))
    print(np.cumsum(list(tiny_burst_counts.values())))
    return tiny_burst_counts, tiny_bursts_indices


def get_tiny_burst_indices_for_merged_data(culture_counts, tiny_bursts_indices):
    tiny_indices_per_culture = list(tiny_bursts_indices.values())
    found_bursts_per_culture = list(culture_counts.values())
    offset = 0
    tiny_bursts_in_data_indices = []

    for i, indices in enumerate(tiny_indices_per_culture):
        if i == 0:
            tiny_bursts_in_data_indices += list(indices)
        else:
            offset += found_bursts_per_culture[i - 1]
            shifted_indices = indices + offset
            tiny_bursts_in_data_indices += list(shifted_indices)
    return tiny_bursts_in_data_indices


def plot_data_burst_distribution(culture_counts, tiny_burst_counts):
    plt.close("all")
    # The position of the bars on the x-axis
    r = range(len(culture_counts.keys()))

    # Names of group and bar width
    keys = np.sort(list(culture_counts.keys()))

    names = ["sparse_" + ".".join(x.split('_')[-3:]) if x.find("Sparse") >= 0 else ".".join(x.split('_')[-3:]) for x in
             keys]
    bar_tiny = [tiny_burst_counts[key] for key in np.sort(list(tiny_burst_counts.keys()))]
    bar_no_tiny = [culture_counts[key.replace("_tiny_index", "")] - tiny_burst_counts[key] for key in
                   np.sort(list(tiny_burst_counts.keys()))]
    # bar = [culture_counts[key] for key in np.sort(list(culture_counts.keys()))]
    barWidth = 1

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(r, bar_no_tiny, edgecolor='white', width=barWidth, label="Bursts")
    ax.bar(r, bar_tiny, bottom=bar_no_tiny, edgecolor='white', width=barWidth, label="tiny Bursts",
           color="lightskyblue")

    # ax.bar(r, bar, edgecolor = 'white', width = barWidth, color = "dodgerblue")

    for i, p in enumerate(ax.patches):
        if len(str(p.get_height())) > 4:
            shift = 0
        elif len(str(p.get_height())) == 4:
            shift = 0.1
        elif len(str(p.get_height())) == 3:
            shift = 0.2
        elif len(str(p.get_height())) == 2:
            shift = 0.3
        else:
            shift = 0.4

        ax.annotate(str(p.get_height()), (p.get_x() + shift, p.get_y() + p.get_height() * 0.4), fontsize=10)
        if i >= int(len(ax.patches) / 2):
            if len(str(culture_counts[keys[i - len(keys)]])) > 4:
                shift = 0
            elif len(str(culture_counts[keys[i - len(keys)]])) == 4:
                shift = 0.1
            elif len(str(culture_counts[keys[i - len(keys)]])) == 3:
                shift = 0.2
            elif len(str(culture_counts[keys[i - len(keys)]])) == 2:
                shift = 0.3
            else:
                shift = 0.4
            ax.annotate(str(culture_counts[keys[i - len(keys)]]), (p.get_x() + shift, p.get_y() + p.get_height() + 100),
                        fontsize=10, fontweight='bold')

    # Custom X axis
    ax.set_xticks(r)
    ax.set_xticklabels(names, fontsize=9, rotation='vertical', fontweight='bold')
    ax.set_xlabel("Culture", fontsize=20, labelpad=10)
    ax.set_ylabel("#Bursts", fontsize=20, labelpad=10)
    ax.set_yticks(range(0, 4000, 500))
    ax.set_yticklabels(range(0, 4000, 500), fontsize=15)
    ax.set_title("Number of detected Bursts per Culture", fontsize=30, pad=20)
    # Show graphic
    ax.legend(fontsize=15)
    # ax.set_title("Burst per culture", fontsize = 40)

"""
data_dir = "data/raw_data/daily_spontanous_dense/day20/"
spike_files = [x for x in os.listdir(data_dir) if x.endswith(".spike")]
data_burst_batches = [x for x in os.listdir(data_dir) if x.find("burst_data_batch_") >= 0 and x.find("tiny") <0]
tiny_burst_indices = [x for x in os.listdir(data_dir) if x.find("burst_data_batch_tiny_index") >= 0]
data_batch_names = [x.split('.')[0] for x in data_burst_batches]
data_batch_names_for_tiny_indices = [x.split('.')[0] for x in tiny_burst_indices]


data_burst_batches = load_batch_files_with_number_bursts(data_dir, data_batch_names)
data, culture_counts = merge_data_batches_ordered(data_burst_batches, day_wise = False)

tiny_burst_counts, tiny_bursts_indices = load_tiny_indices_per_culture(data_dir, data_batch_names_for_tiny_indices, day_wise = False)
tiny_bursts_in_data_indices = get_tiny_burst_indices_for_merged_data(culture_counts, tiny_bursts_indices)


padded_data, data_center = burst_batch_padding(data, padding = "peak")

data_burst_by_time = np.mean(padded_data,axis = 1).T
#data_burst_by_time_shuffled = (np.random.permutation(data_burst_by_time.T)).T
print("Burst data Batch: ", padded_data.shape)
print("Averaged over channels: ", data_burst_by_time.shape)
print("Centered at: ", data_center)

#np.save(data_dir + 'padded_data_day_20.npy', padded_data)
#np.save(data_dir + 'data_burst_by_time_day_20.npy', data_burst_by_time)

tiny_bursts = data_burst_by_time.T[tiny_bursts_in_data_indices]
no_tiny_bursts = np.delete(data_burst_by_time.T,tiny_bursts_in_data_indices,axis = 0)


np.random.seed(2)
np.random.shuffle(tiny_bursts)
np.random.shuffle(no_tiny_bursts)

plt.figure(figsize=(20, 10))
for burst in no_tiny_bursts[0:200]:
    plt.plot(burst)
plt.xlabel("Time (10ms bins)", fontsize=15, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel("Spike Count", fontsize=15, labelpad=10)
plt.title("Burst Examples averaged over Channels", fontsize=20, pad=20)
# plt.xlim((7500,11000))
plt.show()
"""
