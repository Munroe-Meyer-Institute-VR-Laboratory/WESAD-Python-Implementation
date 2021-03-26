import glob
import os
import pickle
import scipy.signal as signal
import numpy as np
import csv


def save_dataset(subject_data):
    data_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
    for sub in range(len(subject_data)):
        with open(subject_data[sub]['subject'] + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for data in data_keys:
                print("Writing line: ", data)
                writer.writerows([np.asarray(subject_data[sub]['signal']['wrist'][data].flatten())])
            writer.writerows([np.asarray(subject_data[sub]['label'].flatten())])

def save_features(list):
    with open('extracted_features.pkl', 'wb') as f:
        pickle.dump(list, f)


def load_features():
    with open('extracted_features.pkl', 'rb') as f:
        dataset, labels = pickle.load(f)
    return dataset, labels


def load_dataset(parent_dir=r'D:\Datasets\WESAD\\'):
    """Recursive function to load pickled WESAD dataset into a dictionary
    Parameters
    :param parent_dir:
    :return:
    """
    datasets_from_dir = []
    unpickled_datasets = []

    for filename in glob.iglob(parent_dir + '**/*', recursive=True):
        if filename.endswith(".pkl"):
            datasets_from_dir.append(filename)

    for filename in datasets_from_dir:
        print("Processing file: " + filename + "...")
        unpickled_datasets.append(pickle.load(open(filename, mode='rb'), encoding='latin1'))

    return unpickled_datasets


def resample_data(ACC, EDA, labels, new_length):
    """Resamples the passed signals to the specified length using signal.resample
    TODO: Should be generalized to use a list of signals
    Parameters
    :param ACC:
    :param EDA:
    :param labels:
    :param new_length:
    :return:
    """
    new_ACC = signal.resample(ACC, new_length)
    new_EDA = signal.resample(EDA, new_length)
    new_label = np.around(signal.resample(labels, new_length)).clip(min=0, max=7)
    return new_ACC, new_EDA, new_label
