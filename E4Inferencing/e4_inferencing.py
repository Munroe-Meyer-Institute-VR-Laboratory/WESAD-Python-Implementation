import wesad_interfacing as inf
import signal_processing as sp
import feature_extraction as fe
import machine_learning_utils as mlu
import csv
import numpy as np
from sklearn.impute import SimpleImputer


def run_testing(load_dataset=False, window_size=5.0):
    """Highest function call to run the pipeline from start to finish.
    Parameters
    :param load_dataset: bool
        Loads the pickled dataset if True
    """
    # Machine learning labels
    ml_algo = ['Linear Discriminant Analysis', 'K-Neighbors Classification',
               'Decision Tree Classification', 'Random Forest Classification',
               'AdaBoost Classification']
    # Perform pre-processing input pipeline
    if load_dataset:
        dataset, labels = inf.load_features()
    else:
        dataset, labels = windowed_e4_feature_extraction(window_size=window_size)
    # Setup variables
    multi_class_x, multi_class_y = [], []
    # Iterate through data, remove undefined labels, and add to multi class
    for data, label in zip(dataset, labels):
        del_data, del_labels = mlu.trim_data(data, label)
        multi_class_x.append(del_data)
        multi_class_y.append(mlu.scale_labels(del_labels))
    # Create binary class dataset
    bin_class_x = multi_class_x.copy()
    bin_class_y = multi_class_y.copy()
    # Combine amusement class with baseline class
    for i in range(0, len(bin_class_y)):
        bin_class_y[i][np.where(bin_class_y[i] == 2)] = 0
    # Perform inferencing on both multi-class and binary class problems
    multi_model_metrics = mlu.get_metrics([np.vstack((multi_class_x[0], multi_class_x[1]))],
                                          [np.hstack((multi_class_y[0], multi_class_y[1]))],
                                          multi_class_x[2], multi_class_y[2])
    bin_model_metrics = mlu.get_metrics([np.vstack((bin_class_x[0], bin_class_x[1]))],
                                        [np.hstack((bin_class_y[0], bin_class_y[1]))],
                                        bin_class_x[2], bin_class_y[2])
    # Print classification results
    print("Multi-Class WESAD Dataset Results")
    for m_metrics, algo in zip(multi_model_metrics, ml_algo):
        print(algo,
              "| Accuracy: {:.2%}".format(m_metrics[0]),
              "| F1-Score: {:.2%}".format(m_metrics[3]),
              "| Precision: {:.2%}".format(m_metrics[1]),
              "| Recall: {:.2%}".format(m_metrics[2]))
        print("Confusion Matrix")
        print(np.matrix(m_metrics[4]), '\n')
    print("Binary-Class WESAD Dataset Results")
    for m_metrics, algo in zip(bin_model_metrics, ml_algo):
        print(algo,
              "| Accuracy: {:.2%}".format(m_metrics[0]),
              "| F1-Score: {:.2%}".format(m_metrics[3]),
              "| Precision: {:.2%}".format(m_metrics[1]),
              "| Recall: {:.2%}".format(m_metrics[2]))
        print("Confusion Matrix")
        print(np.matrix(m_metrics[4]), '\n')


def windowed_e4_feature_extraction(window_size, train_portion=0.7, test_portion=0.2, dev_portion=0.1, write_csv=True, write_pickle=True):
    """Function to run through the entire WESAD dataset and extract the features.
    Parameters
    :param window_size: float
        Window size in seconds, such as one second = 1.0
    :param train_portion: float
        Portion of dataset to be used for training, default = 0.7
    :param test_portion: float
        Portion of dataset to be used for testing, default = 0.2
    :param dev_portion: float
        Portion of dataset to be used for development, default = 0.1
    :param write_csv: bool
        Indicate if features should be written out to CSV file
    :param write_pickle: bool
        Indicate if features should be written out to pickle file
    :return: list
        List of features
    """
    subject_data = inf.load_dataset()
    # Initialize return lists
    windowed_train_data = []
    windowed_train_labels = []
    windowed_test_data = []
    windowed_test_labels = []
    windowed_dev_data = []
    windowed_dev_labels = []
    # Initialize subject lists
    subject_data_list = []
    subject_label_list = []
    window_data = []
    window_label = []
    # Signal window sizes
    bvp_window_size = 64.0 * window_size
    acc_window_size = 32.0 * window_size
    eda_window_size = 4.0 * window_size
    temp_window_size = 4.0 * window_size
    label_window_size = 700.0 * window_size
    # Segment dataset types
    train_samples = int(np.round(len(subject_data) * train_portion))
    test_samples = int(np.round(len(subject_data) * test_portion))
    dev_samples = int(np.round(len(subject_data) * dev_portion))
    print("Train Samples:", train_samples, "| Test Samples:", test_samples, "| Dev Samples:", dev_samples)
    # Iterate through the dataset types
    print("Beginning train sample processing...")
    for train in range(0, train_samples):
        print("Processing subject number:", train)
        bvp_generator = sp.split_set(subject_data[train]['signal']['wrist']['BVP'], bvp_window_size)
        eda_generator = sp.split_set(subject_data[train]['signal']['wrist']['EDA'], eda_window_size)
        acc_generator = sp.split_set(subject_data[train]['signal']['wrist']['ACC'], acc_window_size)
        temp_generator = sp.split_set(subject_data[train]['signal']['wrist']['TEMP'], temp_window_size)
        label_generator = sp.split_set(subject_data[train]['label'], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            window_data.append(get_e4_features(acc, 'ACC'))
            window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
            subject_label_list.append(window_label)
            window_data = []
            window_label = []
        windowed_train_data.append(subject_data_list)
        windowed_train_labels.append(subject_label_list)
        subject_data_list = []
        subject_label_list = []
    for test in range(train_samples, train_samples + test_samples):
        print("Processing subject number:", test)
        bvp_generator = sp.split_set(subject_data[test]['signal']['wrist']['BVP'], bvp_window_size)
        eda_generator = sp.split_set(subject_data[test]['signal']['wrist']['EDA'], eda_window_size)
        acc_generator = sp.split_set(subject_data[test]['signal']['wrist']['ACC'], acc_window_size)
        temp_generator = sp.split_set(subject_data[test]['signal']['wrist']['TEMP'], temp_window_size)
        label_generator = sp.split_set(subject_data[test]['label'], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            window_data.append(get_e4_features(acc, 'ACC'))
            window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
            subject_label_list.append(window_label)
            window_data = []
            window_label = []
        windowed_test_data.append(subject_data_list)
        windowed_test_labels.append(subject_label_list)
        subject_data_list = []
        subject_label_list = []
    for dev in range(train_samples + test_samples, train_samples + test_samples + dev_samples):
        print("Processing subject number:", dev)
        bvp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['BVP'], bvp_window_size)
        eda_generator = sp.split_set(subject_data[dev]['signal']['wrist']['EDA'], eda_window_size)
        acc_generator = sp.split_set(subject_data[dev]['signal']['wrist']['ACC'], acc_window_size)
        temp_generator = sp.split_set(subject_data[dev]['signal']['wrist']['TEMP'], temp_window_size)
        label_generator = sp.split_set(subject_data[dev]['label'], label_window_size)
        for bvp, eda, acc, temp, label in zip(bvp_generator, eda_generator, acc_generator, temp_generator, label_generator):
            window_data.append(get_e4_features(bvp, 'BVP'))
            window_data.append(get_e4_features(eda, 'EDA'))
            window_data.append(get_e4_features(acc, 'ACC'))
            window_data.append(get_e4_features(temp, 'TEMP'))
            window_label.append(get_e4_labels(label))
            subject_data_list.append(window_data)
            subject_label_list.append(window_label)
            window_data = []
            window_label = []
        windowed_dev_data.append(subject_data_list)
        windowed_dev_labels.append(subject_label_list)
        subject_data_list = []
        subject_label_list = []
    print("Converting lists to arrays...")
    datasets = []
    datasets.append(windowed_train_data)
    datasets.append(windowed_test_data)
    datasets.append(windowed_dev_data)
    labels = []
    labels.append(windowed_train_labels)
    labels.append(windowed_test_labels)
    labels.append(windowed_dev_labels)
    datasets_array = []
    labels_array = []
    for dataset, label in zip(datasets, labels):
        x_train = np.empty((58,))
        y_train = np.empty((1,))
        for window, lbl in zip(dataset, label):
            for x, y in zip(window, lbl):
                data_window = np.hstack((np.array(x[0]), np.array(x[1]), np.array(x[2]), np.array(x[3])))
                data_window[np.isnan(data_window)] = 0
                data_window[np.isinf(data_window)] = 0
                data_label = np.array(y[0])
                data_label[np.isnan(data_label)] = 0
                data_label[np.isinf(data_label)] = 0
                x_train = np.vstack((x_train, data_window))
                y_train = np.vstack((y_train, data_label))
        datasets_array.append(x_train)
        labels_array.append(y_train)
    print("Imputing missing features...")
    labels_array[0][0] = 0.0
    datasets_array[0][0] = 0.0
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(datasets_array[0])
    imp.transform(datasets_array[0])
    imp.fit(labels_array[0])
    imp.transform(labels_array[0])
    imp = SimpleImputer(missing_values=np.inf, strategy='mean')
    imp.fit(datasets_array[0])
    imp.transform(datasets_array[0])
    imp.fit(labels_array[0])
    imp.transform(labels_array[0])
    # TODO: Fix formatting for CSV output. Currently outputs an unintelligible mess.
    if write_csv:
        print("Writing to CSV...")
        with open('extracted_features.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for data in range(len(datasets_array)):
                for i in range(len(datasets_array[data])):
                    print("Writing line: ", i)
                    writer.writerows([np.append(datasets_array[data][i], [labels_array[data][i]])])
    if write_pickle:
        print("Currently pickling...")
        inf.save_features([datasets_array, labels_array])
    return datasets_array, labels_array


def get_e4_features(signal, signal_type):
    """Processing pipeline per signal, pass in a slice of the signal that corresponds to the window.
    Specify the signal that is passed in.
    Parameters
    :param signal: ndarray
        Windowed slice of the original signal
    :param signal_type: str
        Signal type passed in.  BVP, Accelerometer, Temperature, and EDA supported
    :return: list
        List of the features extracted
    """
    features = []
    if signal_type == 'BVP':
        signal = signal.reshape(-1)
        m = fe.bvp_signal_processing(signal, 64.0)
        for key in m:
            features.append(m[key])
        frequency_energies = fe.signal_frequency_band_energies(signal, [[0.01, 0.04], [0.04, 0.15], [0.15, 0.4], [0.4, 1.0]], 64.0)
        norm_freq = fe.signal_frequency_normalize([frequency_energies[1], frequency_energies[2]])
        freq_sums = fe.signal_frequency_summation(frequency_energies)
        freq_powers = fe.signal_relative_power(frequency_energies, 1024 / 2)
        for freq in freq_sums:
            features.append(freq)
        for freq in freq_powers:
            features.append(freq)
        features.append(np.average(norm_freq[0]))
        features.append(np.average(norm_freq[1]))
        features.append(fe.signal_lf_hf_ratio(freq_sums[1], freq_sums[2]))
    elif signal_type == 'ACC':
        features.append(fe.signal_mean(signal[:, 0].reshape(-1)))
        features.append(fe.signal_mean(signal[:, 1].reshape(-1)))
        features.append(fe.signal_mean(signal[:, 2].reshape(-1)))
        features.append(fe.signal_mean(signal.flatten()))
        features.append(fe.signal_standard_deviation(signal[:, 0].reshape(-1)))
        features.append(fe.signal_standard_deviation(signal[:, 1].reshape(-1)))
        features.append(fe.signal_standard_deviation(signal[:, 2].reshape(-1)))
        features.append(fe.signal_standard_deviation(signal.reshape(-1)))
        features.append(fe.signal_absolute(fe.signal_integral(signal[:, 0].reshape(-1))))
        features.append(fe.signal_absolute(fe.signal_integral(signal[:, 1].reshape(-1))))
        features.append(fe.signal_absolute(fe.signal_integral(signal[:, 2].reshape(-1))))
        features.append(fe.signal_absolute(fe.signal_integral(signal.reshape(-1))))
        features.append(fe.signal_peak_frequency(signal[:, 0].reshape(-1), 32.0))
        features.append(fe.signal_peak_frequency(signal[:, 1].reshape(-1), 32.0))
        features.append(fe.signal_peak_frequency(signal[:, 2].reshape(-1), 32.0))
    elif signal_type == 'TEMP':
        signal = signal.reshape(-1)
        features.append(fe.signal_mean(signal))
        features.append(fe.signal_standard_deviation(signal))
        minmax = fe.signal_min_max(signal)
        features.append(minmax[0])
        features.append(minmax[1])
        features.append(fe.signal_dyanmic_range(signal))
        features.append(fe.signal_slope(signal, np.arange(0, len(signal))))
    elif signal_type == 'EDA':
        signal = signal.reshape(-1).ravel()
        scr = sp.tarvainen_detrending(signal_lambda=1500, input_signal=signal)
        scl = sp.get_residual(signal, scr)
        features.append(fe.signal_mean(signal))
        features.append(fe.signal_standard_deviation(signal))
        features.append(fe.signal_slope(signal, np.arange(0, len(signal))))
        features.append(fe.signal_dyanmic_range(signal))
        minmax = fe.signal_min_max(signal)
        features.append(minmax[0])
        features.append(minmax[1])
        features.append(fe.signal_mean(scl))
        features.append(fe.signal_mean(scr))
        features.append(fe.signal_standard_deviation(scl))
        features.append(fe.signal_standard_deviation(scr))
        features.append(fe.signal_correlation(scl, np.arange(0, len(scl))))
        features.append(len(fe.signal_peak_count(scr)))
        features.append(fe.signal_integral(scr))
    return features


def get_e4_labels(subject_labels):
    """Averages the labels of the window
    Parameters
    :param subject_labels: ndarray
        Labels to be averaged
    :return: float
        Averaged label value
    """
    return np.around(np.average(subject_labels))











