from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing, metrics
import numpy as np


def scale_labels(subject_labels):
    """Saves two lines of code by wrapping up the fitting and transform methods of the LabelEncoder
    Parameters
    :param subject_labels: ndarray
        Label array to be scaled
    :return: ndarray
        Scaled label array
    """
    encoder = preprocessing.LabelEncoder()
    _ = encoder.fit(subject_labels)
    return encoder.transform(subject_labels)


def trim_data(subject_data, subject_labels):
    """Trims the dataset of the unused labels and windows, prepares it to be converted to a binary and multi-class set
    Parameters
    :param subject_labels: ndarray
        Array of labels for the subject
    :param subject_data: ndarray
        Array for daya for the subject aligned with the labels
    :return: ndarray, ndarray
        Trimmed data and labels
    """
    del_indxs = np.hstack((np.where(subject_labels >= 4.0)[0], np.where(subject_labels == 0.0)[0]))
    del_labels = np.delete(subject_labels, del_indxs, 0)
    del_data = np.delete(subject_data, del_indxs, 0)
    return del_data, del_labels


def get_metrics(datasets, labels, dev_data, dev_labels):
    """Runs a dataset against five machine learning algorithms and returns common metrics for performance
    Parameters
    :param datasets: list
        List of data to be used in training
    :param labels: list
        List of labels to be used in training
    :param dev_data: ndarray
        Array of data used to get accuracy of model
    :param dev_labels: ndarray
        Array of labels used to get accuracy of model
    :return: list
        List of metrics in order of execution
    """
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=20),
                   RandomForestClassifier(min_samples_split=20, n_estimators=100),
                   AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(min_samples_split=20))]
    model_metrics = []
    for clf in classifiers:
        for x, y in zip(datasets, labels):
            print(clf, len(x), len(y))
            clf.fit(x, y)
            y_pred = clf.predict(dev_data)
            accuracy = metrics.accuracy_score(dev_labels, y_pred)
            confusion = metrics.confusion_matrix(dev_labels, y_pred)
            precision, recall, f1_score_weighted, _ = metrics.precision_recall_fscore_support(dev_labels, y_pred, average='weighted')
            model_metrics.append([accuracy, precision, recall, f1_score_weighted, confusion])
    return model_metrics


def run_simple_tests(datasets, labels):
    """Runs a dataset against five machine learning algorithms and returns a simple metric for performance.
    Parameters
    :param datasets: list
        List of ndarrays that contain the data
    :param labels: list
        List of ndarrays that contain the labels
    :return: list
        Returns a list of scores from the models in order
    """
    classifiers = [LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(9),
                   DecisionTreeClassifier(min_samples_split=20),
                   RandomForestClassifier(min_samples_split=20, n_estimators=100),
                   AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(min_samples_split=20))]
    score = []
    for clf in classifiers:
        clf.fit(datasets[0], labels[0])
        score.append(clf.score(datasets[1], labels[1]))
    return score
