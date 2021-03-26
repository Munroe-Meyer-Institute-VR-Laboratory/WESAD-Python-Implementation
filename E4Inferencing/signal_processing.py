import numpy as np
from scipy.sparse import spdiags


def tarvainen_detrending(signal_lambda, filter_matrix=[1, -2, 1], input_signal=np.random.normal(loc=20, scale=20, size=(50, 1))):
    """A time-varying finite-impulse-response high-pass filter for detrending
    If using this in a published study, cite:
        Tarvainen, Mika P., Perttu O. Ranta-Aho, and Pasi A. Karjalainen. "An advanced detrending method with
            application to HRV analysis." IEEE Transactions on Biomedical Engineering 49.2 (2002): 172-175.
    Parameters
    :param signal_lambda: int
        Smoothness parameter, the higher the value the smoother the SCL will be
    :param filter_matrix: ndarray
        Filter weights to be distributed on the D2 matrix
    :param input_signal: ndarray
        Input signal that has a trend
    :return:
    """
    input_length = len(input_signal)
    input_identity = np.identity(input_length)
    output_filter = filter_matrix * np.ones((1, input_length - 2)).T
    z_d2 = spdiags(output_filter.T, (range(0, 3)), input_length - 2, input_length)
    input_detrended = np.inner((input_identity - np.linalg.inv(input_identity + np.square(signal_lambda) * z_d2.T * z_d2)), input_signal.T)
    return np.squeeze(np.asarray(input_detrended))


def get_residual(trended_signal, signal_mean):
    """Simple function to return the residual from the EDA signal, can be used for any signal
    Parameters
    :param eda_signal: ndarray
        The original signal, should be windowed
    :param scr_signal: ndarray
        The trend (SCR) of the signal retrieved using tarvainen_detrending()
    :return: ndarray
        The residual (SCL) for the signal
    """
    return trended_signal - signal_mean


def split_set(signal, split_size):
    """Split a signal into specified number of chunks.  Handles uneven splits. Handled by a generator.
    Very useful for pulling apart a dataaset to calculate features since you don't need the separated dataset.
    Encapsulate result in list: list(split_set(signal, split_size)) if you don't want to use the generator
    Source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Parameters
    :param signal: ndarray
        Signal to be split
    :param split_size: int
        Size of chunks
    :return: generator
        Returns an iterable object to perform this function real-time
    """
    for i in range(0, len(signal), int(split_size)):
        yield signal[i:i + int(split_size)]
