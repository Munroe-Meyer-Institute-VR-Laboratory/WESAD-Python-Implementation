from scipy.stats import linregress, pearsonr
from scipy import integrate
import numpy as np
from scipy.signal import find_peaks
import statistics
import math
from filters import butter_bandpass_filter, get_energy
import heartpy as hp


def signal_peak_frequency(sampled_signal, sampling_rate):
    """Finds the peak FFT frequency band in Hertz
    Source: https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    :param sampled_signal: ndarray
        The signal used to find the peak frequency component
    :param sampling_rate: float
        The sampling rate of the signal when captured
    :return: float
        The peak frequency component in Hertz
    """
    w = np.fft.fft(sampled_signal)
    freqs = np.fft.fftfreq(len(sampled_signal))
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * sampling_rate)
    return freq_in_hertz


def signal_mean(sampled_signal):
    """Calculates the simple geometric mean of a signal
    Parameters
    :param sampled_signal: ndarray
        Signal used to calculate mean
    :return: float
        Geometric mean of signal
    """
    return np.mean(sampled_signal)


def signal_standard_deviation(sampled_signal):
    """Calculates the standard deviation of the signal passed
    Parameters
    :param sampled_signal: ndarray
        Signal used to calculate the standard deviation
    :return: float
        Standard deviation of signal
    """
    return statistics.stdev(sampled_signal)


def signal_integral(sampled_signal):
    """Calculates the integral of the signal using the composite Simpson's rule
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the area under the curve
    :return: float
        Area under the curve
    """
    return integrate.simps(sampled_signal)


def signal_absolute(integrated_signal):
    """Calculates the absolute value of a signal, meant for use after the integration of the signal
    Parameters
    :param integrated_signal: float
        Output of the signal_integration function
    :return: float
        The absolute value of the input
    """
    return np.abs(integrated_signal)


def signal_dyanmic_range(sampled_signal):
    """Calculates the dynamic range of a signal, which is the ratio between the highest value and the lowest value
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the dynamic range
    :return: float
        The dynamic range of the signal
    """
    return max(sampled_signal) / min(sampled_signal)


def signal_rms(sampled_signal):
    """Calculates the root mean square of a signal
    Source: https://stackoverflow.com/questions/40963659/root-mean-square-of-a-function-in-python
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the root mean square
    :return: float
        The root mean square of the signal
    """
    return np.sqrt(np.mean(sampled_signal**2))


def signal_frequency_band_energies(sampled_signal, frequency_bands, sampling_frequency, order=5):
    """Iterates through a list provided in frequency_bands to extract the energies in each band from sampled_signal
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the frequency band energies
    :param frequency_bands: list[[float, float]]
        List of frequency bands in the format [[low, high], [low, high],...,[low, high]]
    :param sampling_frequency: float
        The sampling frequency of the signal used to calculate the frequency band energies
    :param order: int
        The order of the butterworth filter to be used, defaults to 5
    :return: list[float]
        List appended with the calculated energies in the frequency bands of the form [float,...,float]
    """
    energies = []
    for bands in frequency_bands:
        energies.append(butter_bandpass_filter(sampled_signal, bands[0], bands[1], sampling_frequency, order))
    return energies


def signal_frequency_normalize(frequencies, axis=-1, order=2):
    """Normalizes the input frequency list in frequencies
    Source: https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    Parameters
    :param frequencies: list[float]
        List of frequency energies to be normalized
    :param axis: int
        Axis to be normalized on, default is -1
    :param order: int
        The order of the normalization, default is 2
    :return: list[float]
        List containing the normalized frequency energies
    """
    normalized_frequencies = []
    for frequency in frequencies:
        norm = np.atleast_1d(np.linalg.norm(frequency, order, axis))
        norm[norm == 0] = 1
        normalized_frequencies.append(frequency / np.expand_dims(norm, axis))
    return normalized_frequencies


def signal_relative_power(frequencies, scale):
    """Calculates the relative power in dB for the signal, centered around [-1, 1]
    Source: https://stackoverflow.com/questions/2445756/how-can-i-calculate-audio-db-level
    Parameters
    :param frequencies: list[float]
        List containing the frequency energies to be used in the calculation
    :param scale: list[int]
        List containing the high and low range for the signal, i.e. signed 8-bit scale is 128
    :return: list[float]
        List containing the relative powers of the frequencies passed in the same order
    """
    relative_power = []
    for frequency in frequencies:
        norm_freq = frequency / scale
        relative_power.append(20 * math.log10(max(norm_freq) + 1e-7))
    return relative_power


def signal_slope(sampled_signal, time_labels):
    """Calculates the slope of the signal using linear regression
    Source: https://stackoverflow.com/questions/9538525/calculating-slopes-in-numpy-or-scipy
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the slope
    :param time_labels: ndarray
        For time-series data, using np.arange(0, len(sampled_signal)) should work
    :return: float
        Slope of the signal
    """
    slope, intercept, r_value, p_value, std_err = linregress(sampled_signal, time_labels)
    return slope


def signal_percentile(sampled_signal, percentile):
    """Calculates the specified q-th percentile of the signal
    Source: https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the percentile
    :param percentile: ndarray
        Percentile or sequence of percentiles to compute, [0, 100]
    :return: scalar or ndarray
        Percentiles of the signal
    """
    return np.percentile(sampled_signal.reshape(-1), percentile)


def signal_correlation(sampled_signal, time_labels):
    """Calculates the Pearson R Coefficient of the signal against time
    Source: https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
    Parameters
    :param sampled_signal: ndarray
        The signal used to calculate the Pearson R Coefficient
    :param time_labels: ndarray
        Array of labels corresponding to each sample, np.arange(0, len(sampled_signal)) can be used
    :return: float, float
        Pearson R Coefficient of the signal, two-tailed p-value
    """
    r, _ = pearsonr(sampled_signal, time_labels)
    return r


def signal_min_max(sampled_signal):
    """Returns the min and max of the signal
    Parameters
    :param sampled_signal: ndarray
        Signal used to get the min and max
    :return: scalar, scalar
        Returns the max and min of signal in form [max, min]
    """
    return [max(sampled_signal), min(sampled_signal)]


def signal_peak_count(sampled_signal):
    """Counts the number of peaks in the signal
    Parameters
    :param sampled_signal: ndarray
        Signal used to find the peakr
    :return: ndarray
        Returns an array of indexes corresponding to the peaks in the signal
    """
    peaks, _ = find_peaks(sampled_signal)
    return peaks


def signal_frequency_summation(energies):
    """Performs a summation of the frequencies passed in the list energies
    Parameters
    :param energies: list[float]
        List containing the energies in the frequency bands of a signal
    :return: list[float]
        Returns a list of the summed energies in the same order
    """
    band_energies = []
    for energy in energies:
        band_energies.append(get_energy(energy))
    return band_energies


def signal_lf_hf_ratio(lf, hf):
    """Computes the ratio between the high and low frequency components of a signal
    Parameters
    :param lf: scalar
        Summation of low frequency energies in a signal
    :param hf: scalar
        Summation of high frequency energies in a signal
    :return: scalar
        Ratio of the high and low frequency energies in a signal
    """
    return lf / hf


def bvp_signal_processing(bvp_signal, sampling_frequency):
    """Simple function wrapping the HeartPy process method for BVP signal metric extraction
    Parameters
    :param bvp_signal: ndarray
        BVP signal used to get metrics, run bvp_signal.reshape(-1) to make it a (1,n) array
    :param sampling_frequency: float
        The sampling frequency used when capturing the signal
    :return: ndarray, dict
        Returns two variables, the output signal to be used in the hp.plotter method and the metrics in a dict
    """
    try:
        _, m = hp.process(bvp_signal, sample_rate=sampling_frequency)
    except:
        NaN = float('nan')
        m = {'bpm': NaN, 'ibi': NaN, 'sdnn': NaN, 'sdsd': NaN, 'rmssd': NaN, 'pnn20': NaN, 'pnn50': NaN, 'hr_mad': NaN, 'sd1': NaN, 'sd2': NaN, 's': NaN, 'sd1/sd2': NaN, 'breathingrate': NaN}
    return m
