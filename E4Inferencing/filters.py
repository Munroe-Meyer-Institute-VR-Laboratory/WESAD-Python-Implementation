from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Returns a butterworth bandpass filter of the specified order and f range
    Source: https://stackoverflow.com/questions/30659579/calculate-energy-for-each-frequency-band-around-frequency-f-of-interest-in-pytho
    :param lowcut: scalar
        Low pass cutoff frequency
    :param highcut: scalar
        High pass cutoff frequency
    :param fs: float
        Sampling frequency of the signal
    :param order: int
        The order of the butterworth filter being constructed
    :return: ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a fifth-order butterworth bandpass filter to a signal in the specified frequency range
    Source: https://stackoverflow.com/questions/30659579/calculate-energy-for-each-frequency-band-around-frequency-f-of-interest-in-pytho
    :param data: ndarray
        Signal to be filtered
    :param lowcut: scalar
        Low pass cutoff frequency
    :param highcut: scalar
        High pass cutoff frequency
    :param fs: float
        Sampling frequency of the signal
    :param order: int
        The order of the butterworth filter being constructed
    :return: ndarray
        Output of the digital IIR filter
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_energy(filtered_signal):
    """Sums the energies calculated by butter_bandpass_filter for the passed signal.
    Source: https://stackoverflow.com/questions/30659579/calculate-energy-for-each-frequency-band-around-frequency-f-of-interest-in-pytho
    :param filtered_signal: ndarray
        The signal that has been filtered in a particular frequency band by butter_bandpass_filter
    :return: float
        The sum of the energies in the frequency band
    """
    return sum([x*2 for x in filtered_signal])
