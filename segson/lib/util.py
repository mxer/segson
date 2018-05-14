import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa

def time_to_frames(value, config=dict()):
    '''
    Converts time in seconds into frames depending on appropriate config variables.

    Parameters
    ----------
    value : float or np.ndarray [dtype=float]
        Value in seconds.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'

    Returns
    -------
    frames : int or np.ndarray [dtype=int]
        Converted frames.
    '''
    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')
    return librosa.time_to_frames(value, sr=sampling_rate, hop_length=hop_length)

def frames_to_time(value, config=dict()):
    '''
    Converts frames into time in seconds (rounded down) depending on appropriate config variables.

    Parameters
    ----------
    value : int or np.ndarray [dtype=int]
        Value in frames.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'

    Returns
    -------
    time : int or np.ndarray [dtype=int]
        Converted seconds.
    '''
    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')

    if value is float:
        return int(librosa.frames_to_time(value, sr=sampling_rate, hop_length=hop_length))
    else:
        return np.asarray(librosa.frames_to_time(value, sr=sampling_rate, hop_length=hop_length), dtype=int)

def frames_by_seconds(length, config=dict(), step=1):
    '''
    Returns the array of frame indices by time step given in seconds up to length of given framed data.

    Parameters
    ----------
    length : int
        Number of frames.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'

    step : int
        Number of seconds which represents the step between two consecutive frame indices.

    Returns
    -------
    frames : int or np.ndarray [dtype=int]
        Frame indices.
    '''

    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')
    size = int(librosa.frames_to_time(length, sr=sampling_rate, hop_length=hop_length)) + 1
    return librosa.time_to_frames([time for time in range(0, size, step)])

def rolling_window(data, length, shift=1):
    '''
    Returns windowed view into given data. It works for 1D and 2D dimensional data.

    Parameters
    ----------
    data : np.ndarray
        Input data.

    length : int
        Length of windows in number of elements.

    shift : int
        Windowing shift.

    Returns
    -------
    windows : np.ndarray [shape=(n_windows, length[, vector_dim])]
        Computed windows.
    '''

    count = (data.shape[0] - length) // shift + int(data.shape[0] - length <= data.shape[0] // shift)
    shape = (count, length)

    if data.ndim > 1:
        shape = shape + data.shape[1:]

    strides = (shift * data.strides[0],) + data.strides

    return as_strided(data, shape=shape, strides=strides, writeable=False)

def build_feature_matrix(rms, centroid, flux, flatness):
    '''
    Builds 2D numpy array from given features.

    Parameters
    ----------
    rms : np.ndarray [shape=(n,)]
        Root-mean-square energy vector.

    centroid : np.ndarray [shape=(n,)]
        Spectral centroid vector.

    flux : np.ndarray [shape=(n,)]
        Spectral flux vector.

    flatness : np.ndarray [shape=(n,)]
        Spectral flatness vector.

    Returns
    -------
    X : np.ndarray [shape=(n, 4)]
        Matrix built from individual feature vectors.
    '''

    X = np.empty((len(rms), 4))
    X[:, 0] = rms
    X[:, 1] = centroid
    X[:, 2] = flux
    X[:, 3] = flatness
    return X

def get_audio_length(signal=None, frames=None, config=dict()):
    '''
    Computes the audio length in seconds from given either signal or frames vector.

    Parameters
    ----------
    signal : np.ndarray [shape=(n_signal,)] or None
        Signal vector.

    frames : np.ndarray [shape=(n_frames,)] or None
        Frames vector.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'

    Returns
    -------
    time : int
        Number of seconds.
    '''

    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')

    if signal is not None:
        return int(np.round(np.asscalar(librosa.samples_to_time(len(signal), sr=sampling_rate))))
    elif frames is not None:
        return int(np.round(np.asscalar(librosa.frames_to_time(len(frames), sr=sampling_rate, hop_length=hop_length))))
    else:
        raise ValueError('Either `signal` or `frames` parameter must be passed.')

def seconds_to_string(seconds):
    '''
    Converts given number of seconds into time format representation.

    Parameters
    ----------
    seconds : int
        Number of seconds.

    Returns
    -------
    time : string
        String time format.
    '''

    s = seconds % 60
    m = (seconds // 60) % 60
    h = (seconds // 60 // 60) % 60
    return '{}:{:02}:{:02}'.format(h, m, s)

def get(config, key):
    '''
    Gets parameter value from config, or default value if it is not present.

    Parameters
    ----------
    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file.

    key : string
        Parameter name.

    Returns
    -------
    value : any
        Paremeter value from config or its default value.
    '''

    defaults = {
        'audio.sampling_rate': 22050,
        'audio.offset': 0.0,
        'audio.duration': None,
        'spectrum.type': 'cqt',
        'spectrum.hop_length': 512,
        'spectrum.n_bins': {
            'stft': 1025,
            'cqt': 84,
            'mel': 128,
        },
        'feature.flux.normalization': None,
        'feature.evaluation': 'rms',
        'normalization.techniques': ['outlier_removal', 'moving_average', 'min_max'],
        'normalization.moving_average.window_length': 2,
        'segmentation.llr.window_length': 2,
        'segmentation.llr.window_overlap': 1,
        'segmentation.peaks.pre_max': 15,
        'segmentation.peaks.post_max': 5,
        'segmentation.peaks.pre_avg': 10,
        'segmentation.peaks.post_avg': 2,
        'segmentation.peaks.wait': 0,
        'segmentation.peaks.scale_delta': 1,
        'classification.model': 'ensemble',
        'classification.window_length': 40,
        'classification.silence_detection.scale': {
            'stft': 1.0,
            'cqt': 0.9,
            'mel': 1.0,
        },
        'classification.rule_based.scale_rms': {
            'stft': 1.3,
            'cqt': 1.1,
            'mel': 1.1,
        },
        'classification.rule_based.scale_centroid': {
            'stft': 1.3,
            'cqt': 1.1,
            'mel': 1.3,
        },
        'classification.rule_based.scale_flux': 1.1,
        'classification.rule_based.scale_flatness': {
            'stft': 1.3,
            'cqt': 1.3,
            'mel': 1.8,
        },
        'classification.anomaly_detection.scale': 1,
        'classification.anomaly_detection.k': 50,
        'postprocessing.minimum_threshold': 30,
        'postprocessing.closest_neighbor': 10,
    }

    default = defaults[key]

    spectrum_type = config.get('spectrum.type', defaults['spectrum.type'])
    if isinstance(default, dict):
        default = default[spectrum_type]

    return config.get(key, default)
