import numpy as np
from sklearn.neighbors import NearestNeighbors

from .util import build_feature_matrix, time_to_frames, get_audio_length, rolling_window, get
from .normalization import outlier_removal

def classify(rms, centroid, flux, flatness, boundaries, config=dict()):
    '''
    Classifies segments defined by given boundaries and processes the prediction to return song time annotations.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    centroid : np.ndarray [shape=(n_frames,)]
        Spectral centroid feature.

    flux : np.ndarray [shape=(n_frames,)]
        Spectral flux feature.

    flatness : np.ndarray [shape=(n_frames,)]
        Spectral flatness feature.

    boundaries : np.ndarray [shape=(n_boundaries,)]
        Spectral flatness feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'classification.model'

        For other parameters used in classification functions themselves, check their description.

    Returns
    -------
    songs : np.ndarray [shape=(n_songs,)]
        List of estimated songs with their start and end time annotations in seconds.
    '''

    model = get(config, 'classification.model')

    rms = _aggregate(rms, config)
    centroid = _aggregate(centroid, config)
    flux = _aggregate(flux, config)
    flatness = _aggregate(flatness, config)

    if model == 'ensemble':
        predicted = ensemble(rms, centroid, flux, flatness, config)
    elif model == 'silence_detection':
        predicted = silence_detection(rms, config)
    elif model == 'rule_based':
        predicted = rule_based(rms, centroid, flux, flatness, config)
    elif model == 'anomaly_detection':
        predicted = anomaly_detection(rms, centroid, flux, flatness, config)
    # kind of hidden option
    elif model == 'feature_evaluation':
        predicted = evaluate_feature(rms, centroid, flux, flatness, config)
    else:
        raise ValueError('classification model must be one of ' +
                         '"ensemble", "silence_detection", "rule_based", or "anomaly_detection"')

    songs = _process_predicted(predicted, boundaries)

    return songs

def silence_detection(rms, config=dict()):
    '''
    Classifies frames using Silence detection model.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'classification.silence_detection.scale'

    Returns
    -------
    predictions : np.ndarray [shape=(n_frames,), dtype=bool]
        Predictions of frames, where `True` means 'in song', whereas `False` means 'not in song'.
    '''

    scale = get(config, 'classification.silence_detection.scale')
    threshold = scale * np.mean(rms)

    predicted = rms >= threshold
    return predicted

def rule_based(rms, centroid, flux, flatness, config=dict()):
    '''
    Classifies frames using Rule based model.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    centroid : np.ndarray [shape=(n_frames,)]
        Spectral centroid feature.

    flux : np.ndarray [shape=(n_frames,)]
        Spectral flux feature.

    flatness : np.ndarray [shape=(n_frames,)]
        Spectral flatness feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'classification.window_length'
        * 'classification.rule_based.scale_rms'
        * 'classification.rule_based.scale_centroid'
        * 'classification.rule_based.scale_flux'
        * 'classification.rule_based.scale_flatness'

    Returns
    -------
    predictions : np.ndarray [shape=(n_frames,), dtype=bool]
        Predictions of frames, where `True` means 'in song', whereas `False` means 'not in song'.
    '''

    window_length = get(config, 'classification.window_length')
    window = _create_window(window_length)

    def estimate(x, thresholds, g):
        confidence = g(x, thresholds)
        return np.abs(confidence) * np.sign(confidence)

    rms_thresholds = _get_windowed_thresholds(rms, window, get(config, 'classification.rule_based.scale_rms'))
    centroid_thresholds = _get_windowed_thresholds(centroid, window, get(config, 'classification.rule_based.scale_centroid'))
    flux_thresholds = _get_windowed_thresholds(flux, window, get(config, 'classification.rule_based.scale_flux'))
    flatness_thresholds = _get_windowed_thresholds(flatness, window, get(config, 'classification.rule_based.scale_flatness'))

    rms_estimated = estimate(rms, rms_thresholds, _positive)
    centroid_estimated = estimate(centroid, centroid_thresholds, _negative)
    flux_estimated = estimate(flux, flux_thresholds, _positive)
    flatness_estimated = estimate(flatness, flatness_thresholds, _negative)

    combined = rms_estimated + centroid_estimated + flux_estimated + flatness_estimated

    predicted = combined > 0
    return predicted

def anomaly_detection(rms, centroid, flux, flatness, config=dict()):
    '''
    Classifies frames using Anomaly detection model.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    centroid : np.ndarray [shape=(n_frames,)]
        Spectral centroid feature.

    flux : np.ndarray [shape=(n_frames,)]
        Spectral flux feature.

    flatness : np.ndarray [shape=(n_frames,)]
        Spectral flatness feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'classification.anomaly_detection.k'
        * 'classification.anomaly_detection.scale'
        * 'classification.window_length'

    Returns
    -------
    predictions : np.ndarray [shape=(n_frames,), dtype=bool]
        Predictions of frames, where `True` means 'in song', whereas `False` means 'not in song'.
    '''

    X = build_feature_matrix(rms, centroid, flux, flatness)

    k = get(config, 'classification.anomaly_detection.k')
    scale = get(config, 'classification.anomaly_detection.scale')
    window_length = get(config, 'classification.window_length')

    window = _create_window(window_length)

    knn = NearestNeighbors(n_neighbors=k)

    knn.fit(X)
    distances = knn.kneighbors(X, return_distance=True)[0][:, -1]
    distances = outlier_removal(distances, config)

    thresholds = _get_windowed_thresholds(distances, window, scale)

    predicted = distances < thresholds

    return predicted

def ensemble(rms, centroid, flux, flatness, config=dict()):
    '''
    Classifies frames using ensemble model consisting of all models in this module.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    centroid : np.ndarray [shape=(n_frames,)]
        Spectral centroid feature.

    flux : np.ndarray [shape=(n_frames,)]
        Spectral flux feature.

    flatness : np.ndarray [shape=(n_frames,)]
        Spectral flatness feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. For parameters
        used in classification functions themselves, check their description.

    Returns
    -------
    predictions : np.ndarray [shape=(n_frames,), dtype=bool]
        Predictions of frames, where `True` means 'in song', whereas `False` means 'not in song'.
    '''

    model1 = np.asarray(silence_detection(rms, config), dtype=int)
    model2 = np.asarray(rule_based(rms, centroid, flux, flatness, config), dtype=int)
    model3 = np.asarray(anomaly_detection(rms, centroid, flux, flatness, config), dtype=int)

    votes = model1 + model2 + model3

    predicted = votes >= 2

    return predicted

def evaluate_feature(rms, centroid, flux, flatness, config=dict()):
    '''
    Classifies frames using the same approach as in Silence detection model, however evaluates one given feature that
    does not need to be RMS.

    Parameters
    ----------
    rms : np.ndarray [shape=(n_frames,)]
        Root-mean-square energy feature.

    centroid : np.ndarray [shape=(n_frames,)]
        Spectral centroid feature.

    flux : np.ndarray [shape=(n_frames,)]
        Spectral flux feature.

    flatness : np.ndarray [shape=(n_frames,)]
        Spectral flatness feature.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'classification.rule_based.scale_rms'
        * 'classification.rule_based.scale_centroid'
        * 'classification.rule_based.scale_flux'
        * 'classification.rule_based.scale_flatness'

    Returns
    -------
    predictions : np.ndarray [shape=(n_frames,), dtype=bool]
        Predictions of frames, where `True` means 'in song', whereas `False` means 'not in song'.
    '''

    feature = get(config, 'feature.evaluation')

    if feature == 'rms':
        x = rms
        scale = get(config, 'classification.rule_based.scale_rms')
        compare = 'greater'
    elif feature == 'centroid':
        x = centroid
        scale = get(config, 'classification.rule_based.scale_centroid')
        compare = 'less'
    elif feature == 'flux':
        x = flux
        scale = get(config, 'classification.rule_based.scale_flux')
        compare = 'greater'
    elif feature == 'flatness':
        x = flatness
        scale = get(config, 'classification.rule_based.scale_flatness')
        compare = 'less'
    else:
        raise ValueError('feature for evaluation must be one of "rms", "centroid", "flux", or "flatness"')

    threshold = scale * np.mean(x)

    if compare == 'greater':
        predicted = x >= threshold
    else:
        predicted = x <= threshold

    return predicted

def _aggregate(x, config):
    indices = time_to_frames(range(get_audio_length(frames=x, config=config)), config)

    aggregated = []

    for (start, end) in rolling_window(indices, 2):
        aggregated.append(np.mean(x[start:end]))

    aggregated.append(np.mean(x[end:]))

    return np.array(aggregated)

def _process_predicted(predicted, boundaries):
    if boundaries is not None:
        output = []
        for segment in rolling_window(boundaries, 2):
            frames = predicted[segment[0]:segment[1]]
            # classify segment by majority voting
            output.append(np.count_nonzero(frames) / len(frames) >= 0.5)
        return _merge_predicted(output, boundaries)
    else:
        return _merge_predicted(predicted)

def _merge_predicted(predicted, boundaries=None):
    output = []
    start = None

    if boundaries is not None:
        # number of segments is one less than number of boundaries
        for index in range(len(boundaries) - 1):
            if predicted[index]:
                if start is None:
                    start = boundaries[index]
            elif start is not None:
                output.append((start, boundaries[index]))
                start = None

        if start is not None:
            output.append((start, boundaries[-1]))
    else:
        for index in range(len(predicted)):
            if predicted[index]:
                if start is None:
                    start = index
            elif start is not None:
                output.append((start, index))
                start = None

        if start is not None:
            output.append((start, len(predicted)))

    return output

def _get_windowed_thresholds(x, window, scale):
    thresholds = np.zeros_like(x)

    for index, x_n in enumerate(x):
        thresholds[index] = scale * np.mean(window(x, index))

    return thresholds

def _create_window(window_length, config=dict()):
    if window_length is None:
        return lambda x, n: x
    else:
        frames = time_to_frames(window_length / 2, config)
        return lambda x, n: x[max(n - frames, 0):min(n + frames, len(x) - 1)]

def _positive(x, threshold):
    return x - threshold

def _negative(x, threshold):
    return -(x - threshold)
