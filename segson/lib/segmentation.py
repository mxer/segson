import numpy as np
from scipy.stats import multivariate_normal
import librosa

from .util import time_to_frames, frames_to_time, build_feature_matrix, get_audio_length, get
from .normalization import compose

def llr(rms, centroid, flux, flatness, config=dict()):
    '''
    Computes LLR distance curve for given features. Output sequence is normalized using outlier removal and moving
    average.

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

        * 'segmentation.llr.window_length'
        * 'segmentation.llr.window_overlap'

    Returns
    -------
    llr : np.ndarray [shape=(n_frames,)]
        Computed LLR distance curve for given features.
    '''

    X = build_feature_matrix(rms, centroid, flux, flatness)

    window_length = get(config, 'segmentation.llr.window_length')
    window_overlap = get(config, 'segmentation.llr.window_overlap')
    window_shift = window_length - window_overlap

    if window_shift <= 0:
        window_shift = window_length / 2

    log_likelihood_ratio = np.zeros((len(X),))

    for start in np.arange(0, frames_to_time(len(X), config) - window_length, window_shift):
        split = time_to_frames(start + window_length / 2, config)
        end = time_to_frames(start + window_length, config)
        start = time_to_frames(start, config)
        A = X[start:split]
        B = X[split:end]
        C = X[start:end]

        # fit normal distributions
        N_A = multivariate_normal(np.mean(A, axis=0), np.cov(A.T), allow_singular=True)
        N_B = multivariate_normal(np.mean(B, axis=0), np.cov(B.T), allow_singular=True)
        N = multivariate_normal(np.mean(C, axis=0), np.cov(C.T), allow_singular=True)

        # likelihood of hypothesis that there is no change
        L0 = np.sum(N.logpdf(A)) + np.sum(N.logpdf(B))

        # likelihood of hypothesis that there is a change
        L1 = np.sum(N_A.logpdf(A)) + np.sum(N_B.logpdf(B))

        log_likelihood_ratio[start:end] = L1 - L0

    log_likelihood_ratio = compose(log_likelihood_ratio, config)

    return log_likelihood_ratio

def get_boundaries(x, config=dict()):
    '''
    Identifies segment boundaries in audio using given distance curve.

    Parameters
    ----------
    x : np.ndarray [shape=(n_frames,)]
        Distance curve computed by a segmentation method.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'segmentation.peaks.pre_max'
        * 'segmentation.peaks.post_max'
        * 'segmentation.peaks.pre_avg'
        * 'segmentation.peaks.post_avg'
        * 'segmentation.peaks.wait'
        * 'segmentation.peaks.scale_delta'

    Returns
    -------
    boundaries : np.ndarray [shape=(n_boundaries,)]
        List of identified boundaries in seconds.
    '''

    pre_max = time_to_frames(get(config, 'segmentation.peaks.pre_max'), config)
    post_max = time_to_frames(get(config, 'segmentation.peaks.post_max'), config)
    pre_avg = time_to_frames(get(config, 'segmentation.peaks.pre_avg'), config)
    post_avg = time_to_frames(get(config, 'segmentation.peaks.post_avg'), config)
    wait = time_to_frames(get(config, 'segmentation.peaks.wait'), config)

    scale_delta = get(config, 'segmentation.peaks.scale_delta')
    delta = scale_delta * np.std(x)

    # find peaks represented by frame indices
    peaks = librosa.util.peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait)

    # convert frame indices into seconds
    boundaries = np.asarray(np.round(frames_to_time(peaks, config)), dtype=int)

    # add very beginning and very end to boundaries array
    boundaries = np.concatenate(([0], boundaries, [get_audio_length(frames=x, config=config)]))

    # if peak detection parameters are set to low values, multiple peaks can fall into one second
    # filter duplicates
    boundaries = np.unique(boundaries)

    return boundaries
