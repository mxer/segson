import librosa
import numpy as np

from .util import get

def rms(spectrum, config=dict()):
    '''
    Computes root-mean-square energy feature.

    Parameters
    ----------
    spectrum : np.ndarray [shape=(n_bins, n_frames)]
        Spectrum from which the feature is computed.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. This function use
        no parameters.

    Returns
    -------
    feature : np.ndarray [shape=(n_frames,)]
        Computed root-mean-square energy feature.
    '''

    return librosa.feature.rmse(S=spectrum)[0]

def centroid(spectrum, config=dict()):
    '''
    Computes spectral centroid feature.

    Parameters
    ----------
    spectrum : np.ndarray [shape=(n_bins, n_frames)]
        Spectrum from which the feature is computed.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. This function use
        no parameters.

    Returns
    -------
    feature : np.ndarray [shape=(n_frames,)]
        Computed spectral centroid feature.
    '''

    freq = None

    spectrum_type = get(config, 'spectrum.type')
    if spectrum_type == 'cqt':
        freq = librosa.cqt_frequencies(get(config, 'spectrum.n_bins'), fmin=librosa.note_to_hz('C1'))
    elif spectrum_type == 'mel':
        freq = librosa.mel_frequencies(n_mels=get(config, 'spectrum.n_bins'), htk=True)

    return librosa.feature.spectral_centroid(S=spectrum, freq=freq)[0]

def flux(spectrum, config=dict()):
    '''
    Computes spectral flux feature.

    Parameters
    ----------
    spectrum : np.ndarray [shape=(n_bins, n_frames)]
        Spectrum from which the feature is computed.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'feature.flux.normalization'

    Returns
    -------
    feature : np.ndarray [shape=(n_frames,)]
        Computed spectral flux feature.
    '''

    normalization = get(config, 'feature.flux.normalization')

    if normalization == 'peak':
        spectrum_max = np.max(spectrum, axis=0)

        # replace zeros with a threshold
        spectrum_max = np.maximum(spectrum_max, 1e-10)

        diff = spectrum[:, 1:] / spectrum_max[1:] - spectrum[:, :-1] / spectrum_max[:-1]
    elif normalization is None:
        diff = spectrum[:, 1:] - spectrum[:, :-1]
    else:
        raise ValueError('Parameter `feature.flux.normalization` must be either "peak" or None')

    spectral_flux = np.sum(np.square(diff), axis=0)

    # flux(0) := 0
    return np.insert(spectral_flux, 0, 0.0, axis=0)

def flatness(spectrum, config=dict()):
    '''
    Computes spectral flatness feature.

    Parameters
    ----------
    spectrum : np.ndarray [shape=(n_bins, n_frames)]
        Spectrum from which the feature is computed.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. This function use
        no parameters.

    Returns
    -------
    feature : np.ndarray [shape=(n_frames,)]
        Computed spectral flatness feature.
    '''

    return librosa.feature.spectral_flatness(S=spectrum)[0]
