import librosa

from .util import get

def compute(audio, config=dict()):
    '''
    Computes a spectrum from given audio signal. Technique is chosen according to `spectrum.type` parameter in config.

    Parameters
    ----------
    audio : np.ndarray [shape=(n_samples,)]
        Audio signal.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'spectrum.type'

        For other parameters used in spectrum functions themselves, check their description.

    Returns
    -------
    spectrum : np.ndarray [shape=(n_bins, n_samples)]
        Computed spectrum.
    '''

    spectrum = get(config, 'spectrum.type')

    if spectrum == 'stft':
        return stft(audio, config)
    elif spectrum == 'cqt':
        return cqt(audio, config)
    elif spectrum == 'mel':
        return mel(audio, config)
    else:
        raise ValueError('Only "stft", "cqt", or "mel" are supported')

def stft(audio, config=dict()):
    '''
    Computes short-time Fourier transform spectrum.

    Parameters
    ----------
    audio : np.ndarray [shape=(n_samples,)]
        Audio signal.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'spectrum.hop_length'

    Returns
    -------
    spectrum : np.ndarray [shape=(1024, n_samples)]
        Computed spectrum.
    '''

    hop_length = get(config, 'spectrum.hop_length')
    return librosa.magphase(librosa.stft(y=audio, n_fft=hop_length * 4))[0]

def cqt(audio, config=dict()):
    '''
    Computes constant-Q transform spectrum.

    Parameters
    ----------
    audio : np.ndarray [shape=(n_samples,)]
        Audio signal.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'
        * 'spectrum.n_bins'

    Returns
    -------
    spectrum : np.ndarray [shape=(n_bins, n_samples)]
        Computed spectrum.
    '''

    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')
    n_bins = get(config, 'spectrum.n_bins')
    return librosa.magphase(librosa.cqt(y=audio, sr=sampling_rate, hop_length=hop_length, n_bins=n_bins))[0]

def mel(audio, config=dict()):
    '''
    Computes melspectrogram.

    Parameters
    ----------
    audio : np.ndarray [shape=(n_samples,)]
        Audio signal.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'spectrum.hop_length'
        * 'spectrum.n_bins'

    Returns
    -------
    spectrum : np.ndarray [shape=(n_bins, n_samples)]
        Computed spectrum.
    '''

    sampling_rate = get(config, 'audio.sampling_rate')
    hop_length = get(config, 'spectrum.hop_length')
    n_bins = get(config, 'spectrum.n_bins')
    return librosa.feature.melspectrogram(y=audio, sr=sampling_rate, hop_length=hop_length, n_mels=n_bins, htk=True)
